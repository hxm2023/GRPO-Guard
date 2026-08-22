"""Online fault-matrix runs on autodl2: F1-F8 injected into REAL server
rollouts (design doc §11).

For each family the injection targets a GenerationEvent produced by the
REAL vLLM server in this run (authoritative token ids + service logprobs),
and the validator decides in the live closed-loop context.  Outputs:
  - <out>/fault_matrix_online.json      (F1-F4)
  - <out>/fault_matrix_f58_online.json  (F5-F8)

Usage (on autodl2):
  GRPO_GUARD_MODEL_PATH=... GRPO_GUARD_OUT=... \
    python examples/countdown/online_matrix.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/online_matrix_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8002"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51217"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
MAX_COMPLETION = 64

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

# module-level state shared by helpers (set in main)
store = None
log_ = None
epoch = None
run_id = "online"
protocol = None
ckpt_sha = "1371204785c657d6138cfd25ea0516b1cc0a45b1cad205c2ec250f01ce3f6c3a"
tokenizer_sha = "tok-sha-online"
template_sha = "tpl-sha-online"
sampling_sha = "samp-sha-online"
split_train: dict = {}
split_held: dict = {}


def log(msg: str) -> None:
    print(f"[online-matrix] {msg}", flush=True)


def next_lifecycle() -> int:
    return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1


def make_envelope(gen, stage="pre_reward", reward_event=None, parent_identity=None):
    from grpo_guard.schema.artifacts import EventRef, ManifestRef
    from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
    from grpo_guard.store.canonical_json import canonical_sha256

    return TrajectoryEnvelope(
        envelope_id=f"env-{gen.event_id}-{stage}",
        envelope_stage=stage, run_id=run_id, request_id=gen.request_id,
        generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
        scoring_event=None,
        reward_event=reward_event,
        policy_manifest=ManifestRef(uri="", manifest_id="pm-0", sha256=ckpt_sha),
        split_manifest=ManifestRef(uri="", manifest_id="split-train",
                                   sha256=canonical_sha256(split_train)),
        parent_identity_decision=parent_identity,
        training_contract=TrainingContract(
            protocol="strict_on_policy", trainer_parent_policy_version=0,
            consuming_update_id="update-1", max_policy_lag_versions=0,
            behavior_logprob_source="generation_service",
            authoritative_behavior_logprob_event=EventRef(
                uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
            diagnostic_non_authoritative_logprobs_allowed=False,
        ),
    ).seal()


def trajectory(gen):
    """Rebuild a Trajectory from a real generation event + its artifacts."""
    from grpo_guard import testing
    from grpo_guard.schema.manifests import PolicyManifest, SplitManifest

    seq = np.frombuffer(store.get(gen.sequence_token_ids), dtype=np.int32).copy()
    target = np.frombuffer(store.get(gen.completion_target_mask), dtype=np.int8).copy()
    loss = np.frombuffer(store.get(gen.loss_mask), dtype=np.int8).copy()
    lp = np.frombuffer(store.get(gen.service_behavior_logprobs), dtype=np.float32).copy()
    events = {}
    for e in log_.iterate():
        from grpo_guard.schema.events import event_from_payload

        events[e["event_id"]] = event_from_payload(e)
    events.setdefault(gen.event_id, gen)
    return testing.Trajectory(
        run_id=run_id, events=events,
        policy_manifest=PolicyManifest(manifest_id="pm-0", model_id="Qwen/Qwen3-4B",
                                       model_revision="r", policy_version=0, weights=[],
                                       checkpoint_manifest_sha256=ckpt_sha,
                                       tokenizer_sha256=tokenizer_sha, chat_template_sha256=template_sha,
                                       precision="bf16", adapter_kind="full",
                                       code_commit_sha="c", config_sha256=sampling_sha),
        split_manifest=SplitManifest(**split_train),
        envelope=make_envelope(gen), store=store, sequence=seq,
        target_mask=target, loss_mask=loss, logprobs=lp,
        completion_text="", goal=0, target_numbers=[], reward_components={},
        sequence_ref=gen.sequence_token_ids,
    )


def validate_identity(t):
    from grpo_guard.validators.context import ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=protocol,
    )
    return validate_envelope(ctx, "identity_pre_reward").decision_payload


def validate_full(t, update_input=None):
    from grpo_guard.validators.context import ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=protocol,
        split_registry=getattr(t, "split_registry", None),
        eval_protocol_sha256=getattr(t, "eval_protocol_sha256", None),
        update_input_event=update_input,
    )
    return validate_envelope(ctx, "full_pre_update").decision_payload


def f6_trajectory(gen, p, text, eval_proto):
    """pre_update trajectory with a REAL reward computed from the server's
    completion text, then the evaluator-alias injection."""
    from grpo_guard.adapters.countdown_reward import countdown_rule_verifier, reward_protocol_sha256
    from grpo_guard.faults.f5_f8 import inject_f6_evaluator_alias
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.decisions import ValidationDecision
    from grpo_guard.schema.events import RewardEvent, ValidationDecisionEvent

    t = trajectory(gen)
    r = countdown_rule_verifier(text, p["target_numbers"], p["goal"])
    reward = RewardEvent(
        event_id=f"reward-{gen.event_id}-online", event_type="reward_finished",
        run_id=t.run_id, component_id="countdown_reward",
        lifecycle_seq=gen.lifecycle_seq + 1, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
        reward_version="countdown-rule-v1", evaluator_protocol_sha256=reward_protocol_sha256(),
        source_generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
        components=r, terminal_status="success", latency_ms=0.0,
    ).seal()
    t.events[reward.event_id] = reward

    parent = ValidationDecisionEvent(
        event_id=f"vdec-{gen.event_id}-parent", event_type="validation_decision",
        run_id=t.run_id, component_id="validator",
        lifecycle_seq=gen.lifecycle_seq + 2, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        decision_payload=ValidationDecision(decision="allow", validation_stage="identity_pre_reward",
                                            reason_codes=["G001"]),
    ).seal()
    t.events[parent.event_id] = parent

    from grpo_guard.schema.artifacts import EventRef as ER

    t.envelope = make_envelope(gen, stage="pre_update",
                               reward_event=ER(uri="", event_id=reward.event_id,
                                               event_sha256=reward.event_sha256),
                               parent_identity=ER(uri="", event_id=parent.event_id,
                                                  event_sha256=parent.event_sha256))
    return inject_f6_evaluator_alias(t, eval_proto)


def f8_trajectory(gen):
    """F8 with an ISOLATED store clone so the shared evidence is untouched."""
    from grpo_guard import testing
    from grpo_guard.faults.f5_f8 import inject_f8_artifact_mutation
    from grpo_guard.store.artifact_store import ArtifactStore

    t = trajectory(gen)
    clone = Path(tempfile.mkdtemp(prefix="grpo-guard-f8-online-"))
    shutil.copytree(t.store.root, clone, dirs_exist_ok=True)
    t2 = testing.Trajectory(
        run_id=t.run_id, events=t.events, policy_manifest=t.policy_manifest,
        split_manifest=t.split_manifest, envelope=t.envelope, store=ArtifactStore(clone),
        sequence=t.sequence, target_mask=t.target_mask, loss_mask=t.loss_mask,
        logprobs=t.logprobs, completion_text=t.completion_text, goal=t.goal,
        target_numbers=t.target_numbers, reward_components=t.reward_components,
        sync_events=t.sync_events, sequence_ref=t.sequence_ref,
    )
    return inject_f8_artifact_mutation(t2)


def main() -> int:
    global store, log_, epoch, run_id, protocol, split_train, split_held

    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server
    from grpo_guard.adapters.countdown_reward import reward_protocol_sha256
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.events import UpdateInputEvent
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore
    from grpo_guard.validators.context import ProtocolConfig

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT_DIR / "events", ignore_errors=True)
    shutil.rmtree(OUT_DIR / "store", ignore_errors=True)

    run_id = f"online-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-online")
    epoch = log_.acquire_lease()
    protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")

    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    server = start_server(OUT_DIR / "vllm_server.log")
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)
        runtime.set_load_epoch(1)

        prompts = []
        for i in range(8):
            tgt = [i % 7 + 1, (i * 2) % 7 + 1, (i * 3) % 7 + 1]
            goal = (tgt[0] + tgt[1]) * tgt[2] % 40 + 1
            prompts.append({
                "text": f"Use the numbers {tgt} exactly once to reach {goal}.\nReturn only the arithmetic expression.",
                "target_numbers": tgt, "goal": goal, "prompt_id": f"countdown-{i:04d}",
            })
        split_train = {"split_id": "split-train", "split_name": "train",
                       "prompt_ids": [p["prompt_id"] for p in prompts]}
        split_held = {"split_id": "split-held_out", "split_name": "held_out",
                      "prompt_ids": [prompts[0]["prompt_id"]]}

        sync_ref = EventRef(uri="", event_id="sync-canary", event_sha256="0" * 64)
        gens = []
        for p in prompts[:4]:
            for g in range(2):
                res = client.generate([p["text"]], n=1, temperature=1.0, top_p=1.0,
                                      top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                pid, cid, lps = res["prompt_ids"], res["completion_ids"], res["logprobs"]
                text = tokenizer.decode(cid[0], skip_special_tokens=True)
                gen = runtime.emit_generation(
                    pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
                    behavior_policy_version=0, checkpoint_manifest_sha256=ckpt_sha,
                    sync_event=sync_ref, tokenizer_sha256=tokenizer_sha,
                    chat_template_sha256=template_sha, sampling_config_sha256=sampling_sha,
                    prompt_id=p["prompt_id"], request_id=f"req-{p['prompt_id']}-{g}",
                    required_epoch=epoch,
                )
                gens.append((gen, p, text))
        log(f"real rollouts: {len(gens)} GenerationEvents")

        # ---- F1-F4 online injections on the FIRST real generation ----------
        from grpo_guard.faults import (
            inject_f1_static_rollout,
            inject_f2_misbound_logprob,
            inject_f3_retokenization,
            inject_f4_mask_shift,
        )

        base = trajectory(gens[0][0])
        f14_results = []
        for case_id, inject in [
            ("f1_static_rollout", lambda: inject_f1_static_rollout(base, 0, 1)),
            ("f2_misbound_logprob", lambda: inject_f2_misbound_logprob(base, 1)),
            ("f3_retokenization", lambda: inject_f3_retokenization(base)),
            ("f4_mask_shift", lambda: inject_f4_mask_shift(base, 1)),
        ]:
            d = validate_identity(inject())
            f14_results.append({"case_id": case_id, "decision": d.decision,
                                "reason_codes": d.reason_codes})
            log(f"F1-F4 {case_id}: {d.decision} {d.reason_codes[:2]}")

        # ---- F5-F8 online injections ----------------------------------------
        from grpo_guard.faults.f5_f8 import (
            inject_f5_split_leakage,
            inject_f7_event_reorder,
        )

        f58_results = []
        gen0, p0, text0 = gens[0]

        t5 = inject_f5_split_leakage(trajectory(gen0), "held_out")
        d = validate_full(t5)
        f58_results.append({"case_id": "f5_split_leakage", "decision": d.decision,
                            "reason_codes": d.reason_codes})
        log(f"F5-F8 f5_split_leakage: {d.decision} {d.reason_codes[:2]}")

        t6 = f6_trajectory(gen0, p0, text0, reward_protocol_sha256())
        d = validate_full(t6)
        f58_results.append({"case_id": "f6_evaluator_alias", "decision": d.decision,
                            "reason_codes": d.reason_codes})
        log(f"F5-F8 f6_evaluator_alias: {d.decision} {d.reason_codes[:2]}")

        t7 = inject_f7_event_reorder(trajectory(gen0))
        gen = t7.events[t7.envelope.generation_event.event_id]
        upd = UpdateInputEvent(
            event_id=f"uinput-{t7.envelope.envelope_id}", run_id=t7.run_id,
            component_id="materializer", lifecycle_seq=gen.lifecycle_seq + 100,
            created_at_utc=gen.created_at_utc, update_id="update-1",
            preupdate_envelope=t7.envelope.ref(),
            preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
            sequence_token_ids=gen.sequence_token_ids, loss_mask=gen.loss_mask,
            authoritative_behavior_logprob_event=t7.envelope.training_contract.authoritative_behavior_logprob_event,
            authoritative_behavior_logprobs=gen.service_behavior_logprobs,
            reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
            materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
            tokenizer_called=False,
        ).seal()
        d = validate_full(t7, update_input=upd)
        f58_results.append({"case_id": "f7_event_reorder", "decision": d.decision,
                            "reason_codes": d.reason_codes})
        log(f"F5-F8 f7_event_reorder: {d.decision} {d.reason_codes[:2]}")

        t8 = f8_trajectory(gen0)
        d = validate_full(t8)
        f58_results.append({"case_id": "f8_artifact_mutation", "decision": d.decision,
                            "reason_codes": d.reason_codes})
        log(f"F5-F8 f8_artifact_mutation: {d.decision} {d.reason_codes[:2]}")

        # ---- normal set: real generations must stay ALLOW -------------------
        normals = []
        for gen, _, _ in gens[:4]:
            d = validate_identity(trajectory(gen))
            normals.append({"generation": gen.event_id, "decision": d.decision})

        (OUT_DIR / "fault_matrix_online.json").write_text(json.dumps({
            "run_id": run_id, "scope": "F1-F4 online (real server rollouts)",
            "source": "autodl2 vLLM server",
            "results": f14_results, "normal": normals,
            "summary": {
                "matched": sum(1 for r in f14_results if r["decision"] == "reject"),
                "total": len(f14_results),
                "normal_allow": sum(1 for n in normals if n["decision"] == "allow"),
                "normal_total": len(normals),
            },
        }, indent=2), encoding="utf-8")

        (OUT_DIR / "fault_matrix_f58_online.json").write_text(json.dumps({
            "run_id": run_id, "scope": "F5-F8 online (v0.2-preview)",
            "source": "autodl2 vLLM server",
            "results": f58_results,
            "summary": {
                "matched": sum(1 for r in f58_results if r["decision"] in ("reject", "quarantine")),
                "total": len(f58_results),
            },
        }, indent=2), encoding="utf-8")
        log("ONLINE MATRIX DONE")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
