"""Batch online experiment on autodl2 (design doc §11, §13; decision D9).

One GPU session:
  1. 256 REAL rollouts (64 prompts x 4 gens) via trl vllm-serve;
  2. F1-F8 (canonical + variant) injected into EVERY v0 generation —
     decisions must be family-consistent across generations;
  3. normal set: all 32 real generations must validate ALLOW;
  4. online validator timing per envelope (guard overhead in the live
     closed-loop context).

Outputs: <out>/batch_online_matrix.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/batch_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8006"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51221"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
MAX_COMPLETION = 64

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

ckpt_sha = "1371204785c657d6138cfd25ea0516b1cc0a45b1cad205c2ec250f01ce3f6c3a"
tokenizer_sha = "tok-sha-batch"
template_sha = "tpl-sha-batch"
sampling_sha = "samp-sha-batch"

store = None
log_ = None
epoch = None
run_id = "batch"


def log(msg: str) -> None:
    print(f"[batch] {msg}", flush=True)


def next_lifecycle() -> int:
    return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1


def main() -> int:
    global store, log_, epoch, run_id

    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server
    from grpo_guard.adapters.countdown_reward import countdown_rule_verifier, reward_protocol_sha256
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.schema.artifacts import EventRef, ManifestRef
    from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
    from grpo_guard.schema.events import SyncEvent
    from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore
    from grpo_guard.store.canonical_json import canonical_sha256
    from grpo_guard.validators.context import ProtocolConfig, ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT_DIR / "events", ignore_errors=True)
    shutil.rmtree(OUT_DIR / "store", ignore_errors=True)

    run_id = f"batch-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-batch")
    epoch = log_.acquire_lease()
    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")

    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=0.35)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)
        runtime.set_load_epoch(1)

        canary = SyncEvent(
            event_id="sync-batch-canary", event_type="canary_passed",
            run_id=run_id, component_id="trl_control", lifecycle_seq=next_lifecycle(),
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sync_id="sync-batch", attempt=1, lease_epoch=epoch,
            idempotency_key=f"{run_id}:0:rollout-gpu1",
            source_policy_version=0, source_checkpoint_manifest_sha256=ckpt_sha,
            target_runtime_id="rollout-gpu1", observed_runtime_load_epoch=1,
            observed_policy_version=0, upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="batch", status_detail="batch online canary",
        ).seal()
        log_.append(canary, required_epoch=epoch)
        sync_ref = EventRef(uri="", event_id=canary.event_id, event_sha256=canary.event_sha256)

        prompts = []
        for i in range(64):
            tgt = [i % 7 + 1, (i * 2) % 7 + 1, (i * 3) % 7 + 1]
            goal = (tgt[0] + tgt[1]) * tgt[2] % 40 + 1
            prompts.append({
                "text": f"Use the numbers {tgt} exactly once to reach {goal}.\nReturn only the arithmetic expression.",
                "target_numbers": tgt, "goal": goal, "prompt_id": f"countdown-{i:04d}",
            })
        split_train = {"split_id": "split-train", "split_name": "train",
                       "prompt_ids": [p["prompt_id"] for p in prompts]}

        gens = []
        for p in prompts:
            for g in range(4):
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
        log(f"real rollouts: {len(gens)} GenerationEvents (64 prompts x 4 gens)")

        def events_dict():
            from grpo_guard.schema.events import event_from_payload

            return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

        def envelope_for(gen, stage="pre_reward", reward_event=None, parent_identity=None):
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
            from grpo_guard import testing

            seq = np.frombuffer(store.get(gen.sequence_token_ids), dtype=np.int32).copy()
            target = np.frombuffer(store.get(gen.completion_target_mask), dtype=np.int8).copy()
            loss = np.frombuffer(store.get(gen.loss_mask), dtype=np.int8).copy()
            lp = np.frombuffer(store.get(gen.service_behavior_logprobs), dtype=np.float32).copy()
            return testing.Trajectory(
                run_id=run_id, events=events_dict(),
                policy_manifest=PolicyManifest(manifest_id="pm-0", model_id="Qwen/Qwen3-4B",
                                               model_revision="r", policy_version=0, weights=[],
                                               checkpoint_manifest_sha256=ckpt_sha,
                                               tokenizer_sha256=tokenizer_sha,
                                               chat_template_sha256=template_sha,
                                               precision="bf16", adapter_kind="full",
                                               code_commit_sha="c", config_sha256=sampling_sha),
                split_manifest=SplitManifest(**split_train),
                envelope=envelope_for(gen), store=store, sequence=seq,
                target_mask=target, loss_mask=loss, logprobs=lp,
                completion_text="", goal=0, target_numbers=[], reward_components={},
                sequence_ref=gen.sequence_token_ids,
            )

        def validate_identity(t):
            ctx = ValidationContext(
                envelope=t.envelope, store=t.store, events=t.events,
                policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
                protocol=protocol,
            )
            t0 = time.perf_counter()
            d = validate_envelope(ctx, "identity_pre_reward").decision_payload
            return d, (time.perf_counter() - t0) * 1000.0

        def validate_full(t, update_input=None, split_registry=None, eval_proto=None):
            ctx = ValidationContext(
                envelope=t.envelope, store=t.store, events=t.events,
                policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
                protocol=protocol,
                split_registry=split_registry,
                eval_protocol_sha256=eval_proto,
                update_input_event=update_input,
            )
            return validate_envelope(ctx, "full_pre_update").decision_payload

        # ---- normal set: all 32 real generations must ALLOW ----------------
        normals = []
        timings = []
        for gen, _, _ in gens:
            d, ms = validate_identity(trajectory(gen))
            normals.append({"generation": gen.event_id, "decision": d.decision})
            timings.append(ms)
        log(f"normal set: {sum(1 for n in normals if n['decision']=='allow')}/{len(normals)} ALLOW; "
            f"validator mean {np.mean(timings):.1f} ms/env")

        # ---- F1-F4 injected into EVERY generation ---------------------------
        from grpo_guard.faults import (
            inject_f1_static_rollout,
            inject_f2_misbound_logprob,
            inject_f3_retokenization,
            inject_f4_mask_shift,
        )

        f14_results = []
        for gen, _, _ in gens:
            base = trajectory(gen)
            for case_id, inject in [
                ("f1_static_rollout", lambda b=base: inject_f1_static_rollout(b, 0, 1)),
                ("f2_misbound_logprob", lambda b=base: inject_f2_misbound_logprob(b, 1)),
                ("f3_retokenization", lambda b=base: inject_f3_retokenization(b)),
                ("f4_mask_shift", lambda b=base: inject_f4_mask_shift(b, 1)),
            ]:
                d, _ = validate_identity(inject())
                f14_results.append({"case_id": case_id, "generation": gen.event_id,
                                    "decision": d.decision, "reason_codes": d.reason_codes[:2]})
        for case_id in ("f1_static_rollout", "f2_misbound_logprob", "f3_retokenization", "f4_mask_shift"):
            hits = [r for r in f14_results if r["case_id"] == case_id]
            rejects = sum(1 for r in hits if r["decision"] == "reject")
            log(f"{case_id}: {rejects}/{len(hits)} reject across generations")

        # ---- F5-F8 injected into EVERY generation ---------------------------
        from grpo_guard.faults.f5_f8 import (
            inject_f5_split_leakage,
            inject_f6_evaluator_alias,
            inject_f7_event_reorder,
            inject_f8_artifact_mutation,
        )

        f58_results = []
        for gen, _, _ in gens:
            t5 = inject_f5_split_leakage(trajectory(gen), "held_out")
            d = validate_full(t5, split_registry=getattr(t5, "split_registry", None))
            f58_results.append({"case_id": "f5_split_leakage", "generation": gen.event_id,
                                "decision": d.decision, "reason_codes": d.reason_codes[:2]})

            t6 = trajectory(gen)
            from grpo_guard.schema.decisions import ValidationDecision
            from grpo_guard.schema.events import ValidationDecisionEvent

            parent = ValidationDecisionEvent(
                event_id=f"vdec-{gen.event_id}-parent", event_type="validation_decision",
                run_id=run_id, component_id="validator",
                lifecycle_seq=gen.lifecycle_seq + 1,
                created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                decision_payload=ValidationDecision(decision="allow",
                                                    validation_stage="identity_pre_reward",
                                                    reason_codes=["G001"]),
            ).seal()
            t6.events[parent.event_id] = parent
            from grpo_guard.schema.events import RewardEvent

            reward = RewardEvent(
                event_id=f"reward-{gen.event_id}-batch", event_type="reward_finished",
                run_id=run_id, component_id="countdown_reward",
                lifecycle_seq=gen.lifecycle_seq + 2,
                created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                reward_version="countdown-rule-v1", evaluator_protocol_sha256=reward_protocol_sha256(),
                source_generation_event=EventRef(uri="", event_id=gen.event_id,
                                                 event_sha256=gen.event_sha256),
                components={"correctness": 1.0, "format": 1.0}, terminal_status="success",
                latency_ms=0.0,
            ).seal()
            t6.events[reward.event_id] = reward
            t6.envelope = envelope_for(gen, stage="pre_update",
                                       reward_event=EventRef(uri="", event_id=reward.event_id,
                                                             event_sha256=reward.event_sha256),
                                       parent_identity=EventRef(uri="", event_id=parent.event_id,
                                                                event_sha256=parent.event_sha256))
            f6 = inject_f6_evaluator_alias(t6, reward_protocol_sha256())
            d = validate_full(f6, eval_proto=getattr(f6, "eval_protocol_sha256", None))
            f58_results.append({"case_id": "f6_evaluator_alias", "generation": gen.event_id,
                                "decision": d.decision, "reason_codes": d.reason_codes[:2]})

            t7 = inject_f7_event_reorder(trajectory(gen))
            gen7 = t7.events[t7.envelope.generation_event.event_id]
            upd = None
            from grpo_guard.schema.events import UpdateInputEvent

            upd = UpdateInputEvent(
                event_id=f"uinput-{t7.envelope.envelope_id}", run_id=t7.run_id,
                component_id="materializer", lifecycle_seq=gen7.lifecycle_seq + 100,
                created_at_utc=gen7.created_at_utc, update_id="update-1",
                preupdate_envelope=t7.envelope.ref(),
                preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
                sequence_token_ids=gen7.sequence_token_ids, loss_mask=gen7.loss_mask,
                authoritative_behavior_logprob_event=t7.envelope.training_contract.authoritative_behavior_logprob_event,
                authoritative_behavior_logprobs=gen7.service_behavior_logprobs,
                reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
                materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
                tokenizer_called=False,
            ).seal()
            d = validate_full(t7, update_input=upd)
            f58_results.append({"case_id": "f7_event_reorder", "generation": gen.event_id,
                                "decision": d.decision, "reason_codes": d.reason_codes[:2]})

            t8 = trajectory(gen)
            clone = Path(__import__("tempfile").mkdtemp(prefix="grpo-guard-f8-batch-"))
            shutil.copytree(t8.store.root, clone, dirs_exist_ok=True)
            from grpo_guard import testing

            t8b = testing.Trajectory(
                run_id=t8.run_id, events=t8.events, policy_manifest=t8.policy_manifest,
                split_manifest=t8.split_manifest, envelope=t8.envelope,
                store=ArtifactStore(clone),
                sequence=t8.sequence, target_mask=t8.target_mask, loss_mask=t8.loss_mask,
                logprobs=t8.logprobs, completion_text=t8.completion_text, goal=t8.goal,
                target_numbers=t8.target_numbers, reward_components=t8.reward_components,
                sync_events=t8.sync_events, sequence_ref=t8.sequence_ref,
            )
            d = validate_full(inject_f8_artifact_mutation(t8b))
            f58_results.append({"case_id": "f8_artifact_mutation", "generation": gen.event_id,
                                "decision": d.decision, "reason_codes": d.reason_codes[:2]})
        for case_id in ("f5_split_leakage", "f6_evaluator_alias", "f7_event_reorder", "f8_artifact_mutation"):
            hits = [r for r in f58_results if r["case_id"] == case_id]
            ok = sum(1 for r in hits if r["decision"] in ("reject", "quarantine"))
            log(f"{case_id}: {ok}/{len(hits)} rejected/quarantined across generations")

        (OUT_DIR / "batch_online_matrix.json").write_text(json.dumps({
            "run_id": run_id,
            "scope": "F1-F8 batch online matrix (256 real rollouts, 64 prompts x 4 gens, per-generation injection)",
            "source": "autodl2 vLLM server",
            "normal": normals,
            "validator_timing_ms": {"mean": round(float(np.mean(timings)), 2),
                                    "min": round(float(np.min(timings)), 2),
                                    "max": round(float(np.max(timings)), 2),
                                    "count": len(timings)},
            "f14": f14_results,
            "f58": f58_results,
            "summary": {
                "normal_allow": sum(1 for n in normals if n["decision"] == "allow"),
                "normal_total": len(normals),
                "f14_reject": sum(1 for r in f14_results if r["decision"] == "reject"),
                "f14_total": len(f14_results),
                "f58_reject_or_quarantine": sum(1 for r in f58_results if r["decision"] in ("reject", "quarantine")),
                "f58_total": len(f58_results),
            },
        }, indent=2), encoding="utf-8")
        log("BATCH ONLINE MATRIX DONE")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
