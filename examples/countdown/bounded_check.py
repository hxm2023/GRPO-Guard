"""Bounded off-policy online verification on autodl2 (design doc §9.2).

Uses REAL server rollouts; validates the bounded-mode rule branches:
  - lag within bound + declared correction        -> ALLOW
  - lag beyond bound (no correction can save it)  -> reject P005_LAG_EXCEEDS_BOUND
  - bounded mode WITHOUT declared correction      -> reject P006_CORRECTION_UNDECLARED

Output: <out>/bounded_online.json
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
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/bounded_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8003"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51218"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
MAX_COMPLETION = 64

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

ckpt_sha = "1371204785c657d6138cfd25ea0516b1cc0a45b1cad205c2ec250f01ce3f6c3a"
tokenizer_sha = "tok-sha-bounded"
template_sha = "tpl-sha-bounded"
sampling_sha = "samp-sha-bounded"

store = None
log_ = None
epoch = None
run_id = "bounded"


def log(msg: str) -> None:
    print(f"[bounded] {msg}", flush=True)


def next_lifecycle() -> int:
    return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1


def main() -> int:
    global store, log_, epoch, run_id

    import torch  # noqa: F401  (keeps torch import parity with the loop)
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server
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

    run_id = f"bounded-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-bounded")
    epoch = log_.acquire_lease()
    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)
        runtime.set_load_epoch(1)

        canary = SyncEvent(
            event_id="sync-bounded-canary", event_type="canary_passed",
            run_id=run_id, component_id="trl_control", lifecycle_seq=next_lifecycle(),
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sync_id="sync-bounded", attempt=1, lease_epoch=epoch,
            idempotency_key=f"{run_id}:0:rollout-gpu1",
            source_policy_version=0, source_checkpoint_manifest_sha256=ckpt_sha,
            target_runtime_id="rollout-gpu1", observed_runtime_load_epoch=1,
            observed_policy_version=0, upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="bounded", status_detail="bounded online check",
        ).seal()
        log_.append(canary, required_epoch=epoch)
        sync_ref = EventRef(uri="", event_id=canary.event_id, event_sha256=canary.event_sha256)

        p = {
            "text": "Use the numbers [4, 5, 6] exactly once to reach 30.\nReturn only the arithmetic expression.",
            "target_numbers": [4, 5, 6], "goal": 30, "prompt_id": "countdown-0000",
        }
        res = client.generate([p["text"]], n=1, temperature=1.0, top_p=1.0,
                              top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
        pid, cid, lps = res["prompt_ids"], res["completion_ids"], res["logprobs"]
        gen = runtime.emit_generation(
            pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
            behavior_policy_version=0, checkpoint_manifest_sha256=ckpt_sha,
            sync_event=sync_ref, tokenizer_sha256=tokenizer_sha,
            chat_template_sha256=template_sha, sampling_config_sha256=sampling_sha,
            prompt_id=p["prompt_id"], request_id="req-bounded-0", required_epoch=epoch,
        )
        log(f"real generation: {gen.event_id}")

        # ---- rebuild a Trajectory from the real generation ------------------
        seq = np.frombuffer(store.get(gen.sequence_token_ids), dtype=np.int32).copy()
        target = np.frombuffer(store.get(gen.completion_target_mask), dtype=np.int8).copy()
        loss = np.frombuffer(store.get(gen.loss_mask), dtype=np.int8).copy()
        lp = np.frombuffer(store.get(gen.service_behavior_logprobs), dtype=np.float32).copy()

        def events_dict():
            from grpo_guard.schema.events import event_from_payload

            return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

        def envelope_for(contract):
            split = {"split_id": "split-train", "split_name": "train", "prompt_ids": ["countdown-0000"]}
            return TrajectoryEnvelope(
                envelope_id=f"env-{gen.event_id}-bounded",
                envelope_stage="pre_reward", run_id=run_id, request_id=gen.request_id,
                generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                policy_manifest=ManifestRef(uri="", manifest_id="pm-0", sha256=ckpt_sha),
                split_manifest=ManifestRef(uri="", manifest_id="split-train",
                                           sha256=canonical_sha256(split)),
                training_contract=contract,
            ).seal()

        from grpo_guard import testing

        def decide(contract, protocol):
            env = envelope_for(contract)
            t = testing.Trajectory(
                run_id=run_id, events=events_dict(),
                policy_manifest=PolicyManifest(manifest_id="pm-0", model_id="Qwen/Qwen3-4B",
                                               model_revision="r", policy_version=0, weights=[],
                                               checkpoint_manifest_sha256=ckpt_sha,
                                               tokenizer_sha256=tokenizer_sha,
                                               chat_template_sha256=template_sha,
                                               precision="bf16", adapter_kind="full",
                                               code_commit_sha="c", config_sha256=sampling_sha),
                split_manifest=SplitManifest(split_id="split-train", split_name="train",
                                             prompt_ids=["countdown-0000"]),
                envelope=env, store=store, sequence=seq, target_mask=target, loss_mask=loss,
                logprobs=lp, completion_text="", goal=0, target_numbers=[],
                reward_components={}, sequence_ref=gen.sequence_token_ids,
            )
            ctx = ValidationContext(
                envelope=t.envelope, store=t.store, events=t.events,
                policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
                protocol=protocol,
            )
            return validate_envelope(ctx, "identity_pre_reward").decision_payload

        results = []
        for case_id, parent, max_lag, correction, expect in [
            ("lag1_within_bound_corrected", 1, 2, "importance-ratio-v1", "allow"),
            ("lag5_exceeds_bound", 5, 2, "importance-ratio-v1", "reject"),
            ("bounded_without_correction", 1, 2, None, "reject"),
        ]:
            contract = TrainingContract(
                protocol="bounded_off_policy", trainer_parent_policy_version=parent,
                consuming_update_id="update-1", max_policy_lag_versions=max_lag,
                importance_correction=correction,
                behavior_logprob_source="generation_service",
                authoritative_behavior_logprob_event=EventRef(
                    uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                diagnostic_non_authoritative_logprobs_allowed=False,
            )
            protocol = ProtocolConfig(name="bounded_v01", mode="bounded_off_policy",
                                      max_policy_lag_versions=max_lag,
                                      importance_correction=correction)
            d = decide(contract, protocol)
            results.append({"case_id": case_id, "decision": d.decision,
                            "reason_codes": d.reason_codes[:3], "expected": expect,
                            "match": d.decision == expect})
            log(f"{case_id}: {d.decision} {d.reason_codes[:3]}")

        (OUT_DIR / "bounded_online.json").write_text(json.dumps({
            "run_id": run_id, "scope": "bounded off-policy online (design doc §9.2)",
            "source": "autodl2 vLLM server",
            "generation": gen.event_id,
            "results": results,
            "summary": {"matched": sum(1 for r in results if r["match"]), "total": len(results)},
        }, indent=2), encoding="utf-8")
        log("BOUNDED ONLINE DONE")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
