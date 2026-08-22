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
import sys
import time
from pathlib import Path

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/bounded_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8003"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51218"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from examples.countdown.online_matrix import (  # noqa: E402
    ckpt_sha,
    epoch,
    log_,
    make_envelope,
    next_lifecycle,
    run_id,
    sampling_sha,
    store,
    template_sha,
    tokenizer_sha,
    trajectory,
    validate_identity,
)
from grpo_guard.schema.artifacts import EventRef  # noqa: E402
from grpo_guard.schema.events import SyncEvent  # noqa: E402
from grpo_guard.store.append_log import AppendLog  # noqa: E402
from grpo_guard.store.artifact_store import ArtifactStore  # noqa: E402
from grpo_guard.validators.context import ProtocolConfig, ValidationContext  # noqa: E402
from grpo_guard.validators.validator import validate_envelope  # noqa: E402


def log(msg: str) -> None:
    print(f"[bounded] {msg}", flush=True)


def main() -> int:
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter

    import shutil

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT_DIR / "events", ignore_errors=True)
    shutil.rmtree(OUT_DIR / "store", ignore_errors=True)

    global store, log_, epoch, run_id
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
                              top_k=0, max_tokens=64, logprobs=0)
        pid, cid, lps = res["prompt_ids"], res["completion_ids"], res["logprobs"]
        gen = runtime.emit_generation(
            pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
            behavior_policy_version=0, checkpoint_manifest_sha256=ckpt_sha,
            sync_event=sync_ref, tokenizer_sha256=tokenizer_sha,
            chat_template_sha256=template_sha, sampling_config_sha256=sampling_sha,
            prompt_id=p["prompt_id"], request_id="req-bounded-0", required_epoch=epoch,
        )
        log(f"real generation: {gen.event_id}")

        t = trajectory(gen)

        def bounded_decide(contract_parent, max_lag, correction):
            from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
            from grpo_guard.schema.manifests import SplitManifest

            env = make_envelope(gen)
            contract = TrainingContract(
                protocol="bounded_off_policy", trainer_parent_policy_version=contract_parent,
                consuming_update_id="update-1", max_policy_lag_versions=max_lag,
                importance_correction=correction,
                behavior_logprob_source="generation_service",
                authoritative_behavior_logprob_event=EventRef(
                    uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                diagnostic_non_authoritative_logprobs_allowed=False,
            )
            env = env.model_copy(deep=True).model_copy(update={
                "envelope_id": f"{env.envelope_id}-bounded",
                "training_contract": contract,
            })
            env.envelope_sha256 = ""
            env = env.seal()
            from grpo_guard import testing

            t2 = testing.Trajectory(
                run_id=t.run_id, events=t.events, policy_manifest=t.policy_manifest,
                split_manifest=t.split_manifest, envelope=env, store=t.store,
                sequence=t.sequence, target_mask=t.target_mask, loss_mask=t.loss_mask,
                logprobs=t.logprobs, completion_text=t.completion_text, goal=t.goal,
                target_numbers=t.target_numbers, reward_components=t.reward_components,
                sync_events=t.sync_events, sequence_ref=t.sequence_ref,
            )
            protocol = ProtocolConfig(name="bounded_v01", mode="bounded_off_policy",
                                      max_policy_lag_versions=max_lag,
                                      importance_correction=correction)
            ctx = ValidationContext(
                envelope=t2.envelope, store=t2.store, events=t2.events,
                policy_manifest=t2.policy_manifest, split_manifest=t2.split_manifest,
                protocol=protocol,
            )
            return validate_envelope(ctx, "identity_pre_reward").decision_payload

        results = []
        for case_id, parent, max_lag, correction, expect in [
            ("lag1_within_bound_corrected", 1, 2, "importance-ratio-v1", "allow"),
            ("lag5_exceeds_bound", 5, 2, "importance-ratio-v1", "reject"),
            ("bounded_without_correction", 1, 2, None, "reject"),
        ]:
            d = bounded_decide(parent, max_lag, correction)
            codes = d.reason_codes[:3]
            results.append({"case_id": case_id, "decision": d.decision,
                            "reason_codes": codes, "expected": expect,
                            "match": d.decision == expect})
            log(f"{case_id}: {d.decision} {codes}")

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
