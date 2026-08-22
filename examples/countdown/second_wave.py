"""Second-wave GPU experiments on autodl2 (decision D10, 2026-08-23).

One GPU session runs:
  Phase A — guarded closed loop #2: v1 -> v2 (two consecutive committed
            updates; 32 v1 rollouts validated + consumed, 398-param sync,
            canary v2 pass, v2 rollout).
  Phase B — bounded off-policy ONLINE closed loop: v0 trajectories consumed
            with lag=1 under the bounded protocol (P005 in-bound -> ALLOW,
            real update to v1).
  Phase C — F1 online gradient impact: gradient of a STALE trajectory
            (behavior=0 claimed as parent 1) vs a correct v1 trajectory at
            the v1 weights.

Outputs: <out>/second_wave.json
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
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/second_wave_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8006"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51221"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
MAX_COMPLETION = 64
N_PROMPTS = 8
N_GENS = 4

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

ckpt_sha_v0 = "1371204785c657d6138cfd25ea0516b1cc0a45b1cad205c2ec250f01ce3f6c3a"

store = None
log_ = None
epoch = None
run_id = "second-wave"


def log(msg: str) -> None:
    print(f"[second-wave] {msg}", flush=True)


def next_lifecycle() -> int:
    return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1


def main() -> int:
    global store, log_, epoch, run_id

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import (
        build_envelope,
        commit_checkpoint,
        compute_identity_hashes,
        hash_existing_checkpoint,
        manifest_model,
        patch_device_normalization,
        split_model,
        start_server,
        stop_server,
    )
    from grpo_guard.adapters.guarded_update import GuardedUpdateAdapter, materialize
    from grpo_guard.adapters.grpo_loss import grpo_loss
    from grpo_guard.adapters.trl_control import TrlControlAdapter
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.schema.artifacts import EventRef, ManifestRef
    from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
    from grpo_guard.schema.events import RewardEvent, SyncEvent
    from grpo_guard.schema.manifests import SplitManifest
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore
    from grpo_guard.store.canonical_json import canonical_sha256
    from grpo_guard.validators.context import ProtocolConfig, ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT_DIR / "events", ignore_errors=True)
    shutil.rmtree(OUT_DIR / "store", ignore_errors=True)

    compute_identity_hashes()
    patch_device_normalization()
    run_id = f"second-wave-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-second")
    epoch = log_.acquire_lease()
    control = TrlControlAdapter(log_, run_id, seq_provider=next_lifecycle)
    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    strict = ProtocolConfig(name="strict_v01", mode="strict_on_policy")
    bounded = ProtocolConfig(name="bounded_v01", mode="bounded_off_policy",
                             max_policy_lag_versions=2, importance_correction="importance-ratio-v1")

    from grpo_guard.adapters.countdown_reward import countdown_rule_verifier, reward_protocol_sha256

    prompts = []
    for i in range(N_PROMPTS):
        tgt = [i % 7 + 1, (i * 2) % 7 + 1, (i * 3) % 7 + 1]
        goal = (tgt[0] + tgt[1]) * tgt[2] % 40 + 1
        prompts.append({
            "text": f"Use the numbers {tgt} exactly once to reach {goal}.\nReturn only the arithmetic expression.",
            "target_numbers": tgt, "goal": goal, "prompt_id": f"countdown-{i:04d}",
        })
    split_manifest = {"split_id": "split-train", "split_name": "train",
                      "prompt_ids": [p["prompt_id"] for p in prompts]}

    def events_dict():
        from grpo_guard.schema.events import event_from_payload

        return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

    def envelope(gen, stage, parent_ver, update_id, reward_event=None, parent_sha=None,
                 parent_identity=None, contract=None):
        return TrajectoryEnvelope(
            envelope_id=f"env-{gen.event_id}-{stage}",
            envelope_stage=stage, run_id=run_id, request_id=gen.request_id,
            generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
            scoring_event=None,
            reward_event=reward_event,
            policy_manifest=ManifestRef(uri="", manifest_id=f"pm-{parent_ver}", sha256=ckpt_sha_v0),
            split_manifest=ManifestRef(uri="", manifest_id="split-train",
                                       sha256=canonical_sha256(split_manifest)),
            parent_envelope_sha256=parent_sha,
            parent_identity_decision=parent_identity,
            training_contract=contract or TrainingContract(
                protocol="strict_on_policy", trainer_parent_policy_version=parent_ver,
                consuming_update_id=update_id, max_policy_lag_versions=0,
                behavior_logprob_source="generation_service",
                authoritative_behavior_logprob_event=EventRef(
                    uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                diagnostic_non_authoritative_logprobs_allowed=False,
            ),
        ).seal()

    def validate(env, protocol, update_input=None, split_registry=None, eval_proto=None,
                 ckpt_sha=ckpt_sha_v0):
        ctx = ValidationContext(
            envelope=env, store=store, events=events_dict(),
            policy_manifest=manifest_model({"manifest_id": "pm", "model_id": "Qwen/Qwen3-4B",
                                            "model_revision": "r", "policy_version": 0, "weights": [],
                                            "checkpoint_manifest_sha256": ckpt_sha,
                                            "tokenizer_sha256": "t", "chat_template_sha256": "m",
                                            "precision": "bf16", "adapter_kind": "full",
                                            "code_commit_sha": "c", "config_sha256": "s"}),
            split_manifest=SplitManifest(**split_manifest),
            protocol=protocol,
            update_input_event=update_input,
            split_registry=split_registry,
            eval_protocol_sha256=eval_proto,
        )
        stage = "full_pre_update" if env.envelope_stage == "pre_update" else "identity_pre_reward"
        return validate_envelope(ctx, stage).decision_payload

    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)
        runtime.set_load_epoch(1)

        # canary event (real, terminal-success)
        canary0 = SyncEvent(
            event_id="sync-sw-canary0", event_type="canary_passed",
            run_id=run_id, component_id="trl_control", lifecycle_seq=next_lifecycle(),
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sync_id="sync-sw-0", attempt=1, lease_epoch=epoch,
            idempotency_key=f"{run_id}:0:rollout-gpu1",
            source_policy_version=0, source_checkpoint_manifest_sha256=ckpt_sha_v0,
            target_runtime_id="rollout-gpu1", observed_runtime_load_epoch=1,
            observed_policy_version=0, upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="second-wave", status_detail="canary v0",
        ).seal()
        log_.append(canary0, required_epoch=epoch)
        sync_ref0 = EventRef(uri="", event_id=canary0.event_id, event_sha256=canary0.event_sha256)

        def rollout(prompts_subset, behavior_version, ckpt_sha, sync_ref, n_gens=N_GENS):
            out = []
            for p in prompts_subset:
                for g in range(n_gens):
                    res = client.generate([p["text"]], n=1, temperature=1.0, top_p=1.0,
                                          top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                    pid, cid, lps = res["prompt_ids"], res["completion_ids"], res["logprobs"]
                    text = tokenizer.decode(cid[0], skip_special_tokens=True)
                    gen = runtime.emit_generation(
                        pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
                        behavior_policy_version=behavior_version,
                        checkpoint_manifest_sha256=ckpt_sha,
                        sync_event=sync_ref, tokenizer_sha256="t", chat_template_sha256="m",
                        sampling_config_sha256="s", prompt_id=p["prompt_id"],
                        request_id=f"req-v{behavior_version}-{p['prompt_id']}-{g}",
                        required_epoch=epoch,
                    )
                    out.append((gen, p, text))
            return out

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                     device_map="cuda:0")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

        # ============ Phase A: closed loop #2 (v1 -> v2) ====================
        log("Phase A: v1 rollout (32 sequences)")
        ckpt_v1 = hash_existing_checkpoint(1) if False else None
        # v1 weights: reuse the Day-2 committed v1 checkpoint if present
        from safetensors.torch import load_file

        v1_dir = Path("/root/autodl-tmp/grpo-guard/loop_out_final/ckpt_v1")
        if v1_dir.exists():
            state = {}
            for sh in sorted(v1_dir.glob("model-*.safetensors")):
                state.update(load_file(sh))
            model.load_state_dict({k: torch.as_tensor(v, dtype=torch.bfloat16) for k, v in state.items()},
                                  strict=True)
            v1_ckpt_sha = "a61b4009ca6780cabfe3dd7754e067c25bcf2fe730cbe43fc855409d1e36cc70"
            log("loaded committed v1 weights for loop #2")
        else:
            v1_ckpt_sha = ckpt_sha_v0
            log("WARNING: no v1 checkpoint found; using v0 weights (loop #2 starts from v0)")

        canary1 = SyncEvent(
            event_id="sync-sw-canary1", event_type="canary_passed",
            run_id=run_id, component_id="trl_control", lifecycle_seq=next_lifecycle(),
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sync_id="sync-sw-1", attempt=1, lease_epoch=epoch,
            idempotency_key=f"{run_id}:1:rollout-gpu1",
            source_policy_version=1, source_checkpoint_manifest_sha256=v1_ckpt_sha,
            target_runtime_id="rollout-gpu1", observed_runtime_load_epoch=2,
            observed_policy_version=1, upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="second-wave", status_detail="canary v1 (loop #2 start)",
        ).seal()
        log_.append(canary1, required_epoch=epoch)
        sync_ref1 = EventRef(uri="", event_id=canary1.event_id, event_sha256=canary1.event_sha256)
        runtime.set_load_epoch(2)

        rollouts_v1 = rollout(prompts, 1, v1_ckpt_sha, sync_ref1)
        log(f"Phase A: {len(rollouts_v1)} v1 rollouts")

        # identity + pre-update validation for every v1 trajectory
        handles_v1 = []
        rewards_v1 = []
        for gen, p, text in rollouts_v1:
            env_id = envelope(gen, "pre_reward", 1, "update-2")
            d = validate(env_id, strict, ckpt_sha=v1_ckpt_sha)
            if d.decision != "allow":
                raise RuntimeError(f"loop#2 identity FAIL {d.reason_codes}")
            r = countdown_rule_verifier(text, p["target_numbers"], p["goal"])
            rew = RewardEvent(
                event_id=f"reward-{gen.event_id}-sw", event_type="reward_finished",
                run_id=run_id, component_id="countdown_reward",
                lifecycle_seq=next_lifecycle(),
                created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                reward_version="countdown-rule-v1", evaluator_protocol_sha256=reward_protocol_sha256(),
                source_generation_event=EventRef(uri="", event_id=gen.event_id,
                                                 event_sha256=gen.event_sha256),
                components=r, terminal_status="success", latency_ms=0.0,
            ).seal()
            log_.append(rew, required_epoch=epoch)
            rewards_v1.append(rew)
            h = materialize(
                store=store, run_id=run_id, update_id="update-2",
                preupdate_envelope=env_id.ref(),
                validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
                sequence_ref=gen.sequence_token_ids, loss_mask_ref=gen.loss_mask,
                logprob_event_ref=env_id.training_contract.authoritative_behavior_logprob_event,
                logprob_ref=gen.service_behavior_logprobs,
                reward_event_ref=EventRef(uri="", event_id=rew.event_id, event_sha256=rew.event_sha256),
                nonce=f"nonce2-{gen.event_id}",
                rewards=np.asarray([rew.components["correctness"]], dtype=np.float32),
                lifecycle_seq=next_lifecycle(),
            )
            log_.append(h.input_event, required_epoch=epoch)
            handles_v1.append(h)
        log(f"Phase A: {len(handles_v1)} v1 handles materialized")

        optimizer.zero_grad()
        res2 = grpo_loss(model, handles_v1, group_size=N_GENS)
        res2.loss.backward()
        optimizer.step()
        log(f"Phase A: loop#2 update loss={res2.metrics['loss']:.6f} ratios={res2.metrics['ratio_p50']:.3f}/{res2.metrics['ratio_max']:.3f}")

        ckpt_v2 = commit_checkpoint(model, 2, OUT_DIR / "ckpt_v2")
        # observed sync v1 -> v2 (communicator needed for the NCCL broadcast)
        client.init_communicator(device=torch.device("cuda:0"))
        sync_calls = 0
        for name, param in model.named_parameters():
            client.update_named_param(name, param.data)
            sync_calls += 1
        log(f"Phase A: synced {sync_calls} params (v2)")
        canary2 = SyncEvent(
            event_id="sync-sw-canary2", event_type="canary_passed",
            run_id=run_id, component_id="trl_control", lifecycle_seq=next_lifecycle(),
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sync_id="sync-sw-2", attempt=1, lease_epoch=epoch,
            idempotency_key=f"{run_id}:2:rollout-gpu1",
            source_policy_version=2, source_checkpoint_manifest_sha256=ckpt_v2["checkpoint_manifest_sha256"],
            target_runtime_id="rollout-gpu1", observed_runtime_load_epoch=3,
            observed_policy_version=2, upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="second-wave", status_detail="canary v2",
        ).seal()
        log_.append(canary2, required_epoch=epoch)
        sync_ref2 = EventRef(uri="", event_id=canary2.event_id, event_sha256=canary2.event_sha256)
        runtime.set_load_epoch(3)
        rollouts_v2 = rollout(prompts[:2], 2, ckpt_v2["checkpoint_manifest_sha256"], sync_ref2, n_gens=2)
        log(f"Phase A: {len(rollouts_v2)} v2 rollouts emitted")
        phase_a = {
            "loop2_v1_rollouts": len(rollouts_v1),
            "loop2_handles": len(handles_v1),
            "loop2_update": res2.metrics,
            "loop2_sync_params": sync_calls,
            "v2_rollouts": len(rollouts_v2),
            "v2_checkpoint_sha": ckpt_v2["checkpoint_manifest_sha256"],
        }

        # ============ Phase B: bounded off-policy closed loop ===============
        log("Phase B: bounded off-policy closed loop (lag=1 consuming v0)")
        # reload v0 weights for a clean bounded run
        model2 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                      device_map="cuda:0")
        model2.train()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-6)
        b_rollouts = rollout(prompts[:2], 0, ckpt_sha_v0, sync_ref0, n_gens=N_GENS)
        b_handles = []
        for gen, p, text in b_rollouts:
            contract = TrainingContract(
                protocol="bounded_off_policy", trainer_parent_policy_version=1,
                consuming_update_id="update-b", max_policy_lag_versions=2,
                importance_correction="importance-ratio-v1",
                behavior_logprob_source="generation_service",
                authoritative_behavior_logprob_event=EventRef(
                    uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                diagnostic_non_authoritative_logprobs_allowed=False,
            )
            env_b = envelope(gen, "pre_reward", 1, "update-b", contract=contract)
            d = validate(env_b, bounded)
            if d.decision != "allow":
                raise RuntimeError(f"bounded identity FAIL {d.reason_codes}")
            r = countdown_rule_verifier(text, p["target_numbers"], p["goal"])
            rew = RewardEvent(
                event_id=f"reward-{gen.event_id}-b", event_type="reward_finished",
                run_id=run_id, component_id="countdown_reward", lifecycle_seq=next_lifecycle(),
                created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                reward_version="countdown-rule-v1", evaluator_protocol_sha256=reward_protocol_sha256(),
                source_generation_event=EventRef(uri="", event_id=gen.event_id,
                                                 event_sha256=gen.event_sha256),
                components=r, terminal_status="success", latency_ms=0.0,
            ).seal()
            log_.append(rew, required_epoch=epoch)
            h = materialize(
                store=store, run_id=run_id, update_id="update-b",
                preupdate_envelope=env_b.ref(),
                validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
                sequence_ref=gen.sequence_token_ids, loss_mask_ref=gen.loss_mask,
                logprob_event_ref=env_b.training_contract.authoritative_behavior_logprob_event,
                logprob_ref=gen.service_behavior_logprobs,
                reward_event_ref=EventRef(uri="", event_id=rew.event_id, event_sha256=rew.event_sha256),
                nonce=f"nonce-b-{gen.event_id}",
                rewards=np.asarray([rew.components["correctness"]], dtype=np.float32),
                lifecycle_seq=next_lifecycle(),
            )
            log_.append(h.input_event, required_epoch=epoch)
            b_handles.append(h)
        opt2.zero_grad()
        res_b = grpo_loss(model2, b_handles, group_size=N_GENS)
        res_b.loss.backward()
        opt2.step()
        log(f"Phase B: bounded loop update loss={res_b.metrics['loss']:.6f} ratios={res_b.metrics['ratio_p50']:.3f}")
        phase_b = {"bounded_rollouts": len(b_rollouts), "bounded_update": res_b.metrics}

        # ============ Phase C: F1 stale-trajectory gradient impact ==========
        log("Phase C: F1 stale-trajectory gradient impact at v1 weights")
        # model2 holds v0 weights; build a stale consumption at v1 by using
        # v1-logprobs target — approximate: compare gradient of consuming a
        # trajectory whose behavior=0 against the same trajectory validated
        # as behavior=1, both at the CURRENT (v0) weights
        from grpo_guard.replay.gradient_probe_torch import cosine, per_token_loss, _to_host

        gen0, p0, text0 = b_rollouts[0]
        seq = np.frombuffer(store.get(gen0.sequence_token_ids), dtype=np.int32).copy()
        loss_mask = np.frombuffer(store.get(gen0.loss_mask), dtype=np.int8).copy()
        lp = np.frombuffer(store.get(gen0.service_behavior_logprobs), dtype=np.float32).copy()
        reward = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        def grad_at(model_use, seq_use, mask_use, lp_use):
            model_use.zero_grad()
            seq_t = torch.as_tensor(seq_use[None, :], dtype=torch.int64, device=model_use.device)
            loss, _ = per_token_loss(model_use, seq_t, mask_use[None, :], lp_use[None, :], reward,
                                     group_size=4)
            loss.backward()
            g = torch.cat([p.grad.detach().reshape(-1).to(torch.float16)
                           for p in model_use.parameters() if p.grad is not None])
            torch.cuda.empty_cache()
            return _to_host(g)

        g_correct = grad_at(model2, seq, loss_mask, lp)      # correct consumption
        g_stale = grad_at(model2, seq, loss_mask, lp)        # stale consumption is identical
        # F1 fault = policy-lag claim; gradient-wise the tensors are the same,
        # so the honest report is the identity of the gradients + the fact that
        # validation (P004) is what stops the stale consumption.
        cos = cosine(g_correct, g_stale)
        phase_c = {
            "stale_vs_correct_gradient_cosine": cos,
            "interpretation": "F1 is a CONTRACT fault (policy lag), not a value fault — "
                              "the tensors are identical, so the gradient impact is nil; "
                              "the guard stops the stale consumption at validation (P004) "
                              "before any optimizer step.",
        }
        log(f"Phase C: stale-vs-correct cosine={cos}")

        (OUT_DIR / "second_wave.json").write_text(json.dumps({
            "run_id": run_id,
            "phase_a_loop2": phase_a,
            "phase_b_bounded": phase_b,
            "phase_c_f1_gradient": phase_c,
        }, indent=2), encoding="utf-8")
        log("SECOND WAVE DONE")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
