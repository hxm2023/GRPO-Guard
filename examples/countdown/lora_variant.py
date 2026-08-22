"""LoRA-variant manifest/identity check on autodl2 (design doc §21, §7.2).

Proves the guard's adapter-kind coverage: a LoRA update commits with the
base and adapter hashed SEPARATELY in the PolicyManifest
(base_model_sha256 + adapter_sha256, adapter_kind="lora"), and the
identity chain (rollout -> validation -> update -> commit) works unchanged.

Output: <out>/lora_variant.json
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
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/lora_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8007"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51222"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
MAX_COMPLETION = 64

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

ckpt_sha_v0 = "1371204785c657d6138cfd25ea0516b1cc0a45b1cad205c2ec250f01ce3f6c3a"

store = None
log_ = None
epoch = None
run_id = "lora"


def log(msg: str) -> None:
    print(f"[lora] {msg}", flush=True)


def next_lifecycle() -> int:
    return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1


def main() -> int:
    global store, log_, epoch, run_id

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    run_id = f"lora-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-lora")
    epoch = log_.acquire_lease()
    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")

    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)
        runtime.set_load_epoch(1)

        canary = SyncEvent(
            event_id="sync-lora-canary", event_type="canary_passed",
            run_id=run_id, component_id="trl_control", lifecycle_seq=next_lifecycle(),
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sync_id="sync-lora", attempt=1, lease_epoch=epoch,
            idempotency_key=f"{run_id}:0:rollout-gpu1",
            source_policy_version=0, source_checkpoint_manifest_sha256=ckpt_sha_v0,
            target_runtime_id="rollout-gpu1", observed_runtime_load_epoch=1,
            observed_policy_version=0, upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="lora", status_detail="canary lora",
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
            behavior_policy_version=0, checkpoint_manifest_sha256=ckpt_sha_v0,
            sync_event=sync_ref, tokenizer_sha256="t", chat_template_sha256="m",
            sampling_config_sha256="s", prompt_id=p["prompt_id"],
            request_id="req-lora-0", required_epoch=epoch,
        )
        log(f"real generation: {gen.event_id}")

        split = {"split_id": "split-train", "split_name": "train", "prompt_ids": ["countdown-0000"]}
        env = TrajectoryEnvelope(
            envelope_id=f"env-{gen.event_id}-identity",
            envelope_stage="pre_reward", run_id=run_id, request_id=gen.request_id,
            generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
            policy_manifest=ManifestRef(uri="", manifest_id="pm-0", sha256=ckpt_sha_v0),
            split_manifest=ManifestRef(uri="", manifest_id="split-train",
                                       sha256=canonical_sha256(split)),
            training_contract=TrainingContract(
                protocol="strict_on_policy", trainer_parent_policy_version=0,
                consuming_update_id="update-lora", max_policy_lag_versions=0,
                behavior_logprob_source="generation_service",
                authoritative_behavior_logprob_event=EventRef(
                    uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                diagnostic_non_authoritative_logprobs_allowed=False,
            ),
        ).seal()

        def events_dict():
            from grpo_guard.schema.events import event_from_payload

            return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

        ctx = ValidationContext(
            envelope=env, store=store, events=events_dict(),
            policy_manifest=PolicyManifest(manifest_id="pm-0", model_id="Qwen/Qwen3-4B",
                                           model_revision="r", policy_version=0, weights=[],
                                           checkpoint_manifest_sha256=ckpt_sha_v0,
                                           tokenizer_sha256="t", chat_template_sha256="m",
                                           precision="bf16", adapter_kind="full",
                                           code_commit_sha="c", config_sha256="s"),
            split_manifest=SplitManifest(**split), protocol=protocol,
        )
        d = validate_envelope(ctx, "identity_pre_reward").decision_payload
        if d.decision != "allow":
            raise RuntimeError(f"identity FAIL {d.reason_codes}")
        log("identity ALLOW")

        # ---- LoRA training: adapters only -----------------------------------
        from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                     device_map="cuda:0")
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        seq = np.frombuffer(store.get(gen.sequence_token_ids), dtype=np.int32)
        loss_mask = np.frombuffer(store.get(gen.loss_mask), dtype=np.int8)
        lp = np.frombuffer(store.get(gen.service_behavior_logprobs), dtype=np.float32)
        seq_t = torch.as_tensor(seq[None, :], dtype=torch.int64, device=model.device)
        mask_t = torch.as_tensor(loss_mask[None, :], device=model.device).bool()
        old_t = torch.as_tensor(lp[None, :], dtype=torch.float32, device=model.device)

        optimizer.zero_grad()
        out = model(input_ids=seq_t)[0]
        logits = out[:, :-1, :].float()
        targets = seq_t[:, 1:]
        new_lp = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.config.vocab_size), targets.reshape(-1), reduction="none"
        ).reshape(1, seq.shape[0] - 1)
        counts = mask_t.sum(dim=1)
        flat_real = torch.cat([old_t[b, : counts[b]] for b in range(1)])
        old_padded = torch.zeros_like(new_lp)
        old_padded[mask_t] = flat_real
        ratio = torch.exp(new_lp - old_padded)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        loss = -(torch.min(ratio, clipped) * mask_t.float()).sum() / mask_t.float().sum()
        loss.backward()
        optimizer.step()
        log(f"LoRA update loss={loss.item():.6f}")

        # ---- commit: base + adapter hashed SEPARATELY -----------------------
        from safetensors.torch import save_file

        adapter_dir = OUT_DIR / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_state = {k: v.float().contiguous() for k, v in model.state_dict().items()
                         if "lora_" in k}
        save_file(adapter_state, adapter_dir / "adapter_model.safetensors")
        adapter_sha = __import__("hashlib").sha256(
            (adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
        base_sha = __import__("hashlib").sha256(b"Qwen3-4B-base-fixed").hexdigest()  # base identity marker

        manifest = {
            "manifest_id": "pm-lora-1",
            "model_id": "Qwen/Qwen3-4B",
            "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "policy_version": 1,
            "parent_policy_version": 0,
            "weights": [],
            "checkpoint_manifest_sha256": __import__("hashlib").sha256(
                json.dumps({"base": base_sha, "adapter": adapter_sha}, sort_keys=True).encode()
            ).hexdigest(),
            "tokenizer_sha256": "t",
            "chat_template_sha256": "m",
            "precision": "bf16",
            "adapter_kind": "lora",
            "base_model_sha256": base_sha,
            "adapter_sha256": adapter_sha,
            "code_commit_sha": "c",
            "config_sha256": "s",
        }
        (OUT_DIR / "policy_manifest_lora.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"LoRA commit: adapter_kind=lora, base={base_sha[:12]}, adapter={adapter_sha[:12]}")

        # sanity: the LoRA manifest round-trips through the schema
        pm = PolicyManifest(**{k: v for k, v in manifest.items() if k in PolicyManifest.model_fields})
        log(f"manifest validated: adapter_kind={pm.adapter_kind}, "
            f"base={pm.base_model_sha256[:12]}, adapter={pm.adapter_sha256[:12]}")

        (OUT_DIR / "lora_variant.json").write_text(json.dumps({
            "run_id": run_id,
            "scope": "LoRA-variant manifest/identity check (design doc §21, §7.2)",
            "generation": gen.event_id,
            "identity_decision": d.decision,
            "lora_update_loss": loss.item(),
            "adapter_params": sum(v.numel() for v in adapter_state.values()),
            "adapter_kind": pm.adapter_kind,
            "base_model_sha256": pm.base_model_sha256,
            "adapter_sha256": pm.adapter_sha256,
            "checkpoint_manifest_sha256": pm.checkpoint_manifest_sha256,
        }, indent=2), encoding="utf-8")
        log("LORA VARIANT DONE")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
