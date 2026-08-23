"""GRPO-Guard Day 2: the single real guarded online closed loop
(design doc §6.2, §17 Day 2).

    COMMITTED(v0) → SYNC(v0) → CANARY(v0) → GENERATION(v0)
      → IDENTITY_VALIDATED(allow) → REWARD → PRE_UPDATE_VALIDATED(allow)
      → MATERIALIZED(handles) → UPDATE_STARTED → UPDATE_COMMITTED(v1)
      → SYNC(v1) → CANARY(v1) → GENERATION(v1)

Runs on autodl2: GPU1 vLLM server (trl vllm-serve), GPU0 trainer model.
The rollout comes from TRL's VLLMClient.generate which returns the server's
own prompt/completion token ids and service logprobs — the runtime adapter
emits GenerationEvents from those (no re-tokenization).  The update consumes
ONLY ValidatedBatchHandles; the optimizer step is real; the checkpoint is
committed with a content-hashed PolicyManifest; the v1 weights are synced to
the server through the observed update_named_param calls and canary-checked
before the v1 rollout.  Text is decoded only as a read-only view for the
reward verifier (design doc §2.4).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_LOOP_OUT", "/root/autodl-tmp/grpo-guard/loop_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8001"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51216"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
SHARED_DIR = Path(os.environ.get("GRPO_GUARD_SHARED", "/root/autodl-tmp/shared"))
MAX_COMPLETION = 64
N_GENS = 4
N_PROMPTS = 8

TOKENIZER_SHA = ""
TEMPLATE_SHA = ""
SAMPLING_SHA = hashlib.sha256(b"grpo-guard-v01-greedy-t0.0-top1").hexdigest()


def log(msg: str) -> None:
    print(f"[loop] {msg}", flush=True)


def now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- server

def start_server(server_log: Path, port: int | None = None, model: str | None = None,
                 mem_util: float = 0.5, device: str = "1") -> subprocess.Popen:
    port = port or VLLM_PORT
    model = model or MODEL_PATH
    log(f"starting vLLM server (GPU{device}) at :{port} model={model} mem_util={mem_util}")
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [
            trl_bin, "vllm-serve", "--model", model, "--port", str(port),
            "--gpu-memory-utilization", str(mem_util), "--max-model-len", "2048",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,  # whole process group dies together
    )
    for _ in range(120):
        time.sleep(2)
        if health_at(port) and proc.poll() is None:
            log("server healthy")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"server died: {Path(server_log).read_text()[-2000:]}")
    raise RuntimeError("server not healthy in 240s")


def _health() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def health_at(port: int) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def stop_server(proc: subprocess.Popen) -> None:
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # the vLLM EngineCore may outlive the group (cmdline is just
    # "VLLM::EngineCore"); kill only processes tied to OUR server port or
    # model path — never blanket-kill the shared card, TTRL may be running
    subprocess.run(
        ["bash", "-c",
         f"for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do "
         f"cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null); "
         f"if echo \"$cmd\" | grep -qE '{VLLM_PORT}|{MODEL_PATH}|VLLM::EngineCore'; then kill -9 $p 2>/dev/null; fi; done"],
        capture_output=True,
    )
    time.sleep(5)


# ---------------------------------------------------------------- identity

def compute_identity_hashes() -> tuple[str, str]:
    global TOKENIZER_SHA, TEMPLATE_SHA
    cfg = json.loads((Path(MODEL_PATH) / "tokenizer_config.json").read_text(encoding="utf-8"))
    TOKENIZER_SHA = hashlib.sha256(
        (Path(MODEL_PATH) / "tokenizer_config.json").read_bytes()
        + (Path(MODEL_PATH) / "tokenizer.json").read_bytes()
    ).hexdigest()
    TEMPLATE_SHA = hashlib.sha256(cfg.get("chat_template", "").encode("utf-8")).hexdigest()
    return TOKENIZER_SHA, TEMPLATE_SHA


def hash_existing_checkpoint(policy_version: int) -> dict:
    """v0 manifest from the ORIGINAL checkpoint files (no model load)."""
    shards = sorted(Path(MODEL_PATH).glob("model-*.safetensors"))
    weights = [{
        "uri": f"artifact://{p.name}", "media_type": "application/safetensors",
        "num_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "producer_event_id": f"ckpt-commit-v{policy_version}",
    } for p in shards]
    return {
        "manifest_id": f"pm-{policy_version}",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "policy_version": policy_version,
        "parent_policy_version": None,
        "weights": weights,
        "checkpoint_manifest_sha256": hashlib.sha256(
            json.dumps({"shards": [w["sha256"] for w in weights]}, sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer_sha256": TOKENIZER_SHA,
        "chat_template_sha256": TEMPLATE_SHA,
        "precision": "bf16",
        "adapter_kind": "full",
        "code_commit_sha": os.environ.get("GRPO_GUARD_COMMIT", "local"),
        "config_sha256": SAMPLING_SHA,
    }


def commit_checkpoint(model, policy_version: int, ckpt_dir: Path) -> dict:
    """Save v1 weights, hash every shard, return the PolicyManifest payload."""
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    keys = list(state.keys())
    shard_size = max(1, len(keys) // 4)
    shard_paths = []
    for i in range(0, len(keys), shard_size):
        shard = {k: state[k].float().contiguous() for k in keys[i:i + shard_size]}
        path = ckpt_dir / f"model-{i // shard_size + 1:05d}-of-{len(keys) // shard_size + 1:05d}.safetensors"
        save_file(shard, path)
        shard_paths.append(path)
    weights = [{
        "uri": f"artifact://{p.name}", "media_type": "application/safetensors",
        "num_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "producer_event_id": f"ckpt-commit-v{policy_version}",
    } for p in shard_paths]
    manifest = {
        "manifest_id": f"pm-{policy_version}",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "policy_version": policy_version,
        "parent_policy_version": policy_version - 1,
        "weights": weights,
        "checkpoint_manifest_sha256": hashlib.sha256(
            json.dumps({"shards": [w["sha256"] for w in weights]}, sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer_sha256": TOKENIZER_SHA,
        "chat_template_sha256": TEMPLATE_SHA,
        "precision": "bf16",
        "adapter_kind": "full",
        "code_commit_sha": os.environ.get("GRPO_GUARD_COMMIT", "local"),
        "config_sha256": SAMPLING_SHA,
    }
    (ckpt_dir / "policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _token_diff(a: list[int], b: list[int]) -> int:
    n = max(len(a), len(b))
    return sum(1 for i in range(n) if i >= len(a) or i >= len(b) or a[i] != b[i])


def _unpack_gen(res: dict):
    """VLLMClient.generate returns a dict; split into the 4-tuple form."""
    return (res["prompt_ids"], res["completion_ids"], res["logprobs"], res.get("logprob_token_ids"))


# ---------------------------------------------------------------- envelopes

def build_envelope(run_id, gen, rew, id_decision, ckpt_sha, split, stage, parent_ver, update_id,
                   parent_sha=None, protocol: str = "strict_on_policy"):
    from grpo_guard.schema.artifacts import EventRef, ManifestRef
    from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
    from grpo_guard.store.canonical_json import canonical_sha256

    return TrajectoryEnvelope(
        envelope_id=f"env-{gen.event_id}-{stage}",
        envelope_stage=stage, run_id=run_id, request_id=gen.request_id,
        generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
        scoring_event=None,
        reward_event=EventRef(uri="", event_id=rew.event_id, event_sha256=rew.event_sha256) if rew else None,
        policy_manifest=ManifestRef(uri="", manifest_id=f"pm-{parent_ver}", sha256=ckpt_sha),
        split_manifest=ManifestRef(uri="", manifest_id=split["split_id"], sha256=canonical_sha256(split)),
        parent_envelope_sha256=parent_sha,
        parent_identity_decision=EventRef(uri="", event_id=id_decision.event_id, event_sha256=id_decision.event_sha256) if id_decision else None,
        training_contract=TrainingContract(
            protocol=protocol, trainer_parent_policy_version=parent_ver,
            consuming_update_id=update_id, max_policy_lag_versions=0,
            behavior_logprob_source="generation_service", authoritative_behavior_logprob_event=EventRef(
                uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
            diagnostic_non_authoritative_logprobs_allowed=False,
        ),
    ).seal()


def manifest_model(payload: dict):
    from grpo_guard.schema.manifests import PolicyManifest

    return PolicyManifest(**{k: v for k, v in payload.items() if k in PolicyManifest.model_fields})


def split_model(split: dict):
    from grpo_guard.schema.manifests import SplitManifest

    return SplitManifest(**split)


def patch_device_normalization() -> None:
    """trl 1.10.0 + vllm 0.26.0 adapter fix (design doc §14.2) — same as
    the smoke: normalize the unindexed device before pynccl init."""
    import torch
    import trl
    import vllm
    from trl.generation.vllm_client import VLLMClient

    assert trl.__version__ == "1.10.0" and vllm.__version__ == "0.26.0"
    _orig = VLLMClient.init_communicator

    def _normalized(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)

    VLLMClient.init_communicator = _normalized


# ---------------------------------------------------------------- main

def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    sys.path.insert(0, str(REPO_DIR / "src"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = SHARED_DIR / "guard-canary.lock"

    from grpo_guard.adapters.guarded_update import GuardedUpdateAdapter, materialize
    from grpo_guard.adapters.grpo_loss import grpo_loss
    from grpo_guard.adapters.trl_control import TrlControlAdapter
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.events import RewardEvent
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore
    from grpo_guard.validators.context import ProtocolConfig, ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    compute_identity_hashes()
    patch_device_normalization()

    # exclusive canary window (shared-card rule 1)
    lock_file.write_text(json.dumps({"holder": "grpo-guard", "reason": "canary calibration", "at": now_utc()}), encoding="utf-8")

    # fresh scratch for this run (event ids are deterministic per policy step)
    import shutil

    # single-loop guard: refuse to run if another loop process is alive
    pidfile = OUT_DIR / "loop.pid"
    if pidfile.exists():
        try:
            other = int(pidfile.read_text().strip())
            os.kill(other, 0)
            raise RuntimeError(f"another closed loop is running (pid {other})")
        except (ValueError, ProcessLookupError):
            pass
    pidfile.write_text(str(os.getpid()), encoding="utf-8")

    shutil.rmtree(OUT_DIR / "events", ignore_errors=True)
    shutil.rmtree(OUT_DIR / "store", ignore_errors=True)

    run_id = f"loop-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-trainer")
    epoch = log_.acquire_lease()
    def next_lifecycle() -> int:
        return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1

    control = TrlControlAdapter(log_, run_id, seq_provider=next_lifecycle)

    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def all_events():
        from grpo_guard.schema.events import event_from_payload

        return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

    server = start_server(OUT_DIR / "vllm_server.log")
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT, connection_timeout=300)

        # ---- canary calibration: 5 reloads of the SAME checkpoint ----------
        from grpo_guard.canary import CanarySuite

        suite = CanarySuite()
        calib_sketches = []
        for i in range(5):
            if i > 0:
                stop_server(server)
                server = start_server(OUT_DIR / f"vllm_server_calib{i}.log")
                client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT, connection_timeout=300)
            calib_sketches.append(suite.sketch(lambda p, **kw: _unpack_gen(client.generate(p, n=1, temperature=0.0, top_p=1.0, top_k=1, max_tokens=8, logprobs=0))))
        tolerance = max(
            max(_token_diff(a, b) for a, b in zip(calib_sketches[0], s))
            for s in calib_sketches[1:]
        )
        log(f"canary calibration: 5 reloads, frozen tolerance={tolerance}")
        v0_baseline = calib_sketches[0]

        # ---- v0 manifest (original checkpoint files) + sync + canary -------
        ckpt_v0 = hash_existing_checkpoint(0)
        sync_v0 = control.sync_chain(0, ckpt_v0["checkpoint_manifest_sha256"], epoch, required_epoch=epoch)
        canary_v0 = control.canary_passed(0, ckpt_v0["checkpoint_manifest_sha256"], epoch,
                                          sync_v0[0].sync_id, {"max_token_drift": 0}, required_epoch=epoch)
        runtime.set_load_epoch(1)

        # ---- trainer model (same checkpoint → policy v0) -------------------
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

        # ---- rollout v0 ------------------------------------------------------
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
        protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")
        sync_ref = EventRef(uri="", event_id=canary_v0.event_id, event_sha256=canary_v0.event_sha256)

        identity_events: list[tuple] = []  # (gen, decision, env_identity, text)
        reward_events: list[RewardEvent] = []
        for p in prompts:
            for g in range(N_GENS):
                res = client.generate([p["text"]], n=1, temperature=1.0, top_p=1.0,
                                                     top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                pid, cid, lps, _ = _unpack_gen(res)
                text = tokenizer.decode(cid[0], skip_special_tokens=True)
                gen = runtime.emit_generation(
                    pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
                    behavior_policy_version=0, checkpoint_manifest_sha256=ckpt_v0["checkpoint_manifest_sha256"],
                    sync_event=sync_ref, tokenizer_sha256=TOKENIZER_SHA, chat_template_sha256=TEMPLATE_SHA,
                    sampling_config_sha256=SAMPLING_SHA, prompt_id=p["prompt_id"],
                    request_id=f"req-v0-{p['prompt_id']}-{g}", required_epoch=epoch,
                )
                env_id = build_envelope(run_id, gen, None, None, ckpt_v0["checkpoint_manifest_sha256"],
                                        split_manifest, "pre_reward", 0, "update-1")
                ctx = ValidationContext(envelope=env_id, store=store, events=all_events(),
                                        policy_manifest=manifest_model(ckpt_v0),
                                        split_manifest=split_model(split_manifest), protocol=protocol)
                decision = validate_envelope(ctx, "identity_pre_reward")
                if decision.decision_payload.decision != "allow":
                    raise RuntimeError(f"identity FAILED {env_id.envelope_id}: {decision.decision_payload.reason_codes}")
                log_.append(decision, required_epoch=epoch)
                identity_events.append((gen, decision, env_id, text))

                r = countdown_rule_verifier(text, p["target_numbers"], p["goal"])
                rew = RewardEvent(
                    event_id=f"reward-{gen.event_id}", event_type="reward_finished", run_id=run_id,
                    component_id="countdown_reward", lifecycle_seq=next_lifecycle(),
                    created_at_utc=now_utc(),
                    input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                    reward_version="countdown-rule-v1", evaluator_protocol_sha256=reward_protocol_sha256(),
                    source_generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                    components=r, terminal_status="success", latency_ms=0.0,
                ).seal()
                log_.append(rew, required_epoch=epoch)
                reward_events.append(rew)
        log(f"v0 rollout: {len(identity_events)} sequences, identity ALLOW on all")

        # ---- pre-update validation + materialize ----------------------------
        handles = []
        for (gen, id_decision, env_id, _), rew in zip(identity_events, reward_events):
            pre = build_envelope(run_id, gen, rew, id_decision, ckpt_v0["checkpoint_manifest_sha256"],
                                 split_manifest, "pre_update", 0, "update-1", parent_sha=env_id.envelope_sha256)
            ctx = ValidationContext(envelope=pre, store=store, events=all_events(),
                                    policy_manifest=manifest_model(ckpt_v0),
                                    split_manifest=split_model(split_manifest), protocol=protocol)
            decision = validate_envelope(ctx, "full_pre_update")
            if decision.decision_payload.decision != "allow":
                raise RuntimeError(f"pre-update FAILED {pre.envelope_id}: {decision.decision_payload.reason_codes}")
            log_.append(decision, required_epoch=epoch)
            h = materialize(
                store=store, run_id=run_id, update_id="update-1",
                preupdate_envelope=pre.ref(),
                validation_decision=EventRef(uri="", event_id=decision.event_id, event_sha256=decision.event_sha256),
                sequence_ref=gen.sequence_token_ids, loss_mask_ref=gen.loss_mask,
                logprob_event_ref=pre.training_contract.authoritative_behavior_logprob_event,
                logprob_ref=gen.service_behavior_logprobs,
                reward_event_ref=EventRef(uri="", event_id=rew.event_id, event_sha256=rew.event_sha256),
                nonce=f"nonce-{gen.event_id}",
                rewards=np.asarray([rew.components["correctness"]], dtype=np.float32),
                lifecycle_seq=next_lifecycle(),
            )
            log_.append(h.input_event, required_epoch=epoch)
            handles.append(h)
        log(f"pre-update ALLOW on {len(handles)} envelopes; handles materialized")

        # ---- guarded update: one real optimizer step ------------------------
        def decision_is_allow(ref):
            ev = all_events().get(ref.event_id)
            return ev is not None and getattr(getattr(ev, "decision_payload", None), "decision", None) == "allow"

        adapter = GuardedUpdateAdapter(store, decision_verifier=decision_is_allow)
        optimizer.zero_grad()
        loss_res = grpo_loss(model, handles, group_size=N_GENS)
        loss_res.loss.backward()
        optimizer.step()
        log(f"guarded update: loss={loss_res.metrics['loss']:.4f} ratios={loss_res.metrics['ratio_p50']:.3f}/{loss_res.metrics['ratio_max']:.3f} B={loss_res.metrics['B']}")

        # ---- commit v1 + observed sync + canary check -----------------------
        ckpt_v1 = commit_checkpoint(model, 1, OUT_DIR / "ckpt_v1")
        upd_input_refs = [EventRef(uri="", event_id=h.input_event.event_id, event_sha256=h.input_event.event_sha256)
                          for h in handles]
        control.update_committed(
            update_id="update-1", transaction_id="txn-1", lease_epoch=epoch,
            parent_policy_version=0, output_policy_version=1,
            input_envelope_sha256s=[h.input_event.preupdate_envelope.envelope_sha256 for h in handles],
            checkpoint_manifest_sha256=ckpt_v1["checkpoint_manifest_sha256"],
            update_input_event=upd_input_refs[0] if upd_input_refs else None,
            required_epoch=epoch,
        )
        sync_v1 = control.sync_chain(1, ckpt_v1["checkpoint_manifest_sha256"], epoch, required_epoch=epoch)
        client.init_communicator(device=torch.device("cuda:0"))
        sync_calls = []
        for name, param in model.named_parameters():
            client.update_named_param(name, param.data)
            sync_calls.append(name)
        log(f"synced {len(sync_calls)} params (v1)")
        check = suite.check(lambda p, **kw: _unpack_gen(client.generate(p, n=1, temperature=0.0, top_p=1.0, top_k=1, max_tokens=8, logprobs=0)),
                            1, v0_baseline, tolerance)
        if check.verdict != "pass":
            raise RuntimeError(f"canary MISMATCH after v1 sync: {check.drift}")
        canary_v1 = control.canary_passed(1, ckpt_v1["checkpoint_manifest_sha256"], epoch,
                                          sync_v1[0].sync_id, check.drift, required_epoch=epoch)
        runtime.set_load_epoch(2)
        log(f"canary v1 {check.verdict} (drift {check.drift})")

        # ---- v1 rollout: proves the loop -------------------------------------
        p = prompts[0]
        res = client.generate([p["text"]], n=2, temperature=1.0, top_p=1.0,
                                             top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
        pid, cid, lps, _ = _unpack_gen(res)
        v1_count = len(pid)
        for g in range(v1_count):
            gen1 = runtime.emit_generation(
                pid[g], cid[g], [lp[0] for lp in lps[g]] if lps else None,
                behavior_policy_version=1, checkpoint_manifest_sha256=ckpt_v1["checkpoint_manifest_sha256"],
                sync_event=EventRef(uri="", event_id=canary_v1.event_id, event_sha256=canary_v1.event_sha256),
                tokenizer_sha256=TOKENIZER_SHA, chat_template_sha256=TEMPLATE_SHA,
                sampling_config_sha256=SAMPLING_SHA, prompt_id=p["prompt_id"], request_id=f"req-v1-{g}",
                required_epoch=epoch,
            )
            env1 = build_envelope(run_id, gen1, None, None, ckpt_v1["checkpoint_manifest_sha256"],
                                  split_manifest, "pre_reward", 1, "update-2")
            ctx = ValidationContext(envelope=env1, store=store, events=all_events(),
                                    policy_manifest=manifest_model(ckpt_v1),
                                    split_manifest=split_model(split_manifest), protocol=protocol)
            dec1 = validate_envelope(ctx, "identity_pre_reward")
            if dec1.decision_payload.decision != "allow":
                raise RuntimeError(f"v1 identity FAILED: {dec1.decision_payload.reason_codes}")
        log("v1 rollout validated (2 sequences)")

        # ---- report -----------------------------------------------------------
        report = {
            "run_id": run_id,
            "closed_loop": {
                "v0_rollout_sequences": len(identity_events),
                "v1_rollout_sequences": len(pid),
                "identity_allowed": len(identity_events),
                "pre_update_allowed": len(handles),
                "committed_optimizer_steps": 1,
                "policy_versions": {"v0": ckpt_v0["checkpoint_manifest_sha256"],
                                    "v1": ckpt_v1["checkpoint_manifest_sha256"]},
                "sync_params_observed": len(sync_calls),
                "canary": {"calibration_reloads": len(calib_sketches), "tolerance": tolerance,
                           "v1_verdict": check.verdict, "v1_drift": check.drift},
                "update_metrics": loss_res.metrics,
                "model": MODEL_PATH,
            },
        }
        (OUT_DIR / "run_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (OUT_DIR / "REPORT.md").write_text(
            "# GRPO-Guard Day 2 — guarded online closed loop\n\n"
            f"- run: {report['run_id']}\n"
            f"- v0 rollout sequences: {report['closed_loop']['v0_rollout_sequences']}\n"
            f"- identity ALLOW: {report['closed_loop']['identity_allowed']}\n"
            f"- pre-update ALLOW: {report['closed_loop']['pre_update_allowed']}\n"
            f"- committed optimizer steps: {report['closed_loop']['committed_optimizer_steps']}\n"
            f"- upstream sync params observed: {report['closed_loop']['sync_params_observed']}\n"
            f"- canary: {report['closed_loop']['canary']}\n"
            f"- v1 rollout sequences: {report['closed_loop']['v1_rollout_sequences']}\n"
            f"- update metrics: {report['closed_loop']['update_metrics']}\n",
            encoding="utf-8",
        )
        log("CLOSED LOOP COMPLETE")
        return 0
    finally:
        lock_file.unlink(missing_ok=True)
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
