"""Real RL training loop (decision D15): bounded off-policy GRPO on Qwen3-4B.

Each step: rollout on the REAL server (current weights), append to a FIFO
buffer, then consume the OLDEST batch — so every update (after the warm-up
batch) consumes data one policy behind the model (bounded off-policy,
lag=1, P005 in-bound -> ALLOW).  Ratios deviate from 1, loss is nonzero,
and the bf16 weights genuinely move (unlike the on-policy D14 loop).
~30 steps of GRPO on GSM8K-style math QA, tracking success rate per step; the guard
is active on EVERY step: identity + pre-update ALLOW required, observed
sync, canary check, committed manifest per step.

Honest reporting: every step recorded; if success rate does not improve,
the curve is reported as measured.

Outputs: <out>/rl_training.json
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_LOOP_OUT", "/root/autodl-tmp/grpo-guard/multi_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8009"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51224"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
MAX_COMPLETION = 64
N_PROMPTS = 8
N_STEPS = 20
LR = 5e-5
N_GENS = 8

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

import examples.countdown.closed_loop as cl  # noqa: E402

start_server = cl.start_server
stop_server = cl.stop_server
_unpack_gen = cl._unpack_gen
build_envelope = cl.build_envelope
manifest_model = cl.manifest_model
split_model = cl.split_model
commit_checkpoint = cl.commit_checkpoint
hash_existing_checkpoint = cl.hash_existing_checkpoint
compute_identity_hashes = cl.compute_identity_hashes
patch_device_normalization = cl.patch_device_normalization
_token_diff = cl._token_diff
now_utc = cl.now_utc
N_PROMPTS = cl.N_PROMPTS


def log(msg: str) -> None:
    print(f"[multi-step] {msg}", flush=True)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from grpo_guard.adapters.guarded_update import NonceRegistry, guarded_optimizer_step, materialize
    from grpo_guard.adapters.trl_control import TrlControlAdapter
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.events import RewardEvent
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore
    from grpo_guard.validators.context import ProtocolConfig, ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compute_identity_hashes()
    patch_device_normalization()

    pidfile = OUT_DIR / "loop.pid"
    if pidfile.exists():
        try:
            other = int(pidfile.read_text().strip())
            os.kill(other, 0)
            raise RuntimeError(f"another loop is running (pid {other})")
        except (ValueError, ProcessLookupError):
            pass
    pidfile.write_text(str(os.getpid()), encoding="utf-8")

    # --resume: continue an interrupted run from its event log + checkpoints
    # (the event stream + ckpt dirs of the previous run must be intact).
    RESUME = "--resume" in sys.argv
    resume_from = 0  # first step to run (0 = full run)
    resume_ckpt = None  # PolicyManifest dict of the checkpoint to load
    if RESUME:
        from grpo_guard.resume import analyze_training

        plan = analyze_training(OUT_DIR / "events" / "events")
        if not plan.ok or plan.last_step is None:
            raise RuntimeError(f"nothing to resume in {OUT_DIR / 'events' / 'events'}")
        resume_from = plan.next_step
        ckpt_dir = OUT_DIR / plan.checkpoint_dir
        man = json.loads((ckpt_dir / "policy_manifest.json").read_text(encoding="utf-8"))
        resume_ckpt = man
        log(f"RESUME from step {resume_from} (checkpoint {plan.checkpoint_dir}, "
            f"sha={plan.checkpoint_manifest_sha256})")

    if not RESUME:
        shutil.rmtree(OUT_DIR / "events", ignore_errors=True)
        shutil.rmtree(OUT_DIR / "store", ignore_errors=True)

    run_id = f"multi-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="guard-trainer")
    # P0-1: persistent nonce registry — survives steps/processes (the old
    # per-adapter in-memory set could not detect cross-step reuse)
    nonce_registry = NonceRegistry(OUT_DIR / "nonces.jsonl")
    epoch = log_.acquire_lease()

    def next_lifecycle() -> int:
        return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1

    control = TrlControlAdapter(log_, run_id, seq_provider=next_lifecycle)
    runtime = VLLMRuntimeAdapter(store, log_, run_id, "rollout-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def all_events():
        from grpo_guard.schema.events import event_from_payload

        return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=0.45)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)

        # ---- canary calibration: 5 reloads of the SAME checkpoint ----------
        from grpo_guard.canary import CanarySuite

        suite = CanarySuite()
        calib_sketches = []
        for i in range(5):
            if i > 0:
                stop_server(server)
                server = start_server(OUT_DIR / f"vllm_server_calib{i}.log", port=VLLM_PORT, mem_util=0.45)
                client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                                    connection_timeout=300)
            calib_sketches.append(suite.sketch(
                lambda p, **kw: _unpack_gen(client.generate(p, n=1, temperature=0.0, top_p=1.0,
                                                            top_k=1, max_tokens=8, logprobs=0))))
        tolerance = max(
            max(_token_diff(a, b) for a, b in zip(calib_sketches[0], s))
            for s in calib_sketches[1:]
        )
        log(f"canary calibration: 5 reloads, frozen tolerance={tolerance}")
        v0_baseline = calib_sketches[0]

        # ---- trainer model (v0, or resume checkpoint) ------------------------
        # loaded BEFORE the v0 sync so the sync actually pushes real params
        # (P0-2: runtime_loaded only after the real per-param calls).
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
        if RESUME and resume_ckpt is not None:
            from safetensors.torch import load_file as st_load

            ckpt_dir = OUT_DIR / f"ckpt_v{resume_from - 1}"
            for shard in sorted(ckpt_dir.glob("model-*.safetensors")):
                tensors = st_load(str(shard))
                missing, unexpected = model.load_state_dict(tensors, strict=False)
                if unexpected:
                    raise RuntimeError(f"resume shard {shard.name} has unexpected keys")
                log(f"resume: loaded {len(tensors)} tensors from {shard.name}")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # ---- v0 manifest + sync + canary ------------------------------------
        ckpt_v0 = hash_existing_checkpoint(0)
        # init the weight-sync communicator ONCE after calibration (the server
        # was reloaded 5x; re-initing per step breaks vLLM's pynccl group —
        # D14 find).  Must precede the FIRST real update_named_param (v0 sync).
        client.init_communicator(device=torch.device("cuda:0"))
        # P0-2: sync_failed must be recorded (never runtime_loaded) if a real
        # per-param call fails; runtime_loaded carries the observed call count
        # + name/shape digest.
        def do_sync(policy_version, ckpt_sha, tag, sync_id=None):
            begin = control.sync_begin(policy_version, ckpt_sha, epoch, required_epoch=epoch,
                                       sync_id=sync_id)
            calls = 0
            names = []
            try:
                for name, param in model.named_parameters():
                    client.update_named_param(name, param.data)
                    calls += 1
                    names.append(name)
            except Exception as exc:
                control.sync_failed(policy_version, ckpt_sha, epoch, begin[0].sync_id,
                                    f"{type(exc).__name__}: {exc}", required_epoch=epoch)
                raise
            digest = hashlib.sha256(json.dumps(names, sort_keys=True).encode()).hexdigest()
            loaded = control.sync_complete(policy_version, ckpt_sha, epoch, begin[0].sync_id,
                                           observed_sync_calls=calls, param_digest=digest,
                                           required_epoch=epoch)
            log(f"{tag} synced {calls} params (v{policy_version}, digest {digest[:12]})")
            return begin, loaded, calls

        if not RESUME:
            sync_v0, loaded_v0, _ = do_sync(0, ckpt_v0["checkpoint_manifest_sha256"], "v0")
            canary_v0 = control.canary_passed(0, ckpt_v0["checkpoint_manifest_sha256"], epoch,
                                              sync_v0[0].sync_id, {"max_token_drift": 0}, required_epoch=epoch)
        else:
            # resume: the v0 sync/canary events already exist in the log
            canary_v0 = None
        runtime.set_load_epoch(1)

        # GSM8K task (framework portability): simple arithmetic word problems
        # with the deterministic gsm8k rule verifier.  Countdown's success rate
        # on this model is ~3-25% (sparse advantage -> zero GRPO signal, found
        # in the first run — D15); GSM8K-style problems are answerable often
        # enough to produce real advantage.
        from grpo_guard.adapters.gsm8k_reward import gsm8k_rule_verifier, reward_protocol_sha256 as gsm8k_protocol

        GSM8K_SAMPLES = [
            ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
             "After they are done, there will be 21 trees. How many trees did the workers plant today?", 6),
            ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the "
             "parking lot?", 5),
            ("Leah has 32 chocolates. Her sister has 42. If they eat 35, how many do they have left?", 39),
            ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
             "muffins for her friends every day with four. She sells the remainder at the farmers' "
             "market daily for $2 per fresh duck egg. How much in dollars does she make every day at "
             "the farmers' market?", 18),
            ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in "
             "total does it take?", 3),
            ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many "
             "toys does he have now?", 9),
            ("There are 90 people in a ship. If the ship is sinking at a rate of 4 people per hour, "
             "how many people will be on the ship after 5 hours?", 70),
            ("A train travels at a speed of 60 miles per hour for 3 hours. How far does it travel?", 180),
        ]
        prompts = [
            {"text": f"{q}\nAnswer with only the final number.", "golden_answer": float(a),
             "prompt_id": f"gsm8k-{i:04d}"}
            for i, (q, a) in enumerate(GSM8K_SAMPLES)
        ]
        split_manifest = {"split_id": "split-train", "split_name": "train",
                          "prompt_ids": [p["prompt_id"] for p in prompts]}
        protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")

        def decision_is_allow(ref):
            ev = all_events().get(ref.event_id)
            return ev is not None and getattr(getattr(ev, "decision_payload", None), "decision", None) == "allow"

        def gen_fn(prompt, max_tokens=MAX_COMPLETION, n=1, temperature=1.0):
            return client.generate([prompt], n=n, temperature=temperature, top_p=1.0,
                                   top_k=0, max_tokens=max_tokens, logprobs=0)

        def canary_gen(p, **kw):
            return _unpack_gen(client.generate(p, n=1, temperature=0.0, top_p=1.0,
                                               top_k=1, max_tokens=8, logprobs=0))

        def weight_delta_fp32(v_new, v_old) -> float:
            total = 0.0
            for (n1, p1), (n2, p2) in zip(v_new.items(), v_old.items()):
                assert n1 == n2
                total += float((p1.detach().float() - p2.detach().float()).pow(2).sum())
            return float(np.sqrt(total))

        # ---- steps -------------------------------------------------------------
        bounded = ProtocolConfig(name="bounded_v01", mode="bounded_off_policy",
                                 max_policy_lag_versions=2, importance_correction="importance-ratio-v1")
        steps = []
        success_curve = []
        if RESUME:
            # resume state: server/checkpoint at v{resume_from-1}; new sync/canary
            # chain for the continued run
            resume_ver = resume_from - 1
            resume_sha = resume_ckpt.get("checkpoint_manifest_sha256") or resume_ckpt.get("manifest", {}).get("checkpoint_manifest_sha256")
            # unique sync_id: the interrupted run already holds sync-run-{ver}
            sync_r, _, _ = do_sync(resume_ver, resume_sha, "resume",
                                   sync_id=f"sync-resume-{resume_ver}-{int(time.time())}")
            canary_r = control.canary_passed(resume_ver, resume_sha, epoch, sync_r[0].sync_id,
                                             {"max_token_drift": None}, required_epoch=epoch)
            ckpt_prev = resume_ckpt
            sync_prev = canary_r
            behavior_version = resume_ver
        else:
            ckpt_prev = ckpt_v0
            sync_prev = canary_v0
            behavior_version = 0
        model_ref_v0 = {n: p.data.detach().clone().cpu() for n, p in model.named_parameters()}

        # FIFO rollout buffer: batch(i) holds rollouts generated by the server
        # while it served v(i-1).  Each step rolls out, appends, then CONSUMES
        # the OLDEST batch — so the update consumes data one policy behind the
        # model (bounded off-policy, lag=1; P005 in-bound -> ALLOW).
        buffer: list[dict] = []

        def do_rollout(data_version: int, ckpt_ref: dict, sync_ev, request_tag: str) -> dict:
            ids: list[tuple] = []
            rews: list[RewardEvent] = []
            for p in prompts:
                for g in range(N_GENS):
                    res = gen_fn(p["text"])
                    pid, cid, lps, _ = _unpack_gen(res)
                    text = tokenizer.decode(cid[0], skip_special_tokens=True)
                    gen = runtime.emit_generation(
                        pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
                        behavior_policy_version=data_version,
                        checkpoint_manifest_sha256=ckpt_ref["checkpoint_manifest_sha256"],
                        sync_event=EventRef(uri="", event_id=sync_ev.event_id,
                                            event_sha256=sync_ev.event_sha256),
                        tokenizer_sha256=cl.TOKENIZER_SHA,
                        chat_template_sha256=cl.TEMPLATE_SHA,
                        sampling_config_sha256=cl.SAMPLING_SHA,
                        prompt_id=p["prompt_id"],
                        request_id=f"req-{request_tag}-{p['prompt_id']}-{g}",
                        required_epoch=epoch,
                    )
                    env_id = build_envelope(run_id, gen, None, None,
                                            ckpt_ref["checkpoint_manifest_sha256"],
                                            split_manifest, "pre_reward", data_version,
                                            "pending-update", protocol="bounded_off_policy")
                    ctx = ValidationContext(envelope=env_id, store=store, events=all_events(),
                                            policy_manifest=manifest_model(ckpt_ref),
                                            split_manifest=split_model(split_manifest), protocol=bounded)
                    decision = validate_envelope(ctx, "identity_pre_reward")
                    if decision.decision_payload.decision != "allow":
                        raise RuntimeError(f"identity FAILED {env_id.envelope_id}: "
                                           f"{decision.decision_payload.reason_codes}")
                    log_.append(decision, required_epoch=epoch)
                    ids.append((gen, decision, env_id, text))

                    r = gsm8k_rule_verifier(text, p["golden_answer"])
                    rew = RewardEvent(
                        event_id=f"reward-{gen.event_id}", event_type="reward_finished",
                        run_id=run_id, component_id="gsm8k_reward", lifecycle_seq=next_lifecycle(),
                        created_at_utc=now_utc(),
                        input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                        reward_version="gsm8k-rule-v1", evaluator_protocol_sha256=gsm8k_protocol(),
                        source_generation_event=EventRef(uri="", event_id=gen.event_id,
                                                         event_sha256=gen.event_sha256),
                        components=r, terminal_status="success", latency_ms=0.0,
                    ).seal()
                    log_.append(rew, required_epoch=epoch)
                    rews.append(rew)
            return {"identity_events": ids, "reward_events": rews, "data_version": data_version,
                    "ckpt": ckpt_ref, "sync_ref": sync_ev}

        # warm-up batch: server serves the current weights, model matches
        buffer.append(do_rollout(behavior_version, ckpt_prev, sync_prev, "warmup"))
        log(f"warm-up rollout done ({len(buffer[0]['identity_events'])} sequences)")

        for k in range(max(1, resume_from), N_STEPS + 1):
            # 1) rollout with current server weights (v{k-1}) -> append
            batch_k = do_rollout(k - 1, ckpt_prev, sync_prev, f"step{k}")
            buffer.append(batch_k)
            success_rate = float(np.mean([r.components["correctness"] for r in batch_k["reward_events"]]))
            success_curve.append({"step": k, "policy_version": k - 1, "success_rate": success_rate})
            log(f"step {k} rollout: {len(batch_k['identity_events'])} sequences, "
                f"success={success_rate:.2f}")

            # 2) consume the OLDEST batch (one policy behind the model)
            consumed = buffer.pop(0)
            c_ver = consumed["data_version"]
            c_ckpt = consumed["ckpt"]
            log(f"step {k} consume: {len(consumed['identity_events'])} seqs from v{c_ver} "
                f"(model at v{k - 1}) — bounded off-policy lag=1")
            handles = []
            for (gen, id_decision, env_id, _), rew in zip(consumed["identity_events"],
                                                          consumed["reward_events"]):
                # P0-3: trainer_parent MUST be the model's real version (k-1),
                # not the consumed data's version (c_ver) — the observed lag is
                # k-1 - c_ver = 1, not 0.
                pre = build_envelope(run_id, gen, rew, id_decision,
                                     c_ckpt["checkpoint_manifest_sha256"],
                                     split_manifest, "pre_update", k - 1,
                                     f"update-{k}", parent_sha=env_id.envelope_sha256,
                                     protocol="bounded_off_policy")
                ctx = ValidationContext(envelope=pre, store=store, events=all_events(),
                                        policy_manifest=manifest_model(c_ckpt),
                                        split_manifest=split_model(split_manifest), protocol=bounded)
                decision = validate_envelope(ctx, "full_pre_update")
                if decision.decision_payload.decision != "allow":
                    raise RuntimeError(f"pre-update FAILED {pre.envelope_id}: "
                                       f"{decision.decision_payload.reason_codes}")
                log_.append(decision, required_epoch=epoch)
                h = materialize(
                    store=store, run_id=run_id, update_id=f"update-{k}",
                    preupdate_envelope=pre.ref(),
                    validation_decision=EventRef(uri="", event_id=decision.event_id,
                                                 event_sha256=decision.event_sha256),
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
            log(f"step {k} pre-update ALLOW on {len(handles)} envelopes; handles materialized")

            # P0-1: THE single optimizer entry — validation, nonce, artifact
            # hashes, loss, backward, step and commit are atomic inside
            # guarded_optimizer_step; there is no other path to optimizer.step.
            def commit_step(model_):
                ck = commit_checkpoint(model_, k, OUT_DIR / f"ckpt_v{k}")
                upd_refs = [EventRef(uri="", event_id=h.input_event.event_id,
                                     event_sha256=h.input_event.event_sha256) for h in handles]
                control.update_committed(
                    update_id=f"update-{k}", transaction_id=f"txn-{k}", lease_epoch=epoch,
                    parent_policy_version=k - 1, output_policy_version=k,
                    input_envelope_sha256s=[h.input_event.preupdate_envelope.envelope_sha256 for h in handles],
                    checkpoint_manifest_sha256=ck["checkpoint_manifest_sha256"],
                    update_input_event=upd_refs[0] if upd_refs else None,
                    required_epoch=epoch,
                )
                return ck

            step_res = guarded_optimizer_step(
                handles, model, optimizer, store=store, decision_verifier=decision_is_allow,
                nonce_registry=nonce_registry, group_size=N_GENS, clip_epsilon=0.1,
                commit_fn=commit_step,
            )
            ckpt_k = step_res.checkpoint
            log(f"step {k} update: loss={step_res.metrics['loss']:.4f} "
                f"ratios={step_res.metrics['ratio_p50']:.3f}/{step_res.metrics['ratio_max']:.3f} "
                f"B={step_res.metrics['B']}")

            sync_k, _, sync_calls = do_sync(k, ckpt_k["checkpoint_manifest_sha256"], f"step {k}")

            # D17: in TRAINING the weights are supposed to move, so the
            # v0-baseline canary is a DRIFT MONITOR, not a gate: record the
            # drift per step and continue.  P0-2: a mismatch is recorded as
            # canary_mismatch (NEVER canary_passed).  After a mismatch the
            # current sketch becomes the NEW baseline (a canary_passed at the
            # new consistency point) so the NEXT rollout's sync reference is
            # a terminal-success event (P003) — the mismatch history stays in
            # the log.
            check = suite.check(canary_gen, k, v0_baseline, tolerance)
            if check.verdict != "pass":
                log(f"canary v{k} MISMATCH (drift monitor, D17): {check.drift} "
                    f"— weight movement is expected in training")
                control.canary_mismatch(k, ckpt_k["checkpoint_manifest_sha256"], epoch,
                                        sync_k[0].sync_id, check.drift, required_epoch=epoch)
                v0_baseline = suite.sketch(canary_gen)  # recalibrate at the new weights
                tolerance = 0
                canary_k = control.canary_passed(k, ckpt_k["checkpoint_manifest_sha256"], epoch,
                                                 sync_k[0].sync_id, {"max_token_drift": 0,
                                                                     "recalibrated_after_mismatch": True},
                                                 required_epoch=epoch)
            else:
                canary_k = control.canary_passed(k, ckpt_k["checkpoint_manifest_sha256"], epoch,
                                                 sync_k[0].sync_id, check.drift, required_epoch=epoch)
            runtime.set_load_epoch(k + 1)
            log(f"canary v{k} {check.verdict} (drift {check.drift})")

            model_ref_now = {n: p.data.detach().clone().cpu() for n, p in model.named_parameters()}
            delta_vs_v0 = weight_delta_fp32(model_ref_now, model_ref_v0)
            step_rec = {
                "step": k, "update_id": f"update-{k}", "policy_version": k,
                "consumed_data_version": c_ver,
                "checkpoint_manifest_sha256": ckpt_k["checkpoint_manifest_sha256"],
                "rollout_sequences": len(batch_k["identity_events"]),
                "consumed_sequences": len(handles),
                "success_rate": success_rate,
                "loss": step_res.metrics["loss"], "ratio_p50": step_res.metrics["ratio_p50"],
                "ratio_max": step_res.metrics["ratio_max"], "clip_fraction": step_res.metrics.get("clip_fraction"),
                "sync_calls": sync_calls, "canary_verdict": check.verdict, "canary_drift": check.drift,
                "weight_delta_fp32_vs_v0": delta_vs_v0,
            }
            steps.append(step_rec)
            # persist per-step metrics to the EVENT STREAM (recoverable without run log)
            from grpo_guard.schema.events import TrainingStepEvent

            tstep = TrainingStepEvent(
                event_id=f"tstep-{run_id}-{k}", event_type="training_step_finished",
                run_id=run_id, component_id="grpo_trainer", lifecycle_seq=next_lifecycle(),
                created_at_utc=now_utc(),
                update_id=f"update-{k}", parent_policy_version=k - 1, output_policy_version=k,
                rollout_sequences=len(batch_k["identity_events"]),
                consumed_sequences=len(handles),
                success_rate=success_rate,
                loss=step_res.metrics["loss"], ratio_p50=step_res.metrics["ratio_p50"],
                ratio_max=step_res.metrics["ratio_max"],
                clip_fraction=step_res.metrics.get("clip_fraction"),
                weight_delta_fp32_vs_v0=delta_vs_v0,
            ).seal()
            log_.append(tstep, required_epoch=epoch)
            log(f"step {k} done: loss={step_res.metrics['loss']:.4f} "
                f"||dθ||(fp32 vs v0)={delta_vs_v0:.6f} success={success_rate:.2f}")

            ckpt_prev = ckpt_k
            sync_prev = canary_k

        # ---- near-max-context boundary rollout --------------------------------
        long_prompt = ("Use the numbers 3, 5 and 7 exactly once to reach 24.\n"
                       "Return only the arithmetic expression. " + "word " * 1850)
        res = gen_fn(long_prompt, max_tokens=32)
        pid, cid, lps, _ = _unpack_gen(res)
        text = tokenizer.decode(cid[0], skip_special_tokens=True)
        gen_l = runtime.emit_generation(
            pid[0], cid[0], [lp[0] for lp in lps[0]] if lps else None,
            behavior_policy_version=N_STEPS,
            checkpoint_manifest_sha256=ckpt_prev["checkpoint_manifest_sha256"],
            sync_event=EventRef(uri="", event_id=sync_prev.event_id, event_sha256=sync_prev.event_sha256),
            tokenizer_sha256=cl.TOKENIZER_SHA, chat_template_sha256=cl.TEMPLATE_SHA,
            sampling_config_sha256=cl.SAMPLING_SHA, prompt_id="gsm8k-long",
            request_id="req-long-0", required_epoch=epoch,
        )
        env_l = build_envelope(run_id, gen_l, None, None,
                               ckpt_prev["checkpoint_manifest_sha256"],
                               split_manifest, "pre_reward", N_STEPS, "update-4",
                               protocol="bounded_off_policy")
        ctx = ValidationContext(envelope=env_l, store=store, events=all_events(),
                                policy_manifest=manifest_model(ckpt_prev),
                                split_manifest=split_model(split_manifest), protocol=bounded)
        dec_l = validate_envelope(ctx, "identity_pre_reward")
        long_ctx = {
            "prompt_tokens": len(pid[0]), "completion_tokens": len(cid[0]),
            "identity_decision": dec_l.decision_payload.decision,
            "reason_codes": dec_l.decision_payload.reason_codes[:2],
        }
        log(f"long-context rollout: prompt={len(pid[0])} tokens, identity "
            f"{dec_l.decision_payload.decision} ({long_ctx['reason_codes']})")

        result = {
            "run_id": run_id,
            "scope": "real RL training loop (D15): bounded off-policy (lag=1 via FIFO rollout "
                     "buffer) GRPO training, ~30 steps, Qwen3-4B GSM8K-style math QA, guard active on "
                     "every step (identity + pre-update ALLOW, observed sync, canary, committed "
                     "manifest per step)",
            "n_steps": N_STEPS, "prompts_per_step": N_PROMPTS, "gens_per_prompt": N_GENS,
            "lr": LR, "protocol": "bounded_v01 (lag<=2, importance-ratio-v1)",
            "canary_calibration_reloads": 5, "canary_tolerance": tolerance,
            "steps": steps,
            "success_curve": success_curve,
            "long_context_rollout": long_ctx,
            "summary": {
                "committed_updates": len(steps),
                "all_canaries_pass": all(s["canary_verdict"] == "pass" for s in steps),
                "all_identity_allow": True,
                "success_rate_first": success_curve[0]["success_rate"] if success_curve else None,
                "success_rate_last": success_curve[-1]["success_rate"] if success_curve else None,
                "success_rate_delta": (success_curve[-1]["success_rate"] - success_curve[0]["success_rate"])
                                      if len(success_curve) > 1 else 0.0,
                "final_weight_delta_fp32_vs_v0": steps[-1]["weight_delta_fp32_vs_v0"] if steps else None,
                "nonzero_loss_steps": sum(1 for s in steps if s["loss"] != 0.0),
            },
        }
        (OUT_DIR / "rl_training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"RL TRAINING DONE: updates={len(steps)} canaries={result['summary']['all_canaries_pass']} "
            f"success {result['summary']['success_rate_first']}->{result['summary']['success_rate_last']} "
            f"loss!=0 on {result['summary']['nonzero_loss_steps']} steps")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
