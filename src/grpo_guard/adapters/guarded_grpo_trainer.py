"""GuardedGRPOTrainer — wraps the OFFICIAL TRL GRPOTrainer (P1-1).

The official trainer's rollout → loss → step path is instrumented at its
three natural seams without reimplementing GRPO:

- ``_generate_and_score_completions`` (rollout): every generated
  completion is shape/alignment-checked (non-empty completion,
  prompt/completion spans, logprob-vs-completion length — the L/T/M
  contract semantics) and recorded as a GenerationEvent; the REAL
  sequence, canonical completion mask and server behavior-logprob bytes
  are content-addressed in the guard store (no zero-hash placeholders,
  v0.4.0/P0-4).  A violated contract fails closed (raises) BEFORE the
  rollouts reach scoring/loss.
- ``training_step`` (optimizer step): the ACTUAL tensors the official
  step consumes (``input_ids`` rows, old ``logprobs`` completion spans,
  ``advantages`` row count) are hash/identity-compared against the
  recorded rollout artifacts BEFORE ``super().training_step`` — i.e.
  before loss/backward.  Any mismatch fails closed.  Records rotate after
  each step so a later step cannot be validated against stale rollouts.
- ``_save_checkpoint`` (commit): the saved checkpoint gets a
  content-hashed digest attribute (full PolicyManifest/update_committed
  events on this seam are still pending — see Honest scope).

Honest scope (per the 2026-08-25 audit): this is contract INSTRUMENTATION
of the official trainer, not a re-implementation of its loss.  The strict
envelope/validated-handle pipeline remains the shipped guarded path
(grpo_loss + guarded_optimizer_step); GuardedGRPOTrainer adds the guard
seams to the official path.  Fields that need GPU-run wiring (policy
version, checkpoint/tokenizer/template hashes) stay documented
placeholders; the token/mask/logprob/reward artifacts are REAL bytes.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from grpo_guard.schema.artifacts import ArtifactRef, EventRef
from grpo_guard.schema.events import GenerationEvent
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.store.append_log import AppendLog
from grpo_guard.validators.align import reconstruct_alignment


class GuardViolation(RuntimeError):
    """Raised when the official trainer path violates a guard contract."""


def _align_checks(prompt_ids, completion_ids) -> list[str]:
    """Contract checks on one rollout (L/T/M semantics, task-agnostic).

    Returns a list of violation strings (empty == pass).
    """
    violations = []
    if completion_ids is None or len(completion_ids) == 0:
        violations.append("M005_EMPTY_COMPLETION")
        return violations
    P = len(prompt_ids)
    T = P + len(completion_ids)
    if T < 2:
        violations.append("M001_MASK_SHAPE_MISMATCH")
        return violations
    try:
        alignment = reconstruct_alignment(T, P, T, padding_spans=None)
    except Exception as exc:  # alignment itself failed
        violations.append(f"ALIGN_FAILURE: {exc}")
        return violations
    # the canonical completion_target_mask is 1 on [P, T)
    expected = np.zeros(T, dtype=np.int8)
    expected[P:] = 1
    if not np.array_equal(alignment.completion_target_mask, expected):
        violations.append("M004_CANONICAL_MASK_MISMATCH")
    return violations


def _logprob_length_check(completion_ids, logprobs) -> str | None:
    """L004: logprobs must cover the completion (one per completion token)."""
    if logprobs is None:
        return None
    if len(logprobs) != len(completion_ids):
        return "L004_TOKEN_LOGPROB_LENGTH_MISMATCH"
    return None


def _as_np(value):
    """Normalize a torch tensor / list / array to numpy (value view)."""
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


class GuardedGRPOTrainer:
    """Mixin-style wrapper around TRL's GRPOTrainer.

    Subclass TRL's GRPOTrainer AND this class::

        class MyGuardedTrainer(GuardedGRPOTrainer, GRPOTrainer): ...

    The mixin overrides the three seams; all other behavior is the
    official trainer's.
    """

    def __init__(self, *args, guard_events_dir=None, guard_store_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard_run_id = f"guarded-trl-{int(time.time())}"
        self._guard_log = AppendLog(guard_events_dir, run_id=self._guard_run_id,
                                    lease_id="guard-trl") if guard_events_dir else None
        self._guard_store = ArtifactStore(guard_store_dir) if guard_store_dir else None
        self._guard_epoch = self._guard_log.acquire_lease() if self._guard_log else None
        self._guard_rollouts: list[dict] = []  # per-step rollout contract records
        self._last_guard_verified: dict | None = None

    # ---------------------------------------------------------- rollout seam
    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        self._guard_validate_rollouts(result)
        return result

    def _guard_validate_rollouts(self, result) -> None:
        """Check + record every rollout in the official batch result.

        P0-4: the REAL sequence, canonical mask, server behavior-logprob
        and reward bytes are content-addressed (no zero-hash
        placeholders); records are per-step (rotated, not accumulated).
        """
        # TRL returns torch tensors; normalize to python lists
        def _to_lists(key):
            v = result.get(key)
            if v is None:
                return []
            if hasattr(v, "cpu"):  # torch tensor -> numpy -> list
                v = v.cpu().numpy().tolist()
            return v

        prompt_ids_list = _to_lists("prompt_ids")
        completion_ids_list = _to_lists("completion_ids")
        logprobs_list = _to_lists("logprobs")
        rewards_list = _to_lists("rewards")
        step = getattr(getattr(self, "state", None), "global_step", 0)
        records = []
        for i, (pids, cids) in enumerate(zip(prompt_ids_list, completion_ids_list)):
            violations = _align_checks(pids, cids)
            lp = logprobs_list[i] if i < len(logprobs_list) else None
            lp_violation = _logprob_length_check(cids, lp)
            if lp_violation:
                violations.append(lp_violation)
            if violations:
                raise GuardViolation(
                    f"rollout {i} contract violations: {violations}")
            # REAL bytes, content-addressed (P0-4)
            P, C = len(pids), len(cids)
            total = P + C
            seq = np.ascontiguousarray(np.concatenate(
                [np.asarray(pids), np.asarray(cids)]).astype(np.int32))
            seq_ref = self._guard_store.put(
                seq.tobytes(), "application/octet-stream", f"gen-trl-{step}-{i}",
                dtype="int32", shape=[seq.shape[0]])
            mask = np.zeros(total, dtype=np.int8)
            mask[P:] = 1  # canonical completion_target_mask
            mask_ref = self._guard_store.put(
                mask.tobytes(), "application/octet-stream", f"mask-trl-{step}-{i}",
                dtype="int8", shape=[total])
            lp_arr = np.ascontiguousarray(np.asarray(lp, dtype=np.float32))
            lp_ref = self._guard_store.put(
                lp_arr.tobytes(), "application/octet-stream", f"logprob-trl-{step}-{i}",
                dtype="float32", shape=[lp_arr.shape[0]])
            rw_arr = (np.ascontiguousarray(np.asarray(rewards_list[i], dtype=np.float32))
                      if i < len(rewards_list) else None)
            rw_ref = None
            if rw_arr is not None:
                rw_ref = self._guard_store.put(
                    rw_arr.tobytes(), "application/octet-stream", f"reward-trl-{step}-{i}",
                    dtype="float32", shape=list(rw_arr.shape))
            records.append({
                "step": step, "index": i,
                "prompt_len": P, "completion_len": C,
                "sequence_ref": seq_ref, "mask_ref": mask_ref,
                "logprob_ref": lp_ref, "reward_ref": rw_ref,
            })
            if self._guard_log:
                self._guard_emit_generation(i, seq_ref, mask_ref, lp_ref, rw_ref, P, C)
        self._guard_rollouts = records

    def _guard_emit_generation(self, index, seq_ref, mask_ref, lp_ref, rw_ref,
                               prompt_len, completion_len) -> None:
        step = getattr(getattr(self, "state", None), "global_step", 0)
        gen = GenerationEvent(
            event_id=f"gen-gtrl-{step}-{index}",
            event_type="generation_finished", run_id=self._guard_run_id,
            component_id="trl-grpo-trainer", lifecycle_seq=index + 1,
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            request_id=f"gtrl-step{step}-{index}",
            attempt_id="1", prompt_id=f"countdown-{index:04d}", sample_index=index,
            runtime_id="trl-vllm-server", prompt_span=[0, prompt_len],
            completion_span=[prompt_len, prompt_len + completion_len],
            sequence_token_ids=seq_ref, completion_target_mask=mask_ref,
            loss_mask=mask_ref,
            service_behavior_logprobs=lp_ref,
            # Placeholders needing GPU-run wiring (documented, P0-4):
            # the official trainer does not expose a sync/checkpoint
            # manifest at rollout time in the current TRL version.
            behavior_policy_version=0, checkpoint_manifest_sha256="", sync_event=EventRef(
                uri="", event_id="sync-gtrl", event_sha256="0" * 64),
            tokenizer_sha256="", chat_template_sha256="", sampling_config_sha256="",
            runtime_load_epoch=1,
        ).seal()
        self._guard_log.append(gen, required_epoch=self._guard_epoch)

    # ------------------------------------------------------------ step seam
    def training_step(self, model, inputs, num_items_in_batch):
        self._guard_pre_update(inputs)
        out = super().training_step(model, inputs, num_items_in_batch)
        # P0-4: each optimizer step consumes its own rollout records;
        # rotate so the next step cannot validate against stale rollouts.
        self._guard_rollouts = []
        return out

    def _guard_pre_update(self, inputs) -> None:
        """Verify the ACTUAL tensors the official step will consume.

        P0-4: ``input_ids`` rows and old-logprob completion spans must
        byte-match the recorded server artifacts (T001/L004 semantics);
        ``advantages`` row count must match.  All checks run BEFORE
        ``super().training_step`` — i.e. before loss/backward.
        """
        if not self._guard_rollouts:
            return  # no guard records (e.g. non-rollout steps)
        records = self._guard_rollouts
        input_ids = _as_np(inputs.get("input_ids"))
        if input_ids is None:
            raise GuardViolation(
                "training_step batch has no input_ids — cannot verify the "
                "actual consumed tokens (guard refuses to run blind)")
        if input_ids.shape[0] != len(records):
            raise GuardViolation(
                f"batch rows {input_ids.shape[0]} != recorded rollouts {len(records)} "
                f"(stale/partial rollout consumption)")
        for i, rec in enumerate(records):
            row = np.ascontiguousarray(np.asarray(input_ids[i], dtype=np.int32))
            if hashlib.sha256(row.tobytes()).hexdigest() != rec["sequence_ref"].sha256:
                raise GuardViolation(
                    f"row {i}: consumed token ids do not match the recorded server "
                    f"sequence (T001 retokenization/misbinding)")
        logprobs = _as_np(inputs.get("logprobs"))
        if logprobs is not None:
            for i, rec in enumerate(records):
                P = rec["prompt_len"]
                comp = np.ascontiguousarray(np.asarray(logprobs[i, P:], dtype=np.float32))
                if hashlib.sha256(comp.tobytes()).hexdigest() != rec["logprob_ref"].sha256:
                    raise GuardViolation(
                        f"row {i}: consumed old logprobs do not match the recorded "
                        f"server logprobs (L004 misbinding)")
        advantages = _as_np(inputs.get("advantages"))
        if advantages is not None and advantages.shape[0] != len(records):
            raise GuardViolation(
                f"advantages rows {advantages.shape[0]} != recorded rollouts "
                f"{len(records)} (reward misbinding)")
        self._last_guard_verified = {
            "global_step": getattr(getattr(self, "state", None), "global_step", 0),
            "n_rollouts": len(records),
        }

    # ---------------------------------------------------------- commit seam
    def _save_checkpoint(self, model, trial):
        out = super()._save_checkpoint(model, trial)
        self._guard_commit(model)
        return out

    def _guard_commit(self, model) -> None:
        """Content-hash the saved weights and record update_committed.

        Honest scope (P0-4): this seam currently sets a digest attribute
        only; a full PolicyManifest + update_committed event chain on the
        official path is pending GPU-run wiring.
        """
        if not hasattr(model, "named_parameters"):
            return
        digest = hashlib.sha256()
        for name, param in model.named_parameters():
            data = param.data.detach().cpu().numpy().ravel()
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(data).tobytes())
        self._last_guard_commit_sha256 = digest.hexdigest()
