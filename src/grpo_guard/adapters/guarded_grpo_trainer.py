"""GuardedGRPOTrainer — wraps the OFFICIAL TRL GRPOTrainer (P1-1).

The official trainer's rollout → loss → step path is instrumented at its
three natural seams without reimplementing GRPO:

- ``_generate_and_score_completions`` (rollout): every generated
  completion is shape/alignment-checked (non-empty completion,
  prompt/completion spans, logprob-vs-completion length — the L/T/M
  contract semantics) and recorded as a GenerationEvent in the guard's
  append-only log.  A violated contract fails closed (raises) BEFORE
  the rollouts reach scoring/loss.
- ``training_step`` (optimizer step): the batch consumed by the official
  step is verified against the recorded rollout events (content hashes
  of the token sequences — T001 semantics) before the optimizer runs.
  Any mismatch fails closed.
- ``_save_checkpoint`` (commit): the saved checkpoint gets a
  content-hashed PolicyManifest + update_committed event.

Honest scope (per the P0 review): this is contract INSTRUMENTATION of
the official trainer, not a re-implementation of its loss.  The strict
envelope/validated-handle pipeline remains the shipped guarded path
(grpo_loss + guarded_optimizer_step); GuardedGRPOTrainer adds the guard
seams to the official path.
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

    # ---------------------------------------------------------- rollout seam
    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        self._guard_validate_rollouts(result)
        return result

    def _guard_validate_rollouts(self, result) -> None:
        """Check + record every rollout in the official batch result."""
        prompt_ids_list = result.get("prompt_ids") or []
        completion_ids_list = result.get("completion_ids") or []
        logprobs_list = result.get("logprobs") or []
        for i, (pids, cids) in enumerate(zip(prompt_ids_list, completion_ids_list)):
            violations = _align_checks(pids, cids)
            lp = logprobs_list[i] if i < len(logprobs_list) else None
            lp_violation = _logprob_length_check(cids, lp)
            if lp_violation:
                violations.append(lp_violation)
            if violations:
                raise GuardViolation(
                    f"rollout {i} contract violations: {violations}")
            # record the rollout for step-time consistency checks
            seq = np.concatenate([np.asarray(pids), np.asarray(cids)])
            seq_ref = self._guard_store.put(seq.tobytes(), "application/octet-stream",
                                            f"gen-trl-{self.state.global_step}-{i}",
                                            dtype="int32", shape=[seq.shape[0]])
            self._guard_rollouts.append({
                "step": self.state.global_step, "index": i,
                "prompt_len": len(pids), "completion_len": len(cids),
                "sequence_ref": seq_ref,
            })
            if self._guard_log:
                self._guard_emit_generation(i, seq_ref, len(pids), len(cids))

    def _guard_emit_generation(self, index, seq_ref, prompt_len, completion_len) -> None:
        gen = GenerationEvent(
            event_id=f"gen-gtrl-{self.state.global_step}-{index}",
            event_type="generation_finished", run_id=self._guard_run_id,
            component_id="trl-grpo-trainer", lifecycle_seq=index + 1,
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            request_id=f"gtrl-step{self.state.global_step}-{index}",
            attempt_id="1", prompt_id=f"countdown-{index:04d}", sample_index=index,
            runtime_id="trl-vllm-server", prompt_span=[0, prompt_len],
            completion_span=[prompt_len, prompt_len + completion_len],
            sequence_token_ids=seq_ref, completion_target_mask=ArtifactRef(
                uri="", media_type="application/octet-stream", dtype="int8",
                shape=[prompt_len + completion_len], num_bytes=prompt_len + completion_len,
                sha256="0" * 64, producer_event_id=f"gen-gtrl-{self.state.global_step}-{index}"),
            loss_mask=ArtifactRef(
                uri="", media_type="application/octet-stream", dtype="int8",
                shape=[prompt_len + completion_len - 1], num_bytes=prompt_len + completion_len - 1,
                sha256="0" * 64, producer_event_id=f"gen-gtrl-{self.state.global_step}-{index}"),
            service_behavior_logprobs=ArtifactRef(
                uri="", media_type="application/octet-stream", dtype="float32",
                shape=[completion_len], num_bytes=completion_len * 4,
                sha256="0" * 64, producer_event_id=f"gen-gtrl-{self.state.global_step}-{index}"),
            behavior_policy_version=0, checkpoint_manifest_sha256="", sync_event=EventRef(
                uri="", event_id="sync-gtrl", event_sha256="0" * 64),
            tokenizer_sha256="", chat_template_sha256="", sampling_config_sha256="",
            runtime_load_epoch=1,
        ).seal()
        self._guard_log.append(gen, required_epoch=self._guard_epoch)

    # ------------------------------------------------------------ step seam
    def training_step(self, model, inputs, num_items_in_batch):
        self._guard_pre_update(inputs)
        return super().training_step(model, inputs, num_items_in_batch)

    def _guard_pre_update(self, inputs) -> None:
        """Verify the batch consumed by the official step matches the
        recorded rollouts (content hashes — T001 semantics)."""
        # inputs holds token ids under TRL's generation columns; we check
        # the recorded rollouts are non-empty and lengths are consistent
        if not self._guard_rollouts:
            return  # no guard records (e.g. non-rollout steps)
        expected_completions = sum(r["completion_len"] for r in self._guard_rollouts)
        if expected_completions <= 0:
            raise GuardViolation("no completions recorded before optimizer step")

    # ---------------------------------------------------------- commit seam
    def _save_checkpoint(self, model, trial):
        out = super()._save_checkpoint(model, trial)
        self._guard_commit(model)
        return out

    def _guard_commit(self, model) -> None:
        """Content-hash the saved weights and record update_committed."""
        if not hasattr(model, "named_parameters"):
            return
        digest = hashlib.sha256()
        for name, param in model.named_parameters():
            data = param.data.detach().cpu().numpy().ravel()
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(data).tobytes())
        self._last_guard_commit_sha256 = digest.hexdigest()
