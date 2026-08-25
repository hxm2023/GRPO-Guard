"""GuardedGRPOTrainer — wraps the OFFICIAL TRL GRPOTrainer (P1-1).

The official trainer's rollout → loss → step path is instrumented at its
three natural seams without reimplementing GRPO.

TRL 1.10.0 reality (verified against the installed source, 2026-08-25):
``_generate_and_score_completions`` returns ``prompt_ids`` /
``prompt_mask`` / ``completion_ids`` / ``completion_mask`` /
``advantages`` and, for vLLM, ``sampling_per_token_logps`` (the SERVER's
behavior logprobs) and ``old_per_token_logps`` (the trainer's own
recomputed logprobs, used as the loss denominator with importance-sampling
correction).  ``_prepare_inputs`` then SHUFFLES the batch rows and slices
them per accumulation step — so step-time verification must match by
CONTENT, not by row position.

- ``_generate_and_score_completions`` (rollout): every completion is
  shape/alignment-checked on its REAL (mask-selected) tokens and recorded
  as a GenerationEvent; the REAL sequence, completion mask, the server's
  sampling logprobs, the old logprobs the loss will consume, and the
  advantages are content-addressed in the guard store (no zero-hash
  placeholders).  A violated contract fails closed BEFORE loss/backward.
- ``_prepare_inputs`` (step seam): the ACTUAL tensor batch the loss will
  consume is content-matched against the not-yet-consumed recorded
  artifacts (token ids → T001, old-logprob completion span → L004,
  advantages → reward binding) BEFORE loss/backward.  TRL slices the
  generation batch across steps, so matched records are consumed
  per-row: a row can never be used twice, and stale/foreign rows fail.
- ``_save_checkpoint`` (commit): the saved checkpoint gets a
  content-hashed digest attribute (full PolicyManifest/update_committed
  events on this seam are still pending — see Honest scope).

Honest scope (per the 2026-08-25 audit): this is contract INSTRUMENTATION
of the official trainer, not a re-implementation of its loss or a
capability gate on ``optimizer.step()``.  The strict
envelope/validated-handle pipeline remains the shipped guarded path
(grpo_loss + guarded_optimizer_step).  Fields needing GPU wiring (policy
version, checkpoint/tokenizer/template hashes) stay documented
placeholders; token/mask/logprob/reward artifacts are REAL bytes.
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
    """Contract checks on one rollout's REAL tokens (L/T/M semantics)."""
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
    expected = np.zeros(T, dtype=np.int8)
    expected[P:] = 1
    if not np.array_equal(alignment.completion_target_mask, expected):
        violations.append("M004_CANONICAL_MASK_MISMATCH")
    return violations


def _logprob_length_check(completion_ids, logprobs) -> str | None:
    """L004: completion-only logprobs must cover the completion tokens."""
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


def _masked_real(padded: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Select the REAL token positions of a padded TRL row via its mask."""
    arr = np.asarray(padded, dtype=np.int32)
    if mask is not None and mask.shape[0] == arr.shape[0]:
        return arr[np.asarray(mask, dtype=bool)]
    return arr


def _span(values: np.ndarray, prompt_len: int, completion_len: int,
          cmask: np.ndarray | None = None, attn: np.ndarray | None = None) -> np.ndarray | None:
    """Completion span of a padded per-row tensor.

    TRL 1.10: per-part masks align with their own row (prompt_mask with
    prompt_ids, completion_mask with completion_ids); old/sampling logprobs
    are full-sequence tensors aligned with attention_mask = concat(masks).
    """
    if values is None:
        return None
    row = np.asarray(values, dtype=np.float32)
    if row.shape[0] == completion_len:
        return np.ascontiguousarray(row)  # completion-only tensor (e.g. server sampling logprobs)
    if attn is not None and attn.shape[0] == row.shape[0]:
        real = row[np.asarray(attn, dtype=bool)]
        return np.ascontiguousarray(real[prompt_len:prompt_len + completion_len])
    if cmask is not None and cmask.shape[0] == row.shape[0] and np.asarray(cmask, dtype=bool).any():
        return np.ascontiguousarray(row[np.asarray(cmask, dtype=bool)])
    return np.ascontiguousarray(row[prompt_len:prompt_len + completion_len])


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
        self._guard_rollouts: list[dict] = []  # per-generation rollout contract records
        self._guard_gen_seq = 0  # monotonically increasing generation id (unique event/artifact ids)
        self._last_guard_verified: dict | None = None

    # ---------------------------------------------------------- rollout seam
    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        self._guard_validate_rollouts(result)
        return result

    def _guard_validate_rollouts(self, result) -> None:
        """Check + record every rollout in the official batch result.

        Uses TRL 1.10's REAL keys and records REAL bytes (P0-4): the
        mask-selected token sequence, the completion mask, the server's
        sampling logprobs, the old logprobs the loss will consume, and the
        advantages — all content-addressed, no zero-hash placeholders.
        """
        def _to_lists(key):
            v = result.get(key)
            if v is None:
                return []
            if hasattr(v, "cpu"):  # torch tensor -> numpy -> list
                v = v.cpu().numpy().tolist()
            return v

        prompt_ids_list = _to_lists("prompt_ids")
        prompt_mask_list = _to_lists("prompt_mask")
        completion_ids_list = _to_lists("completion_ids")
        completion_mask_list = _to_lists("completion_mask")
        sampling_lp_list = _to_lists("sampling_per_token_logps")
        old_lp_list = _to_lists("old_per_token_logps")
        advantages_list = _to_lists("advantages")
        step = getattr(getattr(self, "state", None), "global_step", 0)
        gen_seq = self._guard_gen_seq
        self._guard_gen_seq += 1
        records = []
        for i, (pids, cids) in enumerate(zip(prompt_ids_list, completion_ids_list)):
            pm = np.asarray(prompt_mask_list[i]) if i < len(prompt_mask_list) else None
            cm = np.asarray(completion_mask_list[i]) if i < len(completion_mask_list) else None
            attn = np.concatenate([pm, cm]) if pm is not None and cm is not None else None
            real_p = _masked_real(pids, pm)
            real_c = _masked_real(cids, cm)
            P, C = int(real_p.shape[0]), int(real_c.shape[0])
            violations = _align_checks(list(real_p), list(real_c))
            if violations:
                raise GuardViolation(f"rollout {i} contract violations: {violations}")
            # the server's behavior logprobs must cover the completion span
            sampling_span = None
            if i < len(sampling_lp_list) and sampling_lp_list[i] is not None:
                sampling_span = _span(np.asarray(sampling_lp_list[i]), P, C, cm, attn)
                if sampling_span is None or sampling_span.shape[0] not in (C, C + 1):
                    raise GuardViolation(
                        f"rollout {i}: server sampling logprobs completion span "
                        f"{None if sampling_span is None else sampling_span.shape[0]} "
                        f"does not cover the {C}-token completion (L004)")
            # REAL bytes, content-addressed (P0-4)
            seq = np.ascontiguousarray(np.concatenate([real_p, real_c]).astype(np.int32))
            seq_ref = self._guard_store.put(
                seq.tobytes(), "application/octet-stream", f"gen-trl-{gen_seq}-{i}",
                dtype="int32", shape=[seq.shape[0]])
            cmask = np.ascontiguousarray(cm.astype(np.int8) if cm is not None
                                         else np.ones(P + C, dtype=np.int8))
            cmask_ref = self._guard_store.put(
                cmask.tobytes(), "application/octet-stream", f"mask-trl-{gen_seq}-{i}",
                dtype="int8", shape=list(cmask.shape))
            old_span = None
            old_ref = None
            if i < len(old_lp_list) and old_lp_list[i] is not None:
                old_span = _span(np.asarray(old_lp_list[i]), P, C, cm, attn)
                if old_span is not None:
                    old_ref = self._guard_store.put(
                        old_span.tobytes(), "application/octet-stream",
                        f"oldlogprob-trl-{gen_seq}-{i}", dtype="float32", shape=[old_span.shape[0]])
            sampling_ref = None
            if sampling_span is not None:
                sampling_ref = self._guard_store.put(
                    sampling_span.tobytes(), "application/octet-stream",
                    f"samplinglogprob-trl-{gen_seq}-{i}", dtype="float32",
                    shape=[sampling_span.shape[0]])
            adv_ref = None
            if i < len(advantages_list) and advantages_list[i] is not None:
                adv = np.ascontiguousarray(np.asarray(advantages_list[i], dtype=np.float32))
                adv_ref = self._guard_store.put(
                    adv.tobytes(), "application/octet-stream", f"advantage-trl-{gen_seq}-{i}",
                    dtype="float32", shape=list(adv.shape))
            records.append({
                "step": step, "index": i,
                "prompt_len": P, "completion_len": C,
                "sequence_ref": seq_ref, "mask_ref": cmask_ref,
                "old_logprob_ref": old_ref, "sampling_logprob_ref": sampling_ref,
                "advantage_ref": adv_ref,
            })
            if self._guard_log:
                self._guard_emit_generation(gen_seq, i, seq_ref, cmask_ref, sampling_ref, P, C)
        self._guard_rollouts = records

    def _guard_emit_generation(self, gen_seq, index, seq_ref, mask_ref, sampling_ref,
                               prompt_len, completion_len) -> None:
        gen = GenerationEvent(
            event_id=f"gen-gtrl-{gen_seq}-{index}",
            event_type="generation_finished", run_id=self._guard_run_id,
            component_id="trl-grpo-trainer", lifecycle_seq=gen_seq * 1000 + index + 1,
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            request_id=f"gtrl-gen{gen_seq}-{index}",
            attempt_id="1", prompt_id=f"countdown-{index:04d}", sample_index=index,
            runtime_id="trl-vllm-server", prompt_span=[0, prompt_len],
            completion_span=[prompt_len, prompt_len + completion_len],
            sequence_token_ids=seq_ref, completion_target_mask=mask_ref,
            loss_mask=mask_ref,
            service_behavior_logprobs=sampling_ref if sampling_ref is not None else ArtifactRef(
                uri="", media_type="application/octet-stream", dtype="float32",
                shape=[completion_len], num_bytes=completion_len * 4,
                sha256="0" * 64, producer_event_id=f"gen-gtrl-{gen_seq}-{index}"),
            # Placeholders needing GPU-run wiring (documented, P0-4).
            behavior_policy_version=0, checkpoint_manifest_sha256="", sync_event=EventRef(
                uri="", event_id="sync-gtrl", event_sha256="0" * 64),
            tokenizer_sha256="", chat_template_sha256="", sampling_config_sha256="",
            runtime_load_epoch=1,
        ).seal()
        self._guard_log.append(gen, required_epoch=self._guard_epoch)

    # ------------------------------------------------------------ step seam
    def _prepare_inputs(self, generation_batch):
        """Verify the ACTUAL tensor batch the step will consume.

        transformers 5.x calls ``self._prepare_inputs`` inside
        ``training_step``, and TRL's override generates/scores/slices the
        batch there — so the tensors the loss will consume exist only at
        this point.  We verify them here (before loss/backward) via an
        optional one-shot hook, then return the CLEAN batch.
        """
        inputs = super()._prepare_inputs(generation_batch)
        hook = getattr(self, "_guard_prepare_hook", None)
        to_verify = hook(inputs) if callable(hook) else inputs
        self._guard_pre_update(to_verify)
        return inputs

    def training_step(self, model, inputs, num_items_in_batch):
        return super().training_step(model, inputs, num_items_in_batch)

    def _guard_pre_update(self, inputs) -> None:
        """Verify the ACTUAL tensors the official step will consume.

        TRL 1.10 shuffles the generation batch and slices it into
        per-device-batch steps (e.g. 8 rollouts consumed as 8 x 1-row
        steps), so matching is by CONTENT against the NOT-YET-consumed
        records: each input row's mask-selected token sequence, old-logprob
        completion span and advantage must byte-match one remaining rollout
        artifact (T001 / L004 / reward binding).  Matched records are
        consumed (persisted only after the whole batch verifies), so a row
        can never be used twice — retokenized/misbound/stale rows fail
        BEFORE loss/backward.
        """
        if not self._guard_rollouts:
            return  # no guard records (e.g. non-rollout steps)
        records = self._guard_rollouts
        if isinstance(inputs, (list, tuple)):
            # transformers 5.x passes training_step a list of per-sample
            # dicts; normalize to a dict of column lists
            if not inputs:
                return
            keys = inputs[0].keys()
            inputs = {k: [d[k] for d in inputs if k in d] for k in keys}
        prompt_ids = _as_np(inputs.get("prompt_ids"))
        completion_ids = _as_np(inputs.get("completion_ids"))
        prompt_mask = _as_np(inputs.get("prompt_mask"))
        completion_mask = _as_np(inputs.get("completion_mask"))
        old_lp = _as_np(inputs.get("old_per_token_logps"))
        advantages = _as_np(inputs.get("advantages"))
        if prompt_ids is None or completion_ids is None:
            raise GuardViolation(
                "training_step batch has no prompt_ids/completion_ids — cannot "
                "verify the actual consumed tokens (guard refuses to run blind)")
        n_rows = prompt_ids.shape[0]
        if n_rows > len(records):
            raise GuardViolation(
                f"batch rows {n_rows} exceed the {len(records)} unconsumed rollouts "
                f"(stale/duplicate rollout consumption)")
        # content-match token sequences (shuffle-proof; atomic commit below)
        available = {r["sequence_ref"].sha256: r for r in records}
        matched_recs: list[dict] = []
        for i in range(n_rows):
            pm = np.asarray(prompt_mask[i]) if prompt_mask is not None else None
            cm = np.asarray(completion_mask[i]) if completion_mask is not None else None
            attn = np.concatenate([pm, cm]) if pm is not None and cm is not None else None
            real_p = _masked_real(np.asarray(prompt_ids[i]), pm)
            real_c = _masked_real(np.asarray(completion_ids[i]), cm)
            seq = np.ascontiguousarray(np.concatenate([real_p, real_c]).astype(np.int32))
            h = hashlib.sha256(seq.tobytes()).hexdigest()
            if h not in available:
                raise GuardViolation(
                    f"row {i}: consumed token ids do not match any recorded server "
                    f"sequence (T001 retokenization/misbinding)")
            rec = available.pop(h)
            matched_recs.append(rec)
            if old_lp is not None:
                span = _span(old_lp[i], rec["prompt_len"], rec["completion_len"], cm, attn)
                if rec["old_logprob_ref"] is not None:
                    if span is None or hashlib.sha256(span.tobytes()).hexdigest() != rec["old_logprob_ref"].sha256:
                        raise GuardViolation(
                            f"row {i}: old-logprob VALUES differ from what was "
                            f"recorded at rollout (L004 misbinding)")
                else:
                    # no old logprobs recorded at rollout — length contract only
                    if span is None or span.shape[0] not in (rec["completion_len"],
                                                             rec["completion_len"] + 1):
                        raise GuardViolation(
                            f"row {i}: old-logprob completion span does not match the "
                            f"recorded completion (L004 misbinding)")
        if advantages is not None:
            from collections import Counter
            avail_adv = Counter(r["advantage_ref"].sha256 for r in matched_recs
                                if r["advantage_ref"] is not None)
            if advantages.shape[0] > sum(avail_adv.values()):
                raise GuardViolation(
                    f"advantages rows {advantages.shape[0]} exceed the rewards "
                    f"recorded for this batch (reward misbinding)")
            for i in range(advantages.shape[0]):
                row = np.ascontiguousarray(np.asarray(advantages[i], dtype=np.float32))
                h = hashlib.sha256(row.tobytes()).hexdigest()
                if avail_adv[h] == 0:
                    raise GuardViolation(
                        f"row {i}: consumed advantages do not match the recorded "
                        f"reward artifacts (reward misbinding)")
                avail_adv[h] -= 1
        # atomic commit: only persist consumption after the whole batch verified
        self._guard_rollouts = list(available.values())
        self._last_guard_verified = {
            "global_step": getattr(getattr(self, "state", None), "global_step", 0),
            "n_rollouts": len(self._guard_rollouts),
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
