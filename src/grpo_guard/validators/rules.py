"""Reason-coded validation rules (design doc §8).

Each rule returns RuleResult(code, decision, detail) or None when the check
does not apply to the envelope.  Decisions combine as: any reject → reject;
else any quarantine → quarantine; else allow.  ``allow`` is only produced
when every rule required for the stage has run to completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from grpo_guard.schema.decisions import Decision, PRE_UPDATE_STAGE_RULES
from grpo_guard.schema.events import (
    GenerationEvent,
    RewardEvent,
    ScoringEvent,
    UpdateInputEvent,
    SYNC_TERMINAL_SUCCESS,
)
from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
from grpo_guard.store.canonical_json import canonical_sha256
from grpo_guard.store.artifact_store import HashMismatchError
from grpo_guard.validators.align import AlignmentError, EmptyCompletionError, mask_from_artifact_bytes, reconstruct_alignment
from grpo_guard.validators.context import ValidationContext

RuleFn = Callable[[ValidationContext], "RuleResult | None"]


@dataclass
class RuleResult:
    code: str
    decision: Decision
    detail: str = ""


_BYTES_PER_ELEMENT = {
    "int8": 1, "uint8": 1, "int16": 2, "int32": 4, "int64": 8,
    "float16": 2, "bf16": 2, "float32": 4, "float64": 8,
}


def element_count(ref, data: bytes) -> int:
    """Number of elements in an artifact, from dtype metadata or shape."""
    bpe = _BYTES_PER_ELEMENT.get(ref.dtype or "")
    if bpe is not None:
        return len(data) // bpe
    if ref.shape:
        n = 1
        for d in ref.shape:
            n *= d
        return n
    return len(data)


def _gen(ctx: ValidationContext) -> GenerationEvent | None:
    ev = ctx.event(ctx.envelope.generation_event.event_id)
    if isinstance(ev, GenerationEvent):
        return ev
    if ev is not None:
        ctx.notes.append(f"generation event {ev.event_id} is type {ev.event_type}")
    return None


def _manifest(ctx: ValidationContext) -> PolicyManifest | None:
    return ctx.policy_manifest


def _split(ctx: ValidationContext) -> SplitManifest | None:
    return ctx.split_manifest


# ---------------------------------------------------------------- P: policy

def p001_missing_policy_manifest(ctx: ValidationContext) -> RuleResult | None:
    if ctx.policy_manifest is None:
        return RuleResult("P001_MISSING_POLICY_MANIFEST", "reject", "no policy manifest bound to generation")
    return None


def p002_checkpoint_hash_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen, man = _gen(ctx), _manifest(ctx)
    if gen is None or man is None:
        return None
    if gen.checkpoint_manifest_sha256 != man.checkpoint_manifest_sha256:
        return RuleResult(
            "P002_CHECKPOINT_HASH_MISMATCH", "reject",
            f"generation checkpoint {gen.checkpoint_manifest_sha256[:12]} != manifest {man.checkpoint_manifest_sha256[:12]}",
        )
    return None


def p003_missing_sync_event(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    sync = ctx.event(gen.sync_event.event_id)
    if sync is None:
        return RuleResult("P003_MISSING_SYNC_EVENT", "quarantine", "generation references no control-plane sync event")
    if sync.event_type not in SYNC_TERMINAL_SUCCESS:
        return RuleResult(
            "P003_MISSING_SYNC_EVENT", "quarantine",
            f"sync event is {sync.event_type}, not a terminal success",
        )
    return None


def _policy_lag(ctx: ValidationContext) -> int | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    parent = ctx.envelope.training_contract.trainer_parent_policy_version
    return max(0, parent - gen.behavior_policy_version)


def p004_stale_policy_strict(ctx: ValidationContext) -> RuleResult | None:
    if ctx.protocol.mode != "strict_on_policy":
        return None
    lag = _policy_lag(ctx)
    if lag is None:
        return None
    if lag > 0:
        return RuleResult("P004_STALE_POLICY_STRICT", "reject", f"strict on-policy lag={lag}")
    return None


def p005_lag_exceeds_bound(ctx: ValidationContext) -> RuleResult | None:
    if ctx.protocol.mode != "bounded_off_policy":
        return None
    lag = _policy_lag(ctx)
    if lag is None:
        return None
    if lag > ctx.protocol.max_policy_lag_versions:
        return RuleResult(
            "P005_LAG_EXCEEDS_BOUND", "reject",
            f"lag={lag} > bound={ctx.protocol.max_policy_lag_versions}",
        )
    return None


def p006_correction_undeclared(ctx: ValidationContext) -> RuleResult | None:
    if ctx.protocol.mode != "bounded_off_policy":
        return None
    if not ctx.protocol.importance_correction:
        return RuleResult("P006_CORRECTION_UNDECLARED", "reject", "bounded mode without importance-correction config")
    return None


def p007_event_order_invalid(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    sync = ctx.event(gen.sync_event.event_id)
    if sync is not None and sync.lifecycle_seq > gen.lifecycle_seq:
        return RuleResult("P007_EVENT_ORDER_INVALID", "reject", "sync event after generation")
    if ctx.update_input_event is not None:
        upd = ctx.update_input_event
        if upd.lifecycle_seq < gen.lifecycle_seq:
            return RuleResult("P007_EVENT_ORDER_INVALID", "reject", "update input before generation")
    return None


def p008_canary_mismatch(ctx: ValidationContext) -> RuleResult | None:
    if ctx.canary_status == "mismatch":
        return RuleResult("P008_CANARY_MISMATCH", "reject", "fixed-env canary outside tolerance")
    return None


# ---------------------------------------------------------------- T: tokens

def t001_artifact_hash_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    refs = [gen.sequence_token_ids, gen.completion_target_mask, gen.loss_mask]
    if gen.service_behavior_logprobs is not None:
        refs.append(gen.service_behavior_logprobs)
    for ref in refs:
        if not ctx.store.verify(ref):
            return RuleResult("T001_ARTIFACT_HASH_MISMATCH", "reject", f"artifact {ref.sha256[:12]} fails content hash")
    return None


def t002_tokenizer_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen, man = _gen(ctx), _manifest(ctx)
    if gen is None or man is None:
        return None
    if gen.tokenizer_sha256 != man.tokenizer_sha256:
        return RuleResult("T002_TOKENIZER_MISMATCH", "reject", "generation tokenizer hash != manifest tokenizer hash")
    return None


def t003_chat_template_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen, man = _gen(ctx), _manifest(ctx)
    if gen is None or man is None:
        return None
    if gen.chat_template_sha256 != man.chat_template_sha256:
        return RuleResult("T003_CHAT_TEMPLATE_MISMATCH", "reject", "generation chat template hash != manifest")
    return None


def t004_token_sequence_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None or ctx.update_input_event is None:
        return None
    upd = ctx.update_input_event
    if not isinstance(upd, UpdateInputEvent):
        return None
    if upd.sequence_token_ids.sha256 != gen.sequence_token_ids.sha256:
        return RuleResult(
            "T004_TOKEN_SEQUENCE_MISMATCH", "reject",
            f"update input sequence {upd.sequence_token_ids.sha256[:12]} != producer {gen.sequence_token_ids.sha256[:12]}",
        )
    return None


def t005_span_out_of_range(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    try:
        _alignment(ctx)
    except EmptyCompletionError:
        return None  # M005 handles empty completions
    except AlignmentError as exc:
        return RuleResult("T005_SPAN_OUT_OF_RANGE", "reject", str(exc))
    return None


# ---------------------------------------------------------------- M: masks

def _alignment(ctx: ValidationContext):
    gen = _gen(ctx)
    if gen is None:
        raise AlignmentError("no generation event")
    try:
        seq = ctx.store.get(gen.sequence_token_ids)
    except (HashMismatchError, FileNotFoundError) as exc:
        # unreadable artifact → T001 fires; alignment rules skip
        raise AlignmentError(f"sequence artifact unreadable: {exc}") from exc
    T = element_count(gen.sequence_token_ids, seq)
    return reconstruct_alignment(
        T,
        gen.completion_span[0],
        gen.completion_span[1],
        padding_spans=gen.padding_spans,
    )


def m001_mask_shape_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    try:
        align = _alignment(ctx)
    except EmptyCompletionError:
        return None  # M005 handles empty completions
    except AlignmentError as exc:
        return RuleResult("M001_MASK_SHAPE_MISMATCH", "reject", str(exc))
    try:
        target = mask_from_artifact_bytes(ctx.store.get(gen.completion_target_mask), align.T)
        loss = mask_from_artifact_bytes(ctx.store.get(gen.loss_mask), align.T - 1)
    except (HashMismatchError, FileNotFoundError):
        return None  # T001 catches corrupt artifacts
    if target.size != align.T or loss.size != align.T - 1:
        return RuleResult("M001_MASK_SHAPE_MISMATCH", "reject", "mask lengths disagree with sequence length")
    if gen.service_behavior_logprobs is not None:
        lp_data = ctx.store.get(gen.service_behavior_logprobs)
        if element_count(gen.service_behavior_logprobs, lp_data) != align.C:
            return RuleResult("M001_MASK_SHAPE_MISMATCH", "reject", "service logprob length != C")
    return None


def _mask_violation(ctx: ValidationContext, check: str) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    try:
        align = _alignment(ctx)
    except AlignmentError:
        return None
    try:
        target = mask_from_artifact_bytes(ctx.store.get(gen.completion_target_mask), align.T)
    except (HashMismatchError, FileNotFoundError):
        return None  # T001 catches corrupt artifacts
    if check == "prompt" and target[: align.P].any():
        return RuleResult("M002_PROMPT_SELECTED", "reject", f"{int(target[: align.P].sum())} target tokens inside prompt")
    if check == "padding":
        for s, e in gen.padding_spans:
            if target[s:e].any():
                return RuleResult("M003_PADDING_SELECTED", "reject", f"target mask selects padding [{s},{e})")
    if check == "canonical":
        loss = mask_from_artifact_bytes(ctx.store.get(gen.loss_mask), align.T - 1)
        if not (target == align.completion_target_mask).all() or not (loss == align.loss_mask).all():
            return RuleResult("M004_CANONICAL_MASK_MISMATCH", "reject", "producer masks differ from span-reconstructed canonical masks")
    return None


def m002_prompt_selected(ctx: ValidationContext) -> RuleResult | None:
    return _mask_violation(ctx, "prompt")


def m003_padding_selected(ctx: ValidationContext) -> RuleResult | None:
    return _mask_violation(ctx, "padding")


def m004_canonical_mask_mismatch(ctx: ValidationContext) -> RuleResult | None:
    return _mask_violation(ctx, "canonical")


def m005_empty_completion(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    if gen.completion_span[1] - gen.completion_span[0] <= 0:
        return RuleResult("M005_EMPTY_COMPLETION", "quarantine", "empty completion span")
    return None


# ---------------------------------------------------------------- L: logprobs

def _scoring(ctx: ValidationContext) -> ScoringEvent | None:
    if ctx.envelope.scoring_event is None:
        return None
    ev = ctx.event(ctx.envelope.scoring_event.event_id)
    return ev if isinstance(ev, ScoringEvent) else None


def _authoritative_ref(ctx: ValidationContext):
    return ctx.envelope.training_contract.authoritative_behavior_logprob_event


def l001_missing_behavior_logprob(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    has_service = gen.service_behavior_logprobs is not None
    has_scoring = _scoring(ctx) is not None
    if not has_service and not has_scoring:
        return RuleResult("L001_MISSING_BEHAVIOR_LOGPROB", "reject", "loss requires behavior logprobs, none present")
    return None


def l002_source_event_mismatch(ctx: ValidationContext) -> RuleResult | None:
    ref = _authoritative_ref(ctx)
    if ref is None:
        return RuleResult("L002_SOURCE_EVENT_MISMATCH", "reject", "no authoritative logprob event declared")
    ev = ctx.event(ref.event_id)
    if ev is None:
        return RuleResult("L002_SOURCE_EVENT_MISMATCH", "reject", f"authoritative event {ref.event_id} missing")
    declared = ctx.envelope.training_contract.behavior_logprob_source
    gen = _gen(ctx)
    if declared == "generation_service":
        if not isinstance(ev, GenerationEvent) or ev.event_id != ctx.envelope.generation_event.event_id:
            return RuleResult("L002_SOURCE_EVENT_MISMATCH", "reject", "generation_service source but event is not the generation event")
        if gen is None or gen.service_behavior_logprobs is None:
            return RuleResult("L002_SOURCE_EVENT_MISMATCH", "reject", "generation event lacks service logprobs")
    elif declared == "exact_behavior_scorer":
        if not isinstance(ev, ScoringEvent):
            return RuleResult("L002_SOURCE_EVENT_MISMATCH", "reject", "exact_behavior_scorer source but event is not a scoring event")
    else:
        return RuleResult("L002_SOURCE_EVENT_MISMATCH", "reject", f"unknown source {declared}")
    return None


def l003_scorer_policy_mismatch(ctx: ValidationContext) -> RuleResult | None:
    scoring = _scoring(ctx)
    if scoring is None:
        return None
    gen = _gen(ctx)
    if gen is None:
        return None
    if scoring.scorer_policy_version != gen.behavior_policy_version:
        return RuleResult("L003_SCORER_POLICY_MISMATCH", "reject",
                          f"scorer policy {scoring.scorer_policy_version} != behavior {gen.behavior_policy_version}")
    if scoring.scorer_checkpoint_manifest_sha256 != gen.checkpoint_manifest_sha256:
        return RuleResult("L003_SCORER_POLICY_MISMATCH", "reject", "scorer checkpoint != behavior checkpoint")
    if scoring.token_artifact_sha256 != gen.sequence_token_ids.sha256:
        return RuleResult("L003_SCORER_POLICY_MISMATCH", "reject", "scorer token artifact != generation token artifact")
    return None


def l004_token_logprob_length_mismatch(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    try:
        align = _alignment(ctx)
    except AlignmentError:
        return None
    if align.padding_within_completion:
        return RuleResult("L004_TOKEN_LOGPROB_LENGTH_MISMATCH", "quarantine", "padding inside completion span needs explicit protocol branch")
    scoring = _scoring(ctx)
    if scoring is not None:
        try:
            lp_data = ctx.store.get(scoring.behavior_logprobs)
        except (HashMismatchError, FileNotFoundError):
            return None
        if element_count(scoring.behavior_logprobs, lp_data) != align.C:
            return RuleResult("L004_TOKEN_LOGPROB_LENGTH_MISMATCH", "reject", f"scorer logprob length != C={align.C}")
    if gen.service_behavior_logprobs is not None:
        try:
            lp_data = ctx.store.get(gen.service_behavior_logprobs)
        except (HashMismatchError, FileNotFoundError):
            return None
        if element_count(gen.service_behavior_logprobs, lp_data) != align.C:
            return RuleResult("L004_TOKEN_LOGPROB_LENGTH_MISMATCH", "reject", f"service logprob length != C={align.C}")
    return None


def l005_scoring_after_update(ctx: ValidationContext) -> RuleResult | None:
    scoring = _scoring(ctx)
    if scoring is None:
        return None
    if ctx.update_input_event is not None and scoring.lifecycle_seq > ctx.update_input_event.lifecycle_seq:
        return RuleResult("L005_SCORING_AFTER_UPDATE", "reject", "scoring occurred after consuming update materialization")
    return None


def l006_unsupported_provenance(ctx: ValidationContext) -> RuleResult | None:
    scoring = _scoring(ctx)
    if scoring is not None and scoring.source_generation_event.event_id != ctx.envelope.generation_event.event_id:
        return RuleResult("L006_UNSUPPORTED_PROVENANCE", "quarantine", "scoring event does not reference this generation")
    return None


def l007_authoritative_source_ambiguous(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    if gen is None:
        return None
    has_service = gen.service_behavior_logprobs is not None
    has_scoring = _scoring(ctx) is not None
    if has_service and has_scoring and not ctx.envelope.training_contract.diagnostic_non_authoritative_logprobs_allowed:
        return RuleResult("L007_AUTHORITATIVE_SOURCE_AMBIGUOUS", "reject", "two logprob sources both present")
    return None


def l008_nonauthoritative_logprob_consumed(ctx: ValidationContext) -> RuleResult | None:
    upd = ctx.update_input_event
    if upd is None or not isinstance(upd, UpdateInputEvent):
        return None
    auth = _authoritative_ref(ctx)
    if auth is None or upd.authoritative_behavior_logprob_event.event_id != auth.event_id:
        return RuleResult("L008_NONAUTHORITATIVE_LOGPROB_CONSUMED", "reject",
                          "materializer consumed a non-authoritative logprob event")
    return None


# ---------------------------------------------------------------- D/R: data + reward

def d001_split_manifest_missing(ctx: ValidationContext) -> RuleResult | None:
    if ctx.split_manifest is None:
        return RuleResult("D001_SPLIT_MANIFEST_MISSING", "reject", "no split manifest bound")
    return None


def d002_prompt_not_in_declared_split(ctx: ValidationContext) -> RuleResult | None:
    gen = _gen(ctx)
    split = _split(ctx)
    if gen is None or split is None:
        return None
    if gen.prompt_id not in split.prompt_ids:
        return RuleResult("D002_PROMPT_NOT_IN_DECLARED_SPLIT", "reject",
                          f"prompt {gen.prompt_id} not in split {split.split_name}")
    return None


def _reward(ctx: ValidationContext) -> RewardEvent | None:
    if ctx.envelope.reward_event is None:
        return None
    ev = ctx.event(ctx.envelope.reward_event.event_id)
    return ev if isinstance(ev, RewardEvent) else None


def r001_reward_protocol_missing(ctx: ValidationContext) -> RuleResult | None:
    reward = _reward(ctx)
    if reward is None:
        return None
    if not reward.evaluator_protocol_sha256:
        return RuleResult("R001_REWARD_PROTOCOL_MISSING", "quarantine", "reward event without protocol hash")
    return None


def r002_infra_error_as_task_fail(ctx: ValidationContext) -> RuleResult | None:
    reward = _reward(ctx)
    if reward is None:
        return None
    if reward.terminal_status in ("infra_error", "timeout", "invalid"):
        if any(abs(v) > 1e-9 for v in reward.components.values()):
            return RuleResult("R002_INFRA_ERROR_AS_TASK_FAIL", "reject",
                              f"non-zero reward on {reward.terminal_status}")
    return None


def r003_reward_missing_pre_update(ctx: ValidationContext) -> RuleResult | None:
    if ctx.envelope.envelope_stage != "pre_update":
        return None
    if ctx.envelope.reward_event is None:
        return RuleResult("R003_REWARD_MISSING_PRE_UPDATE", "reject", "pre-update envelope without reward event")
    return None


def r004_reward_present_pre_reward(ctx: ValidationContext) -> RuleResult | None:
    if ctx.envelope.envelope_stage != "pre_reward":
        return None
    if ctx.envelope.reward_event is not None:
        return RuleResult("R004_REWARD_PRESENT_PRE_REWARD", "reject", "pre-reward envelope already carries reward")
    return None


def r005_parent_identity_not_allowed(ctx: ValidationContext) -> RuleResult | None:
    if ctx.envelope.envelope_stage != "pre_update":
        return None
    if ctx.envelope.parent_identity_decision is None:
        return RuleResult("R005_PARENT_IDENTITY_NOT_ALLOWED", "reject", "pre-update envelope without parent identity decision")
    parent = ctx.event(ctx.envelope.parent_identity_decision.event_id)
    if parent is None:
        return RuleResult("R005_PARENT_IDENTITY_NOT_ALLOWED", "reject", "parent identity decision event missing")
    payload = getattr(parent, "decision_payload", None)
    if payload is None or payload.decision != "allow":
        return RuleResult("R005_PARENT_IDENTITY_NOT_ALLOWED", "reject", "parent identity decision not ALLOW")
    return None


ALL_RULES: dict[str, RuleFn] = {
    "P001_MISSING_POLICY_MANIFEST": p001_missing_policy_manifest,
    "P002_CHECKPOINT_HASH_MISMATCH": p002_checkpoint_hash_mismatch,
    "P003_MISSING_SYNC_EVENT": p003_missing_sync_event,
    "P004_STALE_POLICY_STRICT": p004_stale_policy_strict,
    "P005_LAG_EXCEEDS_BOUND": p005_lag_exceeds_bound,
    "P006_CORRECTION_UNDECLARED": p006_correction_undeclared,
    "P007_EVENT_ORDER_INVALID": p007_event_order_invalid,
    "P008_CANARY_MISMATCH": p008_canary_mismatch,
    "T001_ARTIFACT_HASH_MISMATCH": t001_artifact_hash_mismatch,
    "T002_TOKENIZER_MISMATCH": t002_tokenizer_mismatch,
    "T003_CHAT_TEMPLATE_MISMATCH": t003_chat_template_mismatch,
    "T004_TOKEN_SEQUENCE_MISMATCH": t004_token_sequence_mismatch,
    "T005_SPAN_OUT_OF_RANGE": t005_span_out_of_range,
    "M001_MASK_SHAPE_MISMATCH": m001_mask_shape_mismatch,
    "M002_PROMPT_SELECTED": m002_prompt_selected,
    "M003_PADDING_SELECTED": m003_padding_selected,
    "M004_CANONICAL_MASK_MISMATCH": m004_canonical_mask_mismatch,
    "M005_EMPTY_COMPLETION": m005_empty_completion,
    "L001_MISSING_BEHAVIOR_LOGPROB": l001_missing_behavior_logprob,
    "L002_SOURCE_EVENT_MISMATCH": l002_source_event_mismatch,
    "L003_SCORER_POLICY_MISMATCH": l003_scorer_policy_mismatch,
    "L004_TOKEN_LOGPROB_LENGTH_MISMATCH": l004_token_logprob_length_mismatch,
    "L005_SCORING_AFTER_UPDATE": l005_scoring_after_update,
    "L006_UNSUPPORTED_PROVENANCE": l006_unsupported_provenance,
    "L007_AUTHORITATIVE_SOURCE_AMBIGUOUS": l007_authoritative_source_ambiguous,
    "L008_NONAUTHORITATIVE_LOGPROB_CONSUMED": l008_nonauthoritative_logprob_consumed,
    "D001_SPLIT_MANIFEST_MISSING": d001_split_manifest_missing,
    "D002_PROMPT_NOT_IN_DECLARED_SPLIT": d002_prompt_not_in_declared_split,
    "R001_REWARD_PROTOCOL_MISSING": r001_reward_protocol_missing,
    "R002_INFRA_ERROR_AS_TASK_FAIL": r002_infra_error_as_task_fail,
    "R003_REWARD_MISSING_PRE_UPDATE": r003_reward_missing_pre_update,
    "R004_REWARD_PRESENT_PRE_REWARD": r004_reward_present_pre_reward,
    "R005_PARENT_IDENTITY_NOT_ALLOWED": r005_parent_identity_not_allowed,
}


def ruleset_sha256() -> str:
    """Hash the rule table: the checked ruleset identity bound to decisions."""
    return canonical_sha256({code: fn.__name__ for code, fn in sorted(ALL_RULES.items())})
