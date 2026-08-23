"""Rule-level contract tests: every reason code fires exactly when it should
(design doc §8).  Helper ``validate`` runs the identity stage by default.
"""

import numpy as np
import pytest

from grpo_guard import testing
from grpo_guard.faults import (
    inject_f1_static_rollout,
    inject_f2_misbound_logprob,
    inject_f3_retokenization,
    inject_f3_retokenized_sequence,
    inject_f3_template_variant,
    inject_f4_mask_shift,
)
from grpo_guard.schema.artifacts import EventRef, ManifestRef
from grpo_guard.schema.decisions import PRE_UPDATE_STAGE_RULES, ValidationDecision
from grpo_guard.schema.events import GenerationEvent, RewardEvent, SyncEvent, UpdateInputEvent
from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.store.canonical_json import canonical_sha256
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

STRICT = ProtocolConfig(name="strict_v01", mode="strict_on_policy")
BOUNDED = ProtocolConfig(name="bounded_v01", mode="bounded_off_policy", max_policy_lag_versions=2,
                         importance_correction="importance-ratio-v1")


def make_ctx(t, protocol=STRICT, **kw) -> ValidationContext:
    return ValidationContext(
        envelope=t.envelope,
        store=t.store,
        events=t.events,
        policy_manifest=kw.pop("policy_manifest", t.policy_manifest),
        split_manifest=kw.pop("split_manifest", t.split_manifest),
        protocol=protocol,
        **kw,
    )


def decision_codes(t, protocol=STRICT, stage="identity_pre_reward", **kw):
    dec = validate_envelope(make_ctx(t, protocol=protocol, **kw), stage)
    return dec.decision_payload


# ---------------------------------------------------------------- P rules

def test_p001_missing_policy_manifest():
    t = testing.build_trajectory()
    d = decision_codes(t, policy_manifest=None)
    assert d.decision == "reject"
    assert "P001_MISSING_POLICY_MANIFEST" in d.reason_codes


def test_p002_checkpoint_hash_mismatch():
    t = testing.build_trajectory()
    other = testing.make_policy_manifest(99)
    d = decision_codes(t, policy_manifest=other)
    assert d.decision == "reject"
    assert "P002_CHECKPOINT_HASH_MISMATCH" in d.reason_codes


def test_p003_missing_sync_event():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    req = t.sync_events[0]  # sync_requested is not a terminal success
    new_gen = gen.model_copy(update={
        "event_id": f"{gen.event_id}-nosync",
        "sync_event": EventRef(uri="event://req", event_id=req.event_id, event_sha256=req.event_sha256),
    })
    new_gen.event_sha256 = ""
    new_gen = new_gen.seal()
    t.events[new_gen.event_id] = new_gen
    new_ref = EventRef(uri="", event_id=new_gen.event_id, event_sha256=new_gen.event_sha256)
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-p3",
        "generation_event": new_ref,
        "training_contract": t.envelope.training_contract.model_copy(
            update={"authoritative_behavior_logprob_event": new_ref}
        ),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    d = decision_codes(t)
    assert "P003_MISSING_SYNC_EVENT" in d.reason_codes
    assert d.decision == "quarantine"


def test_p004_stale_policy_strict():
    t = testing.build_trajectory()
    t2 = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
    d = decision_codes(t2)
    assert d.decision == "reject"
    assert "P004_STALE_POLICY_STRICT" in d.reason_codes
    assert d.observed_policy_lag == 1


def test_p005_lag_exceeds_bound():
    t = testing.build_trajectory()
    t2 = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=5)
    d = decision_codes(t2, protocol=BOUNDED)
    assert d.decision == "reject"
    assert "P005_LAG_EXCEEDS_BOUND" in d.reason_codes


def test_p005_lag_within_bound_allows_bounded():
    t = testing.build_trajectory(contract_protocol="bounded_off_policy",
                                 max_policy_lag_versions=2, importance_correction="importance-ratio-v1")
    t2 = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
    d = decision_codes(t2, protocol=BOUNDED)
    assert d.decision == "allow"


def test_p009_contract_protocol_mismatch_rejects():
    # contract claims strict, validator runs bounded -> P009
    t = testing.build_trajectory(contract_protocol="strict_on_policy")
    t2 = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
    d = decision_codes(t2, protocol=BOUNDED)
    assert d.decision == "reject"
    assert "P009_CONTRACT_PROTOCOL_MISMATCH" in d.reason_codes


def test_p006_correction_undeclared():
    t = testing.build_trajectory()
    t2 = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
    no_correction = ProtocolConfig(name="bounded-noisy", mode="bounded_off_policy", max_policy_lag_versions=2)
    d = decision_codes(t2, protocol=no_correction)
    assert "P006_CORRECTION_UNDECLARED" in d.reason_codes


def test_p008_canary_mismatch():
    t = testing.build_trajectory()
    d = decision_codes(t, canary_status="mismatch")
    assert "P008_CANARY_MISMATCH" in d.reason_codes


# ---------------------------------------------------------------- T rules

def test_t001_artifact_hash_mismatch():
    # dedicated store: tampering must not poison the shared global store
    with testing.ArtifactStoreTmp() as store:
        t = testing.build_trajectory(store)
        gen = t.events[t.envelope.generation_event.event_id]
        (t.store.blobs / gen.sequence_token_ids.sha256).write_bytes(b"tampered-bytes")
        d = decision_codes(t)
        assert "T001_ARTIFACT_HASH_MISMATCH" in d.reason_codes


def test_t002_tokenizer_mismatch():
    t = testing.build_trajectory()
    t2 = inject_f3_retokenization(t)
    d = decision_codes(t2)
    assert "T002_TOKENIZER_MISMATCH" in d.reason_codes


def test_t003_template_mismatch():
    t = testing.build_trajectory()
    t2 = inject_f3_template_variant(t)
    d = decision_codes(t2)
    assert "T003_CHAT_TEMPLATE_MISMATCH" in d.reason_codes


def test_t004_sequence_mismatch_at_materialization():
    t = testing.build_trajectory()
    t2 = inject_f3_retokenized_sequence(t, "b" * 64)
    upd = _update_input_with_sequence(t2, t2.bogus_sequence_ref)
    d = decision_codes(t2, stage="full_pre_update", update_input_event=upd)
    assert "T004_TOKEN_SEQUENCE_MISMATCH" in d.reason_codes


def _update_input_with_sequence(t, seq_ref):
    gen = t.events[t.envelope.generation_event.event_id]
    return UpdateInputEvent(
        event_id="uinput-test", run_id=t.run_id, component_id="materializer",
        lifecycle_seq=t.next_seq(), created_at_utc=testing.now_utc(),
        update_id="update-1",
        preupdate_envelope=t.envelope.ref(),
        preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
        sequence_token_ids=seq_ref,
        loss_mask=gen.loss_mask,
        authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
        authoritative_behavior_logprobs=gen.service_behavior_logprobs or seq_ref,
        reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64,
        single_use_nonce_sha256="0" * 64,
        tokenizer_called=False,
    ).seal()


def test_t005_span_out_of_range():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    bad = gen.model_copy(deep=True).model_copy(
        update={"event_id": f"{gen.event_id}-badspan", "completion_span": [4, 99]}
    )
    bad.event_sha256 = ""
    bad = bad.seal()
    t.events[bad.event_id] = bad
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-t5",
        "generation_event": EventRef(uri="", event_id=bad.event_id, event_sha256=bad.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    d = decision_codes(t)
    assert "T005_SPAN_OUT_OF_RANGE" in d.reason_codes


# ---------------------------------------------------------------- M rules

def test_m002_m004_mask_shift_fires():
    t = testing.build_trajectory()
    t2 = inject_f4_mask_shift(t, shift=1)
    d = decision_codes(t2)
    assert d.decision == "reject"
    codes = set(d.reason_codes)
    assert "M002_PROMPT_SELECTED" in codes
    assert "M004_CANONICAL_MASK_MISMATCH" in codes


def test_m003_padding_selected():
    t = testing.build_trajectory(
                                 padding_spans=[[5, 7]])
    # factory zeroes padding in the mask, so inject the fault: mask includes padding
    t2 = inject_f4_mask_shift(t, shift=0)
    # shift 0 == no-op; craft padding overlap manually
    t2 = _mask_with_padding(t, pad_span=[5, 7])
    d = decision_codes(t2)
    assert "M003_PADDING_SELECTED" in d.reason_codes


def _mask_with_padding(t, pad_span):
    gen = t.events[t.envelope.generation_event.event_id]
    T = t.sequence.shape[0]
    mask = np.ones(T, dtype=np.int8)
    mask[: gen.completion_span[0]] = 0
    mask[pad_span[0]: pad_span[1]] = 1  # fault: padding left in mask
    new_ref = t.store.put(mask.tobytes(), "application/octet-stream", f"{gen.event_id}-pad", dtype="int8", shape=[T])
    new_gen = gen.model_copy(deep=True).model_copy(update={
        "event_id": f"{gen.event_id}-pad", "completion_target_mask": new_ref, "padding_spans": [pad_span],
    })
    new_gen.event_sha256 = ""
    new_gen = new_gen.seal()
    t.events[new_gen.event_id] = new_gen
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-pad",
        "generation_event": EventRef(uri="", event_id=new_gen.event_id, event_sha256=new_gen.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    return t


def test_m005_empty_completion():
    t = testing.build_trajectory(completion_span=[4, 4])
    d = decision_codes(t)
    assert "M005_EMPTY_COMPLETION" in d.reason_codes
    assert d.decision == "quarantine"


# ---------------------------------------------------------------- L rules

def test_l001_missing_behavior_logprob():
    t = testing.build_trajectory(logprob_source="none")
    d = decision_codes(t)
    assert "L001_MISSING_BEHAVIOR_LOGPROB" in d.reason_codes


def test_l003_scorer_policy_mismatch():
    t = testing.build_trajectory()
    t2 = inject_f2_misbound_logprob(t, scorer_policy_version=1)
    d = decision_codes(t2)
    assert "L003_SCORER_POLICY_MISMATCH" in d.reason_codes


def test_l007_ambiguous_dual_source():
    t = testing.build_trajectory()
    t2 = inject_f2_misbound_logprob(t, scorer_policy_version=0)  # same policy but two sources
    d = decision_codes(t2)
    assert "L007_AUTHORITATIVE_SOURCE_AMBIGUOUS" in d.reason_codes


def test_l007_diagnostic_allowed_does_not_fire():
    t = testing.build_trajectory(diagnostic_non_authoritative_allowed=True)
    t2 = inject_f2_misbound_logprob(t, scorer_policy_version=0)
    d = decision_codes(t2)
    assert d.decision == "allow"  # dual source tolerated as diagnostic only


def test_l004_logprob_length_mismatch():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    short = np.full(3, -0.5, dtype=np.float16)
    short_ref = t.store.put(short.tobytes(), "application/octet-stream", f"{gen.event_id}-short", dtype="bf16", shape=[3])
    new_gen = gen.model_copy(deep=True).model_copy(
        update={"event_id": f"{gen.event_id}-shortlp", "service_behavior_logprobs": short_ref}
    )
    new_gen.event_sha256 = ""
    new_gen = new_gen.seal()
    t.events[new_gen.event_id] = new_gen
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-l4",
        "generation_event": EventRef(uri="", event_id=new_gen.event_id, event_sha256=new_gen.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    d = decision_codes(t)
    assert "L004_TOKEN_LOGPROB_LENGTH_MISMATCH" in d.reason_codes


def test_l006_unsupported_provenance():
    t = testing.build_trajectory(diagnostic_non_authoritative_allowed=True)
    t2 = inject_f2_misbound_logprob(t, scorer_policy_version=0)
    # point the scoring event at a different generation
    scoring = t2.events[t2.envelope.scoring_event.event_id]
    other_gen = testing.build_trajectory(t2.store, policy_version=0, run_id="run-other")
    other = other_gen.events[other_gen.envelope.generation_event.event_id]
    wrong = scoring.model_copy(deep=True).model_copy(update={
        "event_id": f"{scoring.event_id}-wrongref",
        "source_generation_event": EventRef(uri="", event_id=other.event_id, event_sha256=other.event_sha256),
    })
    wrong.event_sha256 = ""
    wrong = wrong.seal()
    t2.events[wrong.event_id] = wrong
    from grpo_guard.schema.artifacts import EventRef as ER
    t2.envelope = t2.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t2.envelope.envelope_id}-l6",
        "scoring_event": ER(uri="", event_id=wrong.event_id, event_sha256=wrong.event_sha256),
    })
    t2.envelope.envelope_sha256 = ""
    t2.envelope = t2.envelope.seal()
    d = decision_codes(t2)
    assert "L006_UNSUPPORTED_PROVENANCE" in d.reason_codes


def test_l008_nonauthoritative_consumed():
    t = testing.build_trajectory()
    t2 = inject_f2_misbound_logprob(t, scorer_policy_version=0)
    auth = t2.envelope.training_contract.authoritative_behavior_logprob_event
    upd = _update_input_with_sequence(t2, t2.sequence_ref)
    # materializer consumes the generation event instead of the authoritative scoring event
    upd = upd.model_copy(update={
        "event_id": "uinput-l8",
        "authoritative_behavior_logprob_event": EventRef(uri="", event_id="gen-x", event_sha256="0" * 64),
    })
    upd.event_sha256 = ""
    upd = upd.seal()
    d = decision_codes(t2, stage="full_pre_update", update_input_event=upd)
    assert "L008_NONAUTHORITATIVE_LOGPROB_CONSUMED" in d.reason_codes


# ---------------------------------------------------------------- D/R rules

def test_d001_split_manifest_missing():
    t = testing.build_trajectory()
    d = decision_codes(t, stage="full_pre_update", split_manifest=None)
    assert "D001_SPLIT_MANIFEST_MISSING" in d.reason_codes


def test_d002_prompt_not_in_split():
    t = testing.build_trajectory()
    empty = SplitManifest(split_id="split-train", split_name="train", prompt_ids=["countdown-other"])
    d = decision_codes(t, stage="full_pre_update", split_manifest=empty)
    assert "D002_PROMPT_NOT_IN_DECLARED_SPLIT" in d.reason_codes


def test_r003_reward_missing_pre_update():
    t = testing.build_trajectory(stage="pre_update")
    # strip the reward reference: pre-update envelope without RewardEvent
    t.envelope = t.envelope.model_copy(deep=True).model_copy(
        update={"envelope_id": f"{t.envelope.envelope_id}-r3", "reward_event": None}
    )
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    d = decision_codes(t, stage="full_pre_update")
    assert "R003_REWARD_MISSING_PRE_UPDATE" in d.reason_codes


def test_r004_reward_present_pre_reward():
    t = testing.build_trajectory(stage="pre_update")
    d = decision_codes(t, stage="identity_pre_reward")
    assert "R004_REWARD_PRESENT_PRE_REWARD" in d.reason_codes


def test_r005_parent_identity_not_allowed():
    t = testing.build_trajectory(stage="pre_update")
    d = decision_codes(t, stage="full_pre_update")
    assert "R005_PARENT_IDENTITY_NOT_ALLOWED" in d.reason_codes


def test_r001_reward_protocol_missing():
    t = testing.build_trajectory(stage="pre_update")
    parent = testing.validation_event(t.run_id, ValidationDecision(decision="allow", validation_stage="identity_pre_reward", reason_codes=[]), t.next_seq())
    t.events[parent.event_id] = parent
    from grpo_guard.schema.artifacts import EventRef as ER

    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-r1",
        "parent_identity_decision": ER(uri="", event_id=parent.event_id, event_sha256=parent.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()

    reward = t.events[t.envelope.reward_event.event_id]
    no_proto = reward.model_copy(deep=True).model_copy(
        update={"event_id": f"{reward.event_id}-noproto", "evaluator_protocol_sha256": ""}
    )
    no_proto.event_sha256 = ""
    no_proto = no_proto.seal()
    t.events[no_proto.event_id] = no_proto
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-r1b",
        "reward_event": EventRef(uri="", event_id=no_proto.event_id, event_sha256=no_proto.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    d = decision_codes(t, stage="full_pre_update")
    assert "R001_REWARD_PROTOCOL_MISSING" in d.reason_codes


def test_r002_infra_error_as_task_fail():
    t = testing.build_trajectory(stage="pre_update")
    reward = t.events[t.envelope.reward_event.event_id]
    infra = reward.model_copy(deep=True).model_copy(
        update={"event_id": f"{reward.event_id}-infra", "terminal_status": "infra_error"}
    )
    infra.event_sha256 = ""
    infra = infra.seal()
    t.events[infra.event_id] = infra
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-r2",
        "reward_event": EventRef(uri="", event_id=infra.event_id, event_sha256=infra.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    d = decision_codes(t, stage="full_pre_update")
    assert "R002_INFRA_ERROR_AS_TASK_FAIL" in d.reason_codes
