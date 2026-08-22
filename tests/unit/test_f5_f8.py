"""v0.2-preview F5-F8 fault families (design doc §11)."""

from grpo_guard import testing
from grpo_guard.faults.f5_f8 import (
    inject_f5_split_leakage,
    inject_f6_evaluator_alias,
    inject_f7_event_reorder,
    inject_f8_artifact_mutation,
)
from grpo_guard.schema.manifests import SplitManifest
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

STRICT = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def _decide(t, **kw):
    kw.setdefault("split_registry", getattr(t, "split_registry", None))
    kw.setdefault("eval_protocol_sha256", getattr(t, "eval_protocol_sha256", None))
    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=STRICT, **kw,
    )
    return validate_envelope(ctx, "full_pre_update").decision_payload


def test_f5_split_leakage_rejected():
    t = testing.build_trajectory()
    f = inject_f5_split_leakage(t, other_split="held_out")
    d = _decide(f)
    assert d.decision == "reject"
    assert "D003_SPLIT_OVERLAP" in d.reason_codes


def test_f5_normal_no_overlap_allowed():
    t = testing.build_trajectory()
    other = SplitManifest(split_id="split-held_out", split_name="held_out",
                          prompt_ids=["countdown-other"])
    t.split_registry = {t.split_manifest.split_name: t.split_manifest, "held_out": other}
    d = _decide(t)
    assert d.decision == "allow"


def _pre_update_with_parent(**kw):
    """pre_update trajectory with an allowed parent identity (passes R005)."""
    from grpo_guard.schema.decisions import ValidationDecision

    t = testing.build_trajectory(stage="pre_update", **kw)
    parent = testing.validation_event(
        t.run_id,
        ValidationDecision(decision="allow", validation_stage="identity_pre_reward", reason_codes=["G001"]),
        t.next_seq(),
    )
    t.events[parent.event_id] = parent
    from grpo_guard.schema.artifacts import EventRef

    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-parented",
        "parent_identity_decision": EventRef(uri="", event_id=parent.event_id, event_sha256=parent.event_sha256),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    return t


def test_f6_evaluator_alias_quarantined():
    t = _pre_update_with_parent()
    f = inject_f6_evaluator_alias(t, eval_protocol_sha256="eval-proto-abc")
    d = _decide(f)
    assert "R006_EVALUATOR_ALIAS" in d.reason_codes


def test_f6_no_alias_allowed():
    t = _pre_update_with_parent()
    f = inject_f6_evaluator_alias(t, eval_protocol_sha256="eval-proto-abc")
    f.eval_protocol_sha256 = None  # no declared eval protocol → rule not applicable
    d = _decide(f)
    assert d.decision == "allow"


def test_f7_event_reorder_rejected():
    t = testing.build_trajectory()
    f = inject_f7_event_reorder(t)
    from grpo_guard.schema.events import UpdateInputEvent

    gen = f.events[f.envelope.generation_event.event_id]
    upd = UpdateInputEvent(
        event_id="uinput-f7", run_id=f.run_id, component_id="materializer",
        lifecycle_seq=gen.lifecycle_seq + 100, created_at_utc=testing.now_utc(),
        update_id="update-1", preupdate_envelope=f.envelope.ref(),
        preupdate_validation_decision=__import__("grpo_guard.schema.artifacts", fromlist=["EventRef"]).EventRef(
            uri="", event_id="vdec-x", event_sha256="0" * 64),
        sequence_token_ids=gen.sequence_token_ids, loss_mask=gen.loss_mask,
        authoritative_behavior_logprob_event=f.envelope.training_contract.authoritative_behavior_logprob_event,
        authoritative_behavior_logprobs=gen.service_behavior_logprobs,
        reward_event=__import__("grpo_guard.schema.artifacts", fromlist=["EventRef"]).EventRef(
            uri="", event_id="reward-x", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
        tokenizer_called=False,
    ).seal()
    d = _decide(f, update_input_event=upd)
    assert d.decision == "reject"
    assert "L005_SCORING_AFTER_UPDATE" in d.reason_codes


def test_f8_artifact_mutation_rejected():
    with testing.ArtifactStoreTmp() as store:
        t = testing.build_trajectory(store)
        f = inject_f8_artifact_mutation(t)
        d = _decide(f)
        assert d.decision == "reject"
        assert "T001_ARTIFACT_HASH_MISMATCH" in d.reason_codes


def test_f8_does_not_touch_other_artifacts():
    with testing.ArtifactStoreTmp() as store:
        t = testing.build_trajectory(store)
        gen = t.events[t.envelope.generation_event.event_id]
        mask_sha = gen.loss_mask.sha256
        inject_f8_artifact_mutation(t)
        # the loss mask blob is untouched
        assert store.verify(gen.loss_mask)
        assert mask_sha == gen.loss_mask.sha256
