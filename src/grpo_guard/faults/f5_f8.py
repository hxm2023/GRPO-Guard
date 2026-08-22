"""F5-F8 canonical fault injectors — v0.2-preview (design doc §11).

These are NOT part of the v0.1 matrix (design doc §11: F7/F8 minimal
fixtures are already rejected by the general validator, but the full
injection protocol freezes only in v0.2).  Each injector mutates exactly
one target field of a valid trajectory, reconstructed_from_incident.
"""

from __future__ import annotations

from grpo_guard import testing
from grpo_guard.schema.artifacts import EventRef
from grpo_guard.schema.envelope import TrajectoryEnvelope
from grpo_guard.schema.events import GenerationEvent, RewardEvent, ScoringEvent
from grpo_guard.schema.manifests import SplitManifest


def inject_f5_split_leakage(t: testing.Trajectory, other_split: str = "held_out") -> testing.Trajectory:
    """F5 — split leakage: the trajectory's prompt is ALSO listed in another
    split manifest (e.g. a held-out prompt leaked into train).  The envelope
    still declares the train split; the registry carries both → D003.
    """
    gen = t.events[t.envelope.generation_event.event_id]
    other = SplitManifest(
        split_id=f"split-{other_split}",
        split_name=other_split,
        prompt_ids=[gen.prompt_id],  # the leak: same prompt in two splits
    )
    t.split_registry = {t.split_manifest.split_name: t.split_manifest, other_split: other}
    return t


def inject_f6_evaluator_alias(t: testing.Trajectory, eval_protocol_sha256: str) -> testing.Trajectory:
    """F6 — evaluator alias: the TRAIN reward event is produced under the
    DECLARED EVAL protocol (same judge/calibration serving both) → R006.
    """
    t.eval_protocol_sha256 = eval_protocol_sha256
    if t.envelope.reward_event is not None:
        reward = t.events[t.envelope.reward_event.event_id]
        new_reward = reward.model_copy(deep=True).model_copy(update={
            "event_id": f"{reward.event_id}-f6",
            "evaluator_protocol_sha256": eval_protocol_sha256,
        })
        new_reward.event_sha256 = ""
        new_reward = new_reward.seal()
        t.events[new_reward.event_id] = new_reward
        new_ref = EventRef(uri="", event_id=new_reward.event_id, event_sha256=new_reward.event_sha256)
        t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
            "envelope_id": f"{t.envelope.envelope_id}-f6",
            "reward_event": new_ref,
        })
        t.envelope.envelope_sha256 = ""
        t.envelope = t.envelope.seal()
    return t


def inject_f7_event_reorder(t: testing.Trajectory) -> testing.Trajectory:
    """F7 — event reorder: a scoring event is placed AFTER the consuming
    update (lifecycle_seq > update input) → L005 fires (and P007 if the
    sync lands after the generation)."""
    gen = t.events[t.envelope.generation_event.event_id]
    lp_ref = gen.service_behavior_logprobs
    assert lp_ref is not None

    gen_ref = EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)
    late = ScoringEvent(
        event_id=f"score-{gen.event_id}-f7late",
        event_type="behavior_scoring_finished",
        run_id=t.run_id, component_id="behavior_scorer",
        lifecycle_seq=gen.lifecycle_seq + 9999,  # after the consuming update
        created_at_utc=testing.now_utc(),
        input_events=[gen_ref], output_artifacts=[lp_ref],
        source_generation_event=gen_ref,
        scorer_policy_version=gen.behavior_policy_version,
        scorer_checkpoint_manifest_sha256=gen.checkpoint_manifest_sha256,
        token_artifact_sha256=gen.sequence_token_ids.sha256,
        scoring_dtype="bf16", behavior_logprobs=lp_ref,
    ).seal()
    late_ref = EventRef(uri="", event_id=late.event_id, event_sha256=late.event_sha256)

    t.events[late.event_id] = late
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-f7",
        "scoring_event": late_ref,
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    t.requires_update_input = True  # frozen cases must replay with an update input
    return t


def inject_f8_artifact_mutation(t: testing.Trajectory) -> testing.Trajectory:
    """F8 — artifact mutation: the sequence blob is overwritten AFTER the
    event was sealed → T001 fires at validation (bytes no longer match)."""
    gen = t.events[t.envelope.generation_event.event_id]
    blob = t.store.blobs / gen.sequence_token_ids.sha256
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(bytes(data))
    return t
