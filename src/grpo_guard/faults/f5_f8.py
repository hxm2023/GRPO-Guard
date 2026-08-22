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


def inject_f5_split_leakage(
    t: testing.Trajectory,
    other_split: str = "held_out",
    prompt_id: str | None = None,
    extra_split: str | None = None,
) -> testing.Trajectory:
    """F5 — split leakage: the trajectory's prompt is ALSO listed in another
    split manifest (e.g. a held-out prompt leaked into train).  The envelope
    still declares the train split; the registry carries both → D003.

    Variants: canonical (one other split), boundary (three splits with the
    same prompt), held-out (a different prompt leaked).
    """
    gen = t.events[t.envelope.generation_event.event_id]
    leaked = prompt_id or gen.prompt_id
    other = SplitManifest(
        split_id=f"split-{other_split}",
        split_name=other_split,
        prompt_ids=[leaked],  # the leak: same prompt in two splits
    )
    registry = {t.split_manifest.split_name: t.split_manifest, other_split: other}
    if extra_split:
        registry[extra_split] = SplitManifest(
            split_id=f"split-{extra_split}", split_name=extra_split, prompt_ids=[leaked],
        )
    t.split_registry = registry
    return t


def inject_f6_evaluator_alias(
    t: testing.Trajectory,
    eval_protocol_sha256: str,
    eval_reward_version: str | None = None,
) -> testing.Trajectory:
    """F6 — evaluator alias: the TRAIN reward event is produced under the
    DECLARED EVAL protocol (same judge/calibration serving both) → R006.

    Variants: canonical (protocol sha collision), held-out (protocol sha
    collision AND an eval-style reward_version marker).
    """
    t.eval_protocol_sha256 = eval_protocol_sha256
    if t.envelope.reward_event is not None:
        reward = t.events[t.envelope.reward_event.event_id]
        new_reward = reward.model_copy(deep=True).model_copy(update={
            "event_id": f"{reward.event_id}-f6",
            "evaluator_protocol_sha256": eval_protocol_sha256,
            **({"reward_version": eval_reward_version} if eval_reward_version else {}),
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


def inject_f6_no_alias(t: testing.Trajectory, eval_protocol_sha256: str) -> testing.Trajectory:
    """F6 boundary variant: the reward uses a DIFFERENT protocol than the
    declared eval protocol — NOT an alias → allow."""
    t.eval_protocol_sha256 = eval_protocol_sha256
    return t


def inject_f7_event_reorder(
    t: testing.Trajectory,
    scorer_policy_version: int | None = None,
) -> testing.Trajectory:
    """F7 — event reorder: a scoring event is placed AFTER the consuming
    update (lifecycle_seq > update input) → L005 fires (and P007 if the
    sync lands after the generation).

    Variants: canonical/boundary with an optional different scorer policy
    (boundary also fires L003), held-out via inject_f7_sync_reorder."""
    gen = t.events[t.envelope.generation_event.event_id]
    lp_ref = gen.service_behavior_logprobs
    assert lp_ref is not None

    gen_ref = EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)
    scorer_ver = scorer_policy_version if scorer_policy_version is not None else gen.behavior_policy_version
    late = ScoringEvent(
        event_id=f"score-{gen.event_id}-f7late",
        event_type="behavior_scoring_finished",
        run_id=t.run_id, component_id="behavior_scorer",
        lifecycle_seq=gen.lifecycle_seq + 9999,  # after the consuming update
        created_at_utc=testing.now_utc(),
        input_events=[gen_ref], output_artifacts=[lp_ref],
        source_generation_event=gen_ref,
        scorer_policy_version=scorer_ver,
        scorer_checkpoint_manifest_sha256=(
            testing.make_policy_manifest(scorer_ver).checkpoint_manifest_sha256
            if scorer_ver != gen.behavior_policy_version else gen.checkpoint_manifest_sha256
        ),
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


def inject_f8_artifact_mutation(
    t: testing.Trajectory,
    which: str = "sequence",
) -> testing.Trajectory:
    """F8 — artifact mutation: a sealed event's blob is overwritten AFTER
    sealing → T001 fires at validation (bytes no longer match).

    Variants: canonical (sequence blob), boundary (loss-mask blob),
    held-out (logprobs blob).
    """
    gen = t.events[t.envelope.generation_event.event_id]
    refs = {
        "sequence": gen.sequence_token_ids,
        "loss_mask": gen.loss_mask,
        "completion_target_mask": gen.completion_target_mask,
        "logprobs": gen.service_behavior_logprobs,
    }
    ref = refs[which]
    if ref is None:
        raise ValueError(f"no artifact for variant {which}")
    blob = t.store.blobs / ref.sha256
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(bytes(data))
    return t


def inject_f7_sync_reorder(t: testing.Trajectory) -> testing.Trajectory:
    """F7 held-out variant: the SYNC event is placed AFTER the generation
    (lifecycle_seq > generation) → P007 fires."""
    gen = t.events[t.envelope.generation_event.event_id]
    sync = t.events[gen.sync_event.event_id]
    late_sync = sync.model_copy(deep=True).model_copy(update={
        "event_id": f"{sync.event_id}-f7late",
        "lifecycle_seq": gen.lifecycle_seq + 5000,
    })
    late_sync.event_sha256 = ""
    late_sync = late_sync.seal()
    late_ref = EventRef(uri="", event_id=late_sync.event_id, event_sha256=late_sync.event_sha256)

    new_gen = gen.model_copy(deep=True).model_copy(update={
        "event_id": f"{gen.event_id}-f7sync",
        "sync_event": late_ref,
    })
    new_gen.event_sha256 = ""
    new_gen = new_gen.seal()
    new_gen_ref = EventRef(uri="", event_id=new_gen.event_id, event_sha256=new_gen.event_sha256)

    t.events[late_sync.event_id] = late_sync
    t.events[new_gen.event_id] = new_gen
    t.envelope = t.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t.envelope.envelope_id}-f7sync",
        "generation_event": new_gen_ref,
        "training_contract": t.envelope.training_contract.model_copy(
            update={"authoritative_behavior_logprob_event": new_gen_ref}
        ),
    })
    t.envelope.envelope_sha256 = ""
    t.envelope = t.envelope.seal()
    return t
