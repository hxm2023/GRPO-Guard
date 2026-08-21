"""F2 — misbound old-logprob (reconstructed_from_incident, design doc §2.3/§11).

Incident: the trainer computed "old" log-probs with its CURRENT local model
against re-encoded text, while the trajectory came from a different (static)
service — the value was labeled behavior log-prob but was not produced by
the behavior policy.

Injection: bind a scoring event produced by a DIFFERENT policy version
(``scorer_policy_version != generation.behavior_policy_version``) to the
generation, and declare it authoritative → L003 fires.
"""

from __future__ import annotations

from grpo_guard import testing
from grpo_guard.schema.artifacts import EventRef
from grpo_guard.schema.envelope import TrajectoryEnvelope
from grpo_guard.schema.events import ScoringEvent


def inject_f2_misbound_logprob(
    t: testing.Trajectory,
    scorer_policy_version: int,
) -> testing.Trajectory:
    """Attach a scorer of policy ``scorer_policy_version`` (≠ behavior) to the
    same generation event, declared authoritative in exact_behavior_scorer mode.
    """
    gen = t.events[t.envelope.generation_event.event_id]
    seq_ref = gen.sequence_token_ids
    lp_ref = gen.service_behavior_logprobs
    assert lp_ref is not None, "F2 needs a generation with service logprobs to re-source"

    gen_ref = EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256)
    scoring = ScoringEvent(
        event_id=f"score-{gen.event_id}-f2",
        event_type="behavior_scoring_finished",
        run_id=t.run_id,
        component_id="behavior_scorer",
        lifecycle_seq=gen.lifecycle_seq + 1,
        created_at_utc=testing.now_utc(),
        input_events=[gen_ref],
        output_artifacts=[lp_ref],
        source_generation_event=gen_ref,
        scorer_policy_version=scorer_policy_version,
        scorer_checkpoint_manifest_sha256=testing.make_policy_manifest(scorer_policy_version).checkpoint_manifest_sha256,
        token_artifact_sha256=seq_ref.sha256,
        scoring_dtype="bf16",
        behavior_logprobs=lp_ref,
    ).seal()

    scoring_ref = EventRef(uri=f"event://{scoring.event_id}", event_id=scoring.event_id, event_sha256=scoring.event_sha256)

    env = TrajectoryEnvelope(
        envelope_id=f"{t.envelope.envelope_id}-f2",
        envelope_stage=t.envelope.envelope_stage,
        run_id=t.run_id,
        request_id=t.envelope.request_id,
        generation_event=t.envelope.generation_event,
        scoring_event=scoring_ref,
        reward_event=t.envelope.reward_event,
        policy_manifest=t.envelope.policy_manifest,
        split_manifest=t.envelope.split_manifest,
        parent_envelope_sha256=t.envelope.parent_envelope_sha256,
        parent_identity_decision=t.envelope.parent_identity_decision,
        training_contract=t.envelope.training_contract.model_copy(
            update={"behavior_logprob_source": "exact_behavior_scorer", "authoritative_behavior_logprob_event": scoring_ref}
        ),
    ).seal()

    events = dict(t.events)
    events[scoring.event_id] = scoring

    return testing.Trajectory(
        run_id=t.run_id,
        events=events,
        policy_manifest=t.policy_manifest,
        split_manifest=t.split_manifest,
        envelope=env,
        store=t.store,
        sequence=t.sequence,
        target_mask=t.target_mask,
        loss_mask=t.loss_mask,
        logprobs=t.logprobs,
        completion_text=t.completion_text,
        goal=t.goal,
        target_numbers=t.target_numbers,
        reward_components=t.reward_components,
        sync_events=t.sync_events,
        sequence_ref=t.sequence_ref,
        bogus_sequence_ref=t.bogus_sequence_ref,
    )
