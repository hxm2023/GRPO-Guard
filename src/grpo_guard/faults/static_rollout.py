"""F1 — static rollout policy (reconstructed_from_incident, design doc §2.2/§11).

Incident: the trainer updated its local weights every iteration but the
rollout runtime never loaded them; generations kept coming from the old
policy.  Guard-relevant symptom: generation behavior policy version < the
update's parent policy version with no declared off-policy correction.

Injection: keep the generation bound to policy ``runtime_version`` while the
training contract claims parent policy ``claimed_parent`` (with claimed
parent > runtime version, no importance correction declared).
"""

from __future__ import annotations

from grpo_guard import testing
from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
from grpo_guard.schema.events import GenerationEvent


def inject_f1_static_rollout(
    t: testing.Trajectory,
    runtime_version: int,
    claimed_parent: int,
) -> testing.Trajectory:
    """Static rollout: runtime serves v=runtime_version, trainer consumes as
    parent v=claimed_parent in strict mode → P004 fires (or P005 in bounded).
    """
    gen = t.events[t.envelope.generation_event.event_id]
    assert isinstance(gen, GenerationEvent)

    contract = TrainingContract(
        protocol="strict_on_policy",
        trainer_parent_policy_version=claimed_parent,
        consuming_update_id=t.envelope.training_contract.consuming_update_id,
        max_policy_lag_versions=0,
        importance_correction=None,
        behavior_logprob_source=t.envelope.training_contract.behavior_logprob_source,
        authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
        diagnostic_non_authoritative_logprobs_allowed=False,
    )

    env = TrajectoryEnvelope(
        envelope_id=f"{t.envelope.envelope_id}-f1",
        envelope_stage=t.envelope.envelope_stage,
        run_id=t.run_id,
        request_id=t.envelope.request_id,
        generation_event=t.envelope.generation_event,
        scoring_event=t.envelope.scoring_event,
        reward_event=t.envelope.reward_event,
        policy_manifest=t.envelope.policy_manifest,
        split_manifest=t.envelope.split_manifest,
        parent_envelope_sha256=t.envelope.parent_envelope_sha256,
        parent_identity_decision=t.envelope.parent_identity_decision,
        training_contract=contract,
    ).seal()

    return testing.Trajectory(
        run_id=t.run_id,
        events=t.events,
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
