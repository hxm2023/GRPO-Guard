"""F3 — retokenization (reconstructed_from_incident, design doc §2.4/§11).

Incident: the rollout sampled tokens with the server chat template while the
trainer re-encoded ``prompt + completion`` as a plain string with a different
tokenizer/template — the sequence the trainer optimized was not the sequence
the policy sampled.

Injection variants:
- canonical: generation claims a tokenizer hash that differs from the policy
  manifest's (T002_TOKENIZER_MISMATCH);
- template variant: chat template hash differs (T003);
- held-out: the pre-update materializer consumes a DIFFERENT token artifact
  than the producer's (T004_TOKEN_SEQUENCE_MISMATCH).
"""

from __future__ import annotations

from grpo_guard import testing
from grpo_guard.schema.envelope import TrajectoryEnvelope
from grpo_guard.schema.events import GenerationEvent


def inject_f3_retokenization(
    t: testing.Trajectory,
    tokenizer_sha: str | None = None,
    template_sha: str | None = None,
) -> testing.Trajectory:
    """Rewrite the generation event's tokenizer/template identity hashes."""
    gen = t.events[t.envelope.generation_event.event_id]
    assert isinstance(gen, GenerationEvent)

    new_gen = gen.model_copy(deep=True).model_copy(
        update={
            "event_id": f"{gen.event_id}-f3",
            "tokenizer_sha256": tokenizer_sha or ("bad-" + testing.DEFAULT_TOKENIZER_SHA[:60]),
            "chat_template_sha256": template_sha or gen.chat_template_sha256,
        }
    )
    new_gen.event_sha256 = ""
    new_gen = new_gen.seal()
    events = dict(t.events)
    events[new_gen.event_id] = new_gen

    from grpo_guard.schema.artifacts import EventRef

    new_ref = EventRef(uri=f"event://{new_gen.event_id}", event_id=new_gen.event_id, event_sha256=new_gen.event_sha256)
    env = t.envelope.model_copy(deep=True).model_copy(
        update={
            "envelope_id": f"{t.envelope.envelope_id}-f3",
            "generation_event": new_ref,
            "training_contract": t.envelope.training_contract.model_copy(
                update={"authoritative_behavior_logprob_event": new_ref}
            ),
        }
    )
    env.envelope_sha256 = ""
    env = env.seal()

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


def inject_f3_retokenized_sequence(
    t: testing.Trajectory,
    wrong_sequence_sha: str,
) -> testing.Trajectory:
    """Held-out variant: the materializer consumed a re-encoded sequence
    artifact whose content hash differs from the producer's (T004)."""
    from grpo_guard.schema.artifacts import ArtifactRef

    assert wrong_sequence_sha != t.sequence_ref.sha256, "must differ from producer sequence"

    bogus = ArtifactRef(
        uri=f"artifact://{wrong_sequence_sha}",
        media_type="application/octet-stream",
        dtype="int32",
        shape=list(t.sequence.shape),
        num_bytes=t.sequence.nbytes,
        sha256=wrong_sequence_sha,
        producer_event_id="gen-f3-bogus",
    )
    t.bogus_sequence_ref = bogus
    return t


def inject_f3_template_variant(t: testing.Trajectory) -> testing.Trajectory:
    return inject_f3_retokenization(t, template_sha="bad-" + testing.DEFAULT_TEMPLATE_SHA[:60])
