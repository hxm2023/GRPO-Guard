"""F4 — mask shift (reconstructed_from_incident, design doc §2.5/§11).

Incident: the completion mask was selected as ``masks[i, T-comp_len[i]:]``
(last positions) which under padding/truncation picks prompt or padding
tokens; action masks used ``str.find`` substring guesses.

Injection: shift the completion target mask (and the shifted loss mask) by
``shift`` tokens without touching the token sequence or logprobs → the
canonical reconstruction disagrees (M004), and a rightward shift leaks the
last prompt token into the completion mask (M002).
"""

from __future__ import annotations

import numpy as np

from grpo_guard import testing
from grpo_guard.schema.envelope import TrajectoryEnvelope
from grpo_guard.schema.events import GenerationEvent


def inject_f4_mask_shift(
    t: testing.Trajectory,
    shift: int = 1,
) -> testing.Trajectory:
    """Move the completion mask by ``shift`` tokens (1..3 canonical range).

    positive shift → mask covers [P-shift, T-shift) (prompt leak at left);
    negative shift → mask covers [P-shift, T-shift) (truncation at right).
    """
    gen = t.events[t.envelope.generation_event.event_id]
    assert isinstance(gen, GenerationEvent)

    T = t.sequence.shape[0]
    P = gen.completion_span[0]

    shifted_target = np.zeros(T, dtype=np.int8)
    shifted_target[max(P - shift, 0): T - shift] = 1
    for s, e in gen.padding_spans:
        shifted_target[s:e] = 0
    shifted_loss = shifted_target[1:].copy()

    new_target_ref = t.store.put(
        shifted_target.tobytes(), "application/octet-stream", f"{gen.event_id}-f4", dtype="int8", shape=[T]
    )
    new_loss_ref = t.store.put(
        shifted_loss.tobytes(), "application/octet-stream", f"{gen.event_id}-f4", dtype="int8", shape=[T - 1]
    )

    new_gen = gen.model_copy(deep=True).model_copy(
        update={
            "event_id": f"{gen.event_id}-f4",
            "completion_target_mask": new_target_ref,
            "loss_mask": new_loss_ref,
            "output_artifacts": [gen.sequence_token_ids, new_target_ref, new_loss_ref]
            + ([gen.service_behavior_logprobs] if gen.service_behavior_logprobs else []),
        }
    )
    new_gen.event_sha256 = ""
    new_gen = new_gen.seal()
    events = dict(t.events)
    events[new_gen.event_id] = new_gen

    from grpo_guard.schema.artifacts import EventRef

    env = t.envelope.model_copy(deep=True).model_copy(
        update={
            "envelope_id": f"{t.envelope.envelope_id}-f4",
            "generation_event": EventRef(
                uri=f"event://{new_gen.event_id}", event_id=new_gen.event_id, event_sha256=new_gen.event_sha256
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
        target_mask=shifted_target,
        loss_mask=shifted_loss,
        logprobs=t.logprobs,
        completion_text=t.completion_text,
        goal=t.goal,
        target_numbers=t.target_numbers,
        reward_components=t.reward_components,
        sync_events=t.sync_events,
        sequence_ref=t.sequence_ref,
        bogus_sequence_ref=t.bogus_sequence_ref,
    )
