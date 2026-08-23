"""F9-F10 canonical fault injectors — v0.2.1 (design doc §11 status note, D16).

Task-agnostic, deterministic, reconstructed_from_incident:

- F9  reward injection: the reward event claims a verifier that is not in
      the registered evaluator registry (or a registered name with a
      wrong protocol hash) → R008_REWARD_VERIFIER_UNREGISTERED.
- F10 data poisoning: the prompt's token content is substituted under the
      same id → D004_PROMPT_CONTENT_MISMATCH (vs frozen content_sha256s).
"""

from __future__ import annotations

from grpo_guard import testing


def inject_f9_reward_hacking(
    t: testing.Trajectory,
    fake_version: str = "fake-reward-v1",
    wrong_protocol: str | None = None,
) -> testing.Trajectory:
    """F9 — reward injection: replace the reward's verifier identity.

    Variants: canonical (unregistered verifier name), held-out (registered
    name but wrong protocol sha), boundary (empty registry -> rule inert,
    envelope-level check still allows).
    """
    registry = {
        "countdown-rule-v1": "0" * 64,
        "gsm8k-rule-v1": "1" * 64,
    }
    t.reward_verifier_registry = registry
    if t.envelope.reward_event is not None:
        reward = t.events[t.envelope.reward_event.event_id]
        reward.reward_version = fake_version
        if wrong_protocol:
            reward.evaluator_protocol_sha256 = wrong_protocol
        else:
            reward.evaluator_protocol_sha256 = "f" * 64
    return t


def inject_f10_data_poisoning(
    t: testing.Trajectory,
    which: str = "prompt_tokens",
) -> testing.Trajectory:
    """F10 — data poisoning: the prompt's TOKEN CONTENT is substituted while
    keeping the same prompt id and span (content_sha256s mismatch → D004).

    The frozen manifest registers the ORIGINAL prompt content hash, then
    the prompt span tokens are replaced under the same id.
    """
    import hashlib

    import numpy as np

    gen = t.events[t.envelope.generation_event.event_id]
    seq = t.sequence.copy()
    start, end = gen.prompt_span
    t.split_manifest.content_sha256s[gen.prompt_id] = hashlib.sha256(seq[start:end].tobytes()).hexdigest()
    seq[start:end] = (seq[start:end] + 1) % 32000  # shift every prompt token
    ref = t.store.put(seq.tobytes(), "application/octet-stream",
                      f"gen-{gen.event_id}-poisoned", dtype="int32", shape=[seq.shape[0]])
    gen.sequence_token_ids = ref
    t.sequence = seq
    t.sequence_ref = ref
    return t
