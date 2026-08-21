"""Canonical token/mask/logprob alignment (design doc §7.9).

Fixed layout for the authoritative sequence of length T, prompt/completion
boundary P, completion length C = T - P::

    sequence_token_ids:       shape [T]
    completion_target_mask:   shape [T], 1 at [P, T)
    next_token_logits:         conceptual shape [T-1, V]
    loss_mask:                 shape [T-1], 1 at [P-1, T-1)
    behavior_logprobs:         shape [C], aligned to targets [P, T)
    prediction_positions:     [P-1, P, ..., T-2]

The validator always rebuilds these masks from spans and compares elementwise
with producer artifacts; it never trusts the producer's mask bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class AlignmentError(ValueError):
    """Invalid span/length combination — explicit protocol branch required."""


class EmptyCompletionError(AlignmentError):
    """C == 0 — handled by M005 (quarantine), not by shape rules."""


@dataclass
class Alignment:
    T: int
    P: int
    C: int
    completion_target_mask: np.ndarray  # int8 [T]
    loss_mask: np.ndarray  # int8 [T-1]
    prediction_positions: list[int] = field(default_factory=list)
    padding_within_completion: bool = False


def reconstruct_alignment(
    T: int,
    completion_start: int,
    completion_end: int,
    padding_spans: list[list[int]] | None = None,
    prompt_start: int = 0,
) -> Alignment:
    """Rebuild the canonical masks from spans.

    Raises AlignmentError on illegal spans (caller maps to T005/M001).
    """
    padding_spans = padding_spans or []
    if not (0 <= prompt_start <= completion_start <= completion_end <= T):
        raise AlignmentError(f"illegal spans: T={T} prompt=[{prompt_start},{completion_start}) completion=[{completion_start},{completion_end})")
    if completion_end - completion_start <= 0:
        raise EmptyCompletionError("empty completion")

    target = np.zeros(T, dtype=np.int8)
    target[completion_start:completion_end] = 1
    padding_within = False
    for s, e in padding_spans:
        if not (0 <= s < e <= T):
            raise AlignmentError(f"illegal padding span [{s},{e})")
        if s < completion_end and e > completion_start:
            padding_within = True
        target[s:e] = 0

    if T - 1 <= 0:
        raise AlignmentError("sequence too short to carry a loss mask")

    # loss_mask[i] = target_mask[i+1]: the target at position j is predicted
    # by logits at position j-1, so padding exclusion shifts along with it.
    loss = np.zeros(T - 1, dtype=np.int8)
    loss[:] = target[1:]

    positions = list(range(completion_start - 1, completion_end - 1))
    return Alignment(
        T=T,
        P=completion_start,
        C=completion_end - completion_start,
        completion_target_mask=target,
        loss_mask=loss,
        prediction_positions=positions,
        padding_within_completion=padding_within,
    )


def mask_from_artifact_bytes(data: bytes, expected_len: int) -> np.ndarray:
    """Decode a producer mask artifact (int8 numpy bytes) and validate length."""
    arr = np.frombuffer(data, dtype=np.int8)
    if arr.size != expected_len:
        raise AlignmentError(f"mask bytes length {arr.size} != expected {expected_len}")
    if not set(arr.tolist()).issubset({0, 1}):
        raise AlignmentError("mask contains non-binary values")
    return arr
