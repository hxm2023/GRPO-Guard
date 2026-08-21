"""Guarded GRPO loss over the validated batch handle (design doc §7.3.3, §9.1).

The optimizer consumes ONLY the materialized handle tensors: the exact
token sequence the server sampled, the canonical loss mask, the
authoritative behavior logprobs, and the reward tensors.  No text, no
re-tokenization, no trainer-side recomputation of "old" logprobs.

Batch layout (materializer-normalized): sequence [B, T_max] right-padded,
loss_mask [B, T_max-1] with ones exactly at completion prediction
positions [P-1, T-1) per row (zeros elsewhere), old_logprobs [B, C_max]
row-aligned to completion targets.  Padding positions contribute zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from grpo_guard.adapters.guarded_update import ValidatedBatchHandle


@dataclass
class GuardedLossResult:
    loss: torch.Tensor
    metrics: dict = field(default_factory=dict)


def _as_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.int32:
        return torch.as_tensor(arr, dtype=torch.int64)
    return torch.as_tensor(arr, dtype=dtype)


def grpo_loss(
    model: torch.nn.Module,
    handle: ValidatedBatchHandle,
    group_size: int,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> GuardedLossResult:
    """GRPO loss on the handle's tensors; only masked positions contribute."""
    if not isinstance(handle, ValidatedBatchHandle):
        raise TypeError("grpo_loss accepts only ValidatedBatchHandle (no text fallback)")
    batch = handle.consume()
    seq = _as_tensor(batch.sequence_token_ids)
    loss_mask = _as_tensor(batch.loss_mask)
    old_logps = _as_tensor(batch.behavior_logprobs)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)

    B, T = seq.shape
    V = model.config.vocab_size
    mask = loss_mask.bool()  # [B, T-1]

    out = model(input_ids=seq)[0]  # [B, T, V]
    logits = out[:, :-1, :].float()  # [B, T-1, V]: position j predicts token j+1
    targets = seq[:, 1:]

    new_logps = -F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none"
    ).reshape(B, T - 1)

    # scatter authoritative old logprobs onto their masked positions (row order)
    old_logps_padded = torch.zeros_like(new_logps)
    old_logps_padded[mask] = old_logps.reshape(-1)

    ratio = torch.exp(new_logps - old_logps_padded)  # 1.0 on non-masked slots
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    rewards_g = rewards.view(-1, group_size)
    mean = rewards_g.mean(dim=1, keepdim=True)
    # population std (unbiased=False): single-element groups give 0, not NaN
    std = rewards_g.var(dim=1, unbiased=False, keepdim=True).sqrt() + 1e-6
    advantage = ((rewards_g - mean) / std).view(-1)

    per_token = -torch.min(ratio, clipped) * advantage.unsqueeze(1)
    per_token = per_token * mask.float()
    loss = per_token.sum() / (mask.float().sum() + 1e-9)

    masked_ratio = ratio[mask]
    metrics = {
        "ratio_p50": float(torch.quantile(masked_ratio, 0.5).item()),
        "ratio_p95": float(torch.quantile(masked_ratio, 0.95).item()),
        "ratio_max": float(masked_ratio.max().item()),
        "clip_fraction": float(
            ((masked_ratio < 1.0 - clip_epsilon) | (masked_ratio > 1.0 + clip_epsilon)).float().mean().item()
        ),
        "loss": float(loss.item()),
        "B": int(B),
        "T": int(T),
    }
    return GuardedLossResult(loss=loss, metrics=metrics)
