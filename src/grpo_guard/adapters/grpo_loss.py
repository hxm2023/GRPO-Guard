"""Guarded GRPO loss over validated batch handles (design doc §7.3.3, §9.1).

The optimizer consumes ONLY the materialized handle tensors: the exact
token sequences the server sampled, the canonical loss masks, the
authoritative behavior logprobs, and the reward tensors.  No text, no
re-tokenization, no trainer-side recomputation of "old" logprobs.

Handles are stacked into one batch (right-padded to T_max); the canonical
loss_mask per row carries the completion positions [P-1, T-1), so padding
contributes zero.  Rewards are grouped per prompt (group_size generations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from grpo_guard.adapters.guarded_update import MaterializedBatch, ValidatedBatchHandle


@dataclass
class GuardedLossResult:
    loss: torch.Tensor
    metrics: dict = field(default_factory=dict)


def _as_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.int32:
        return torch.as_tensor(arr, dtype=torch.int64)
    return torch.as_tensor(arr, dtype=dtype)


def _stack_batches(batches: Sequence[MaterializedBatch]):
    """Stack already-materialized batches into one padded batch (no consume)."""
    T_max = max(b.sequence_token_ids.shape[0] for b in batches)
    B = len(batches)
    seq = np.zeros((B, T_max), dtype=np.int32)
    mask = np.zeros((B, T_max - 1), dtype=np.int8)
    lp_rows = []
    rewards = np.zeros(B, dtype=np.float32)
    for i, b in enumerate(batches):
        T_i = b.sequence_token_ids.shape[0]
        seq[i, :T_i] = b.sequence_token_ids[:T_i]
        mask[i, :T_i - 1] = b.loss_mask[:T_i - 1]
        lp_rows.append(b.behavior_logprobs)
        rewards[i] = float(b.rewards[0])
    C_max = max(len(r) for r in lp_rows)
    logprobs = np.zeros((B, C_max), dtype=np.float32)
    for i, r in enumerate(lp_rows):
        logprobs[i, :len(r)] = r
    return seq, mask, logprobs, rewards


def _loss_from_batches(
    model: torch.nn.Module,
    batches: Sequence[MaterializedBatch],
    group_size: int,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> GuardedLossResult:
    """GRPO loss over ALREADY-materialized batches (consumed handles).

    Internal shared path for ``grpo_loss`` and ``guarded_optimizer_step``:
    the optimizer may only reach the loss through validated, consumed
    handles — never through raw tensors taken out by the caller.
    """
    if not batches:
        raise ValueError("no materialized batches")
    seq_np, mask_np, lp_np, rewards_np = _stack_batches(batches)
    device = next(model.parameters()).device
    seq = _as_tensor(seq_np).to(device)
    loss_mask = _as_tensor(mask_np).to(device)
    old_logps = _as_tensor(lp_np).to(device)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32).to(device)

    B, T = seq.shape
    V = model.config.vocab_size
    mask = loss_mask.bool()  # [B, T-1]

    out = model(input_ids=seq)[0]  # [B, T, V]
    logits = out[:, :-1, :].float()  # [B, T-1, V]: position j predicts token j+1
    targets = seq[:, 1:]

    new_logps = -F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none"
    ).reshape(B, T - 1)

    # scatter authoritative old logprobs onto their masked positions (row order);
    # each row's real values are the first sum(mask[row]) entries (rest is pad)
    counts = mask.sum(dim=1)
    flat_real = torch.cat([old_logps[b, : counts[b]] for b in range(B)])
    old_logps_padded = torch.zeros_like(new_logps)
    old_logps_padded[mask] = flat_real

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
        "group_size": int(group_size),
    }
    return GuardedLossResult(loss=loss, metrics=metrics)


def grpo_loss(
    model: torch.nn.Module,
    handles: ValidatedBatchHandle | Sequence[ValidatedBatchHandle],
    group_size: int,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> GuardedLossResult:
    """GRPO loss on the handle tensors; only masked positions contribute."""
    if isinstance(handles, ValidatedBatchHandle):
        handles = [handles]
    if not isinstance(handles, (list, tuple)) or not all(isinstance(h, ValidatedBatchHandle) for h in handles):
        raise TypeError("grpo_loss accepts only ValidatedBatchHandle(s), no text fallback")

    batches = [h.consume() for h in handles]
    return _loss_from_batches(model, batches, group_size, clip_epsilon=clip_epsilon, beta=beta)
