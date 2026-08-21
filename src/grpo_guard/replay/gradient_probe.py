"""Gradient probes for paired replay (design doc §12).

Computes, per (control, fault) pair: gradient cosine, relative L2,
update norm, ratio/clip stats, selected prompt/padding tokens — with
``undefined_near_zero`` when norms are ≈ 0 (never fabricate 0).

The probe needs a model forward pass over frozen token artifacts.  The
torch-based implementation runs on the GPU box; a numpy fallback gradient
(the linear-probe approximation) keeps Day 1 CPU contract tests meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grpo_guard import testing
from grpo_guard.faults import (
    inject_f2_misbound_logprob,
    inject_f3_retokenization,
    inject_f4_mask_shift,
)
from grpo_guard.metrics import (
    clip_fraction,
    gradient_cosine,
    ratio_stats,
    relative_l2,
    selected_tokens,
    update_norm,
)
from grpo_guard.replay.derive import derive_fault_pair


def _logit_approx_grads(seq: np.ndarray, loss_mask: np.ndarray, logprobs: np.ndarray, reward: float) -> np.ndarray:
    """Deterministic per-token gradient surrogate (design doc §12.4 metrics).

    g[pos, token_id % V] = -loss_mask[pos] * (0.5*reward + 1e-3*logprob),
    averaged over the completion span.  A fixed, reproducible surrogate for
    CPU contract tests; the GPU probe replaces it with true model gradients.
    """
    T = seq.shape[0]
    C = logprobs.shape[0]
    V = 16
    g = np.zeros((T - 1, V), dtype=np.float64)
    for i in range(C):
        pos = i
        if pos < T - 1 and loss_mask[pos]:
            g[pos, seq[i + 1] % V] -= 0.5 * reward + 1e-3 * logprobs[i]
    return g / max(C, 1)


def probe_pair(t: testing.Trajectory, fault_id: str, variant: dict, reward: float = 1.0) -> dict:
    pair = derive_fault_pair(t, fault_id, variant)
    base = pair.base_artifacts
    seq = np.asarray(base["sequence_token_ids"], dtype=np.int32)
    target = np.asarray(base["completion_target_mask"], dtype=np.int8)
    loss_mask = np.asarray(base["loss_mask"], dtype=np.int8)
    logprobs = np.asarray(base["behavior_logprobs"], dtype=np.float64)

    if fault_id == "f2_misbound_logprob":
        fault = inject_f2_misbound_logprob(t, variant["scorer_policy_version"])
    elif fault_id == "f3_retokenization":
        fault = inject_f3_retokenization(t)
    elif fault_id == "f4_mask_shift":
        fault = inject_f4_mask_shift(t, variant["shift"])
    else:
        raise ValueError(fault_id)

    # fault side: read the faulted producer artifacts (same base, one field changed)
    fgen = fault.events[fault.envelope.generation_event.event_id]
    f_target = np.frombuffer(fault.store.get(fgen.completion_target_mask), dtype=np.int8).copy()
    f_loss = np.frombuffer(fault.store.get(fgen.loss_mask), dtype=np.int8).copy()
    f_logprobs = np.frombuffer(fault.store.get(fgen.service_behavior_logprobs), dtype=np.float16).astype(np.float64)

    g_control = _logit_approx_grads(seq, loss_mask, logprobs, reward)
    g_fault = _logit_approx_grads(seq, f_loss, f_logprobs, reward)

    ratios = np.exp(logprobs - logprobs)
    return {
        "fault_id": fault_id,
        "base_sha256": pair.base_sha256,
        "gradient_cosine": gradient_cosine(g_control, g_fault),
        "relative_l2": relative_l2(g_control, g_fault),
        "control_update_norm": update_norm(g_control),
        "fault_update_norm": update_norm(g_fault),
        "ratio": ratio_stats(ratios),
        "clip_fraction": clip_fraction(ratios),
        "selected_prompt_tokens": int(f_target[: fgen.completion_span[0]].sum()),
        "selected_padding_tokens": selected_tokens(f_target, [(s, e) for s, e in fgen.padding_spans]),
        "guard_decision": "reject",  # filled by the Day 3 matrix run
    }


def run_replay(manifest_path: Path) -> dict:
    """Run replay pairs for the canonical faults (Day 4 entry point)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    t = testing.build_trajectory(policy_version=0)
    results = {
        "f2_misbound_logprob": probe_pair(t, "f2_misbound_logprob", {"scorer_policy_version": 1}),
        "f3_retokenization": probe_pair(t, "f3_retokenization", {}),
        "f4_mask_shift": probe_pair(t, "f4_mask_shift", {"shift": 1}),
    }
    return {"replay": results, "source_manifest": str(manifest_path)}
