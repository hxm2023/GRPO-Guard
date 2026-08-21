"""Paired-replay metrics (design doc §12.4).

When either gradient norm is ≈ 0 the cosine is unstable: report
``undefined_near_zero`` with the norms instead of fabricating a 0.
"""

from __future__ import annotations

import math

import numpy as np


def gradient_cosine(g_control: np.ndarray, g_fault: np.ndarray, eps: float = 1e-12) -> float | str:
    n_c = float(np.linalg.norm(g_control))
    n_f = float(np.linalg.norm(g_fault))
    if n_c < eps or n_f < eps:
        return "undefined_near_zero"
    return float(np.dot(g_control.ravel(), g_fault.ravel()) / (n_c * n_f))


def relative_l2(g_control: np.ndarray, g_fault: np.ndarray, eps: float = 1e-12) -> float:
    n_c = float(np.linalg.norm(g_control))
    return float(np.linalg.norm(g_fault - g_control) / (n_c + eps))


def update_norm(delta_theta: np.ndarray) -> float:
    return float(np.linalg.norm(delta_theta))


def ratio_stats(ratios: np.ndarray) -> dict:
    if ratios.size == 0:
        return {"p50": None, "p95": None, "max": None, "count": 0}
    return {
        "p50": float(np.percentile(ratios, 50)),
        "p95": float(np.percentile(ratios, 95)),
        "max": float(np.max(ratios)),
        "count": int(ratios.size),
    }


def clip_fraction(ratios: np.ndarray, clip_range: tuple[float, float] = (0.2, 0.2)) -> float:
    if ratios.size == 0:
        return 0.0
    lo, hi = 1.0 - clip_range[0], 1.0 + clip_range[1]
    return float(np.mean((ratios < lo) | (ratios > hi)))


def selected_tokens(mask: np.ndarray, spans: list[tuple[int, int]]) -> int:
    """Count mask-selected tokens that fall inside any given span."""
    selected = mask.astype(bool)
    total = 0
    for s, e in spans:
        total += int(selected[s:e].sum())
    return total
