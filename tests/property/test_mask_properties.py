"""Property-based mask invariants (design doc §15.2)."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from grpo_guard.validators.align import AlignmentError, reconstruct_alignment


@st.composite
def span_layout(draw):
    T = draw(st.integers(min_value=3, max_value=64))
    P = draw(st.integers(min_value=1, max_value=T - 1))
    end = draw(st.integers(min_value=P + 1, max_value=T))
    pads = draw(st.lists(
        st.tuples(st.integers(min_value=0, max_value=T), st.integers(min_value=0, max_value=T)),
        max_size=2,
    ))
    pads = [(s, e) for s, e in pads if s < e]
    return T, P, end, pads


@given(span_layout())
@settings(max_examples=150)
def test_mask_invariants(layout):
    T, P, end, pads = layout
    a = reconstruct_alignment(T, P, end, padding_spans=pads)
    # completion indices covered by any padding span (no double counting)
    covered = np.zeros(end - P, dtype=bool)
    for s, e in pads:
        lo, hi = max(s, P), min(e, end)
        if hi > lo:
            covered[lo - P: hi - P] = True
    assert a.completion_target_mask.sum() == (end - P) - int(covered.sum())
    # completion mask never touches prompt
    assert (a.completion_target_mask[:P] == 0).all()
    # completion mask never touches padding
    for s, e in pads:
        assert (a.completion_target_mask[s:e] == 0).all()
    # shifted loss mask covers exactly the same completion span
    assert a.loss_mask.sum() == a.completion_target_mask.sum()
    if not a.padding_within_completion:
        # prediction positions align with loss mask ones (padding inside the
        # completion needs the explicit protocol branch, design doc §7.9)
        ones = np.nonzero(a.loss_mask)[0].tolist()
        assert ones == a.prediction_positions[: len(ones)]
    assert len(a.prediction_positions) == a.C


@given(st.integers(min_value=1, max_value=64), st.integers(min_value=1, max_value=64))
@settings(max_examples=50)
def test_any_shift_breaks_canonical(T, shift):
    if T < 3:
        return
    P = T // 2
    a = reconstruct_alignment(T, P, T)
    shifted = np.zeros(T, dtype=np.int8)
    shifted[max(P - shift, 0): T - shift] = 1
    if not (shifted == a.completion_target_mask).all():
        return  # a shift of 0 is not a fault
    assert shift == 0


@given(span_layout())
@settings(max_examples=100)
def test_reconstruct_is_deterministic(layout):
    T, P, end, pads = layout
    a1 = reconstruct_alignment(T, P, end, padding_spans=pads)
    a2 = reconstruct_alignment(T, P, end, padding_spans=pads)
    assert (a1.completion_target_mask == a2.completion_target_mask).all()
    assert (a1.loss_mask == a2.loss_mask).all()
    assert a1.prediction_positions == a2.prediction_positions


@given(st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10))
@settings(max_examples=50)
def test_illegal_spans_raise(s, e):
    if s >= e:
        return
    with pytest.raises(AlignmentError):
        reconstruct_alignment(T=s, completion_start=s, completion_end=e)  # T too small
