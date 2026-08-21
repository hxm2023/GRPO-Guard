"""Canonical alignment contract (design doc §7.9)."""

import pytest

from grpo_guard.validators.align import AlignmentError, mask_from_artifact_bytes, reconstruct_alignment


def test_basic_alignment():
    a = reconstruct_alignment(T=12, completion_start=4, completion_end=12)
    assert a.C == 8
    assert a.completion_target_mask.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert a.loss_mask.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert a.prediction_positions == list(range(3, 11))
    assert a.completion_target_mask.sum() == 8
    assert a.loss_mask.sum() == 8


def test_masks_do_not_overlap_prompt_or_padding():
    a = reconstruct_alignment(T=10, completion_start=4, completion_end=10, padding_spans=[[8, 10]])
    assert a.completion_target_mask.sum() == 4
    assert a.completion_target_mask[:4].sum() == 0
    assert a.completion_target_mask[8:].sum() == 0
    assert a.loss_mask.sum() == 4
    assert a.padding_within_completion is True


def test_left_padding_excluded():
    a = reconstruct_alignment(T=10, completion_start=4, completion_end=10, padding_spans=[[0, 2]])
    assert a.completion_target_mask[:2].sum() == 0


def test_empty_completion_raises():
    with pytest.raises(AlignmentError):
        reconstruct_alignment(T=10, completion_start=5, completion_end=5)


def test_span_out_of_range_raises():
    with pytest.raises(AlignmentError):
        reconstruct_alignment(T=10, completion_start=9, completion_end=11)


def test_mask_bytes_validation():
    a = reconstruct_alignment(T=6, completion_start=2, completion_end=6)
    good = a.completion_target_mask.tobytes()
    assert mask_from_artifact_bytes(good, 6).sum() == 4
    with pytest.raises(AlignmentError):
        mask_from_artifact_bytes(good, 5)  # wrong length
    bad = bytes([0, 2, 1, 1, 1, 1])
    with pytest.raises(AlignmentError):
        mask_from_artifact_bytes(bad, 6)  # non-binary
