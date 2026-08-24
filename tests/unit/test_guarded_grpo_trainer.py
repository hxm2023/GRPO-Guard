"""P1-1 GuardedGRPOTrainer: official-TRL-path guard seams (CPU tests)."""

from __future__ import annotations

import numpy as np
import pytest

from grpo_guard.adapters.guarded_grpo_trainer import (
    GuardViolation,
    GuardedGRPOTrainer,
    _align_checks,
    _logprob_length_check,
)


class _MockTrainer:
    """Minimal stand-in for TRL's GRPOTrainer (no GPU needed)."""

    def __init__(self, *args, **kwargs):
        self.state = type("S", (), {"global_step": 0})()

    def _generate_and_score_completions(self, inputs):
        return inputs

    def training_step(self, model, inputs, num_items_in_batch):
        return "step-ok"

    def _save_checkpoint(self, model, trial):
        return "ckpt-ok"


class _GuardedMock(GuardedGRPOTrainer, _MockTrainer):
    pass


def test_align_checks_pass_on_valid_rollout():
    pids = list(range(10))
    cids = list(range(10, 20))
    assert _align_checks(pids, cids) == []


def test_align_checks_empty_completion():
    assert _align_checks([1, 2, 3], []) == ["M005_EMPTY_COMPLETION"]


def test_logprob_length_mismatch():
    assert _logprob_length_check([1, 2, 3], [0.1, 0.2]) == "L004_TOKEN_LOGPROB_LENGTH_MISMATCH"
    assert _logprob_length_check([1, 2, 3], [0.1, 0.2, 0.3]) is None
    assert _logprob_length_check([1, 2, 3], None) is None


def test_guarded_rollout_fails_closed_on_violation(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    bad = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[]], "logprobs": [[]]}
    with pytest.raises(GuardViolation) as exc:
        t._generate_and_score_completions(bad)
    assert "M005_EMPTY_COMPLETION" in str(exc.value)


def test_guarded_rollout_records_real_artifacts(tmp_path):
    """P0-4: mask/logprob/reward artifacts are REAL bytes, not placeholders."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]], "rewards": [[0.5]]}
    out = t._generate_and_score_completions(good)
    assert out == good
    rec = t._guard_rollouts[0]
    assert rec["completion_len"] == 3
    for ref in (rec["sequence_ref"], rec["mask_ref"], rec["logprob_ref"], rec["reward_ref"]):
        assert ref.sha256 != "0" * 64
        assert t._guard_store.verify(ref)  # bytes actually stored + hashable
    events = list(t._guard_log.iterate())
    assert len(events) == 1
    assert events[0]["event_type"] == "generation_finished"
    assert events[0]["service_behavior_logprobs"]["sha256"] == rec["logprob_ref"].sha256


def _rollout_inputs(good):
    pids, cids = good["prompt_ids"][0], good["completion_ids"][0]
    seq = np.asarray(pids + cids, dtype=np.int32)
    lp_row = np.zeros(len(pids) + len(cids), dtype=np.float32)
    lp_row[len(pids):] = good["logprobs"][0]
    return {"input_ids": seq.reshape(1, -1), "logprobs": lp_row.reshape(1, -1)}


def test_guarded_training_step_verifies_actual_inputs_and_rotates(tmp_path):
    """P0-4: the ACTUAL consumed tensors must match the recorded artifacts;
    records rotate after the step (no stale validation)."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]]}
    t._generate_and_score_completions(good)
    assert t.training_step(None, _rollout_inputs(good), 1) == "step-ok"
    assert t._last_guard_verified == {"global_step": 0, "n_rollouts": 1}
    assert t._guard_rollouts == []  # rotated


def test_guarded_training_step_refuses_retokenized_tokens(tmp_path):
    """P0-4: tampered input_ids (retokenization wiring bug) fail BEFORE
    super().training_step — i.e. before loss/backward."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]]}
    t._generate_and_score_completions(good)
    bad = _rollout_inputs(good)
    bad["input_ids"][0][3] = 999  # completion tokens re-encoded
    with pytest.raises(GuardViolation, match="T001"):
        t.training_step(None, bad, 1)


def test_guarded_training_step_refuses_misbound_logprobs(tmp_path):
    """P0-4: tampered old logprobs (misbinding wiring bug) fail before step."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]]}
    t._generate_and_score_completions(good)
    bad = _rollout_inputs(good)
    bad["logprobs"][0][4] = -9.9  # different old-logprob values
    with pytest.raises(GuardViolation, match="L004"):
        t.training_step(None, bad, 1)


def test_guarded_training_step_refuses_row_count_mismatch(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]]}
    t._generate_and_score_completions(good)
    bad = {"input_ids": np.zeros((2, 6), dtype=np.int32)}  # 2 rows, 1 recorded
    with pytest.raises(GuardViolation, match="batch rows"):
        t.training_step(None, bad, 1)


def test_guarded_training_step_refuses_missing_input_ids(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]]}
    t._generate_and_score_completions(good)
    with pytest.raises(GuardViolation, match="no input_ids"):
        t.training_step(None, {}, 1)


def test_guarded_training_step_refuses_reward_misbinding(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = {"prompt_ids": [[1, 2, 3]], "completion_ids": [[10, 11, 12]],
            "logprobs": [[0.1, 0.2, 0.3]]}
    t._generate_and_score_completions(good)
    bad = _rollout_inputs(good)
    bad["advantages"] = np.zeros((3, 1), dtype=np.float32)  # 3 rows != 1 recorded
    with pytest.raises(GuardViolation, match="advantages rows"):
        t.training_step(None, bad, 1)


def test_guarded_commit_records_sha(tmp_path):
    class FakeParam:
        def __init__(self, arr):
            self.data = type("D", (), {"detach": lambda self: type("C", (), {
                "cpu": lambda self: type("C", (), {"numpy": lambda self: arr})()})()})()

    class FakeModel:
        def named_parameters(self):
            yield "w0", FakeParam(np.zeros((2, 2), dtype=np.float32))
            yield "b0", FakeParam(np.ones(2, dtype=np.float32))

    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    assert t._save_checkpoint(FakeModel(), None) == "ckpt-ok"
    assert len(t._last_guard_commit_sha256) == 64
