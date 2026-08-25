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


def _good_result(n=1, prompt=None, completion=None, lp=None, pad=False):
    """A TRL 1.10-shaped generation result (REAL key names)."""
    prompt = prompt if prompt is not None else [1, 2, 3]
    completion = completion if completion is not None else [10, 11, 12]
    lp = lp if lp is not None else [0.1, 0.2, 0.3]
    P, C = len(prompt), len(completion)
    if pad:
        P_, C_ = 5, 5  # fixed padded widths (left-pad prompt, right-pad completion)
        pp = [0] * (P_ - P) + prompt
        cc = completion + [0] * (C_ - C)
        pm = [0] * (P_ - P) + [1] * P
        cm = [1] * C + [0] * (C_ - C)
        full_lp = [0.0] * len(pp) + lp + [0.0] * (C_ - C)
    else:
        pp, cc, pm, cm = prompt, completion, [1] * P, [1] * C
        full_lp = [0.0] * P + lp
    return {
        "prompt_ids": [list(pp)] * n, "prompt_mask": [list(pm)] * n,
        "completion_ids": [list(cc)] * n, "completion_mask": [list(cm)] * n,
        "sampling_per_token_logps": [list(lp)] * n,
        "old_per_token_logps": [list(full_lp)] * n,
        "advantages": [[0.5]] * n,
    }


def test_guarded_rollout_records_real_artifacts(tmp_path):
    """P0-4: mask/logprob/reward artifacts are REAL bytes, not placeholders."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    out = t._generate_and_score_completions(good)
    assert out == good
    rec = t._guard_rollouts[0]
    assert rec["completion_len"] == 3
    for ref in (rec["sequence_ref"], rec["mask_ref"], rec["old_logprob_ref"],
                rec["sampling_logprob_ref"], rec["advantage_ref"]):
        assert ref is not None and ref.sha256 != "0" * 64
        assert t._guard_store.verify(ref)  # bytes actually stored + hashable
    events = list(t._guard_log.iterate())
    assert len(events) == 1
    assert events[0]["event_type"] == "generation_finished"
    assert events[0]["service_behavior_logprobs"]["sha256"] == rec["sampling_logprob_ref"].sha256


def test_guarded_training_step_verifies_actual_inputs_and_rotates(tmp_path):
    """P0-4: the ACTUAL consumed tensors must match the recorded artifacts;
    records rotate after the step (no stale validation)."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    assert t.training_step(None, good, 1) == "step-ok"
    assert t._last_guard_verified == {"global_step": 0, "n_rollouts": 1}
    assert t._guard_rollouts == []  # rotated


def test_guarded_step_content_matching_survives_shuffle_and_padding(tmp_path):
    """TRL 1.10 shuffles rows and pads per-part sequences — the guard must
    match by CONTENT (mask-selected real tokens), not by position."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result(n=3, pad=True)
    # different prompt/completion lengths per row -> distinct hashes
    good["prompt_ids"][1] = [0, 0, 4, 5, 6]
    good["prompt_mask"][1] = [0, 0, 1, 1, 1]
    good["old_per_token_logps"][1] = [0.0] * 5 + [0.1, 0.2] + [0.0] * 3
    good["completion_ids"][2] = [20, 21, 22, 23, 0]
    good["completion_mask"][2] = [1, 1, 1, 1, 0]
    good["sampling_per_token_logps"][2] = [0.1, 0.2, 0.3, 0.4]
    good["old_per_token_logps"][2] = [0.0] * 5 + [0.1, 0.2, 0.3, 0.4] + [0.0]
    good["old_per_token_logps"][0] = [0.0] * 5 + [0.1, 0.2, 0.3] + [0.0] * 2
    t._generate_and_score_completions(good)
    shuffled = {"prompt_ids": [good["prompt_ids"][2], good["prompt_ids"][0], good["prompt_ids"][1]],
                "prompt_mask": [good["prompt_mask"][2], good["prompt_mask"][0], good["prompt_mask"][1]],
                "completion_ids": [good["completion_ids"][2], good["completion_ids"][0], good["completion_ids"][1]],
                "completion_mask": [good["completion_mask"][2], good["completion_mask"][0], good["completion_mask"][1]],
                "old_per_token_logps": [good["old_per_token_logps"][2], good["old_per_token_logps"][0], good["old_per_token_logps"][1]],
                "advantages": [[0.5]] * 3}
    assert t.training_step(None, shuffled, 3) == "step-ok"


def test_guarded_step_accepts_list_of_sample_dicts(tmp_path):
    """transformers 5.x passes training_step a LIST of per-sample dicts;
    the guard must normalize and still verify (P0-4)."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    sample = {k: v[0] for k, v in good.items()}
    assert t.training_step(None, [sample], 1) == "step-ok"
    assert t._last_guard_verified == {"global_step": 0, "n_rollouts": 1}


def test_guarded_step_rejects_tampered_token_in_list_of_dicts(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    sample = {k: v[0] for k, v in good.items()}
    sample["completion_ids"] = [10, 11, 999]  # re-encoded token
    with pytest.raises(GuardViolation, match="T001"):
        t.training_step(None, [sample], 1)


def test_guarded_training_step_refuses_retokenized_tokens(tmp_path):
    """P0-4: tampered tokens (retokenization wiring bug) fail BEFORE
    super().training_step — i.e. before loss/backward."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    bad = dict(good)
    bad["completion_ids"] = [[10, 11, 999]]  # completion re-encoded
    with pytest.raises(GuardViolation, match="T001"):
        t.training_step(None, bad, 1)


def test_guarded_training_step_refuses_misbound_logprobs(tmp_path):
    """P0-4: tampered old logprobs (misbinding wiring bug) fail before step."""
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    bad = dict(good)
    bad["old_per_token_logps"] = [[0.0, 0.0, 0.0, 0.1, -9.9, 0.3]]  # one value swapped
    with pytest.raises(GuardViolation, match="L004"):
        t.training_step(None, bad, 1)


def test_guarded_training_step_refuses_row_count_mismatch(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    bad = dict(good)
    bad["prompt_ids"] = [[1, 2, 3], [1, 2, 3]]  # 2 rows, 1 recorded
    with pytest.raises(GuardViolation, match="batch rows"):
        t.training_step(None, bad, 1)


def test_guarded_training_step_refuses_missing_tokens(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    with pytest.raises(GuardViolation, match="no prompt_ids"):
        t.training_step(None, {"advantages": [[0.5]]}, 1)


def test_guarded_training_step_refuses_reward_misbinding(tmp_path):
    t = _GuardedMock(guard_events_dir=tmp_path / "events", guard_store_dir=tmp_path / "store")
    good = _good_result()
    t._generate_and_score_completions(good)
    bad = dict(good)
    bad["advantages"] = [[-9.9]]  # different reward values
    with pytest.raises(GuardViolation, match="advantages"):
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
