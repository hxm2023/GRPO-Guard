"""Canary suite contract (design doc §10)."""

from grpo_guard.canary import CanarySuite, _token_diff


def _fake_gen(prompt, n=1, temperature=0.0, top_p=1.0, top_k=1, max_tokens=8):
    # deterministic completion ids per prompt (sketch behavior)
    base = sum(ord(c) for c in prompt[0]) % 100
    return ([list(range(10))], [[(base + i) % 50 + 1 for i in range(max_tokens)]], [[[0.5] * max_tokens]], None)


def test_sketch_deterministic():
    suite = CanarySuite(prompts=["a", "b"])
    s1 = suite.sketch(_fake_gen)
    s2 = suite.sketch(_fake_gen)
    assert s1 == s2
    assert len(s1) == 2


def test_calibration_freezes_tolerance():
    suite = CanarySuite(prompts=["x"])
    cal = suite.calibrate(_fake_gen, reloads=5)
    assert cal["reloads"] == 5
    assert cal["observed_max_token_drift"] == 0
    assert cal["frozen_tolerance"] >= 0


def test_check_pass_and_mismatch():
    suite = CanarySuite(prompts=["x"])

    def stable_gen(prompt, **kw):
        return _fake_gen(prompt, **kw)

    r = suite.check(stable_gen, policy_version=1, baseline=_fake_gen(["x"])[1], tolerance=0)
    assert r.verdict == "pass"

    def drifted_gen(prompt, **kw):
        pid, cid, lps, _ = _fake_gen(prompt, **kw)
        return pid, [[t + 7 for t in cid[0]]], lps, None

    r2 = suite.check(drifted_gen, policy_version=1, baseline=_fake_gen(["x"])[1], tolerance=0)
    assert r2.verdict == "mismatch"
    assert r2.drift["max_token_drift"] > 0


def test_token_diff():
    assert _token_diff([1, 2, 3], [1, 2, 3]) == 0
    assert _token_diff([1, 2, 3], [1, 9, 3]) == 1
    assert _token_diff([1, 2], [1, 2, 3]) == 1
