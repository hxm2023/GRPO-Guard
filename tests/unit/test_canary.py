"""Canary suite contract (design doc §10)."""

from grpo_guard.canary import CanarySuite, _token_diff


def _fake_gen(prompt, n=1, temperature=0.0, top_p=1.0, top_k=1, max_tokens=8):
    # deterministic completion ids per prompt (sketch behavior)
    base = sum(ord(c) for c in prompt[0]) % 100
    return ([list(range(10))], [[(base + i) % 50 + 1 for i in range(max_tokens)]], [[[0.5] * max_tokens]], None)


def test_sketch_accepts_dict_from_trl_client():
    # TRL's VLLMClient.generate returns a dict; tuple-unpacking a dict
    # yields its KEYS — the constant-sketch bug.  The suite must read the
    # dict keys explicitly.
    suite = CanarySuite(prompts=["x"])

    def dict_gen(prompt, **kw):
        pid, cid, lps, lti = _fake_gen(prompt, **kw)
        return {"prompt_ids": pid, "completion_ids": cid, "logprobs": lps,
                "logprob_token_ids": lti}

    s1 = suite.sketch(dict_gen)
    s2 = suite.sketch(dict_gen)
    assert s1 == s2
    assert isinstance(s1[0], list) and all(isinstance(t, int) for t in s1[0])
    # and a drift in the dict-based sketch IS detected
    drifted = {"prompt_ids": _fake_gen(["x"])[0],
               "completion_ids": [[t + 9 for t in _fake_gen(["x"])[1][0]]],
               "logprobs": [], "logprob_token_ids": None}
    assert suite.check(dict_gen, policy_version=1, baseline=s1, tolerance=0).verdict == "pass"
    r = suite.check(lambda p, **kw: drifted, policy_version=1, baseline=s1, tolerance=0)
    assert r.verdict == "mismatch"


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
