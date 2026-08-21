"""Countdown rule verifier contract (design doc §7.6)."""

from grpo_guard.adapters.countdown_reward import (
    countdown_rule_verifier,
    reward_protocol_sha256,
)


def test_correct_expression():
    r = countdown_rule_verifier("(1+2)*3", [1, 2, 3], 9)
    assert r["correctness"] == 1.0
    assert r["format"] == 1.0


def test_wrong_value():
    r = countdown_rule_verifier("(1+2)*3", [1, 2, 3], 10)
    assert r["correctness"] == 0.0


def test_wrong_numbers_used():
    r = countdown_rule_verifier("1+2+2", [1, 2, 3], 5)
    assert r["correctness"] == 0.0  # used 2 twice, never used 3


def test_prose_gets_zero_format():
    r = countdown_rule_verifier("let me think... 3*3", [3, 3], 9)
    assert r["format"] == 0.0


def test_division_by_zero_no_crash():
    r = countdown_rule_verifier("1/(3-3)", [1, 3, 3], 1)
    assert r["correctness"] == 0.0


def test_protocol_hash_stable():
    assert len(reward_protocol_sha256()) == 64
    assert reward_protocol_sha256() == reward_protocol_sha256()
