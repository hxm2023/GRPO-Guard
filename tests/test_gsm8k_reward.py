"""GSM8K rule verifier: unit + frozen-sample tests (framework portability)."""

from __future__ import annotations

import pytest

from grpo_guard.adapters.gsm8k_reward import gsm8k_rule_verifier, reward_protocol_sha256

# Frozen GSM8K-style samples: (question, golden answer, completion, expected)
SAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the workers plant today?",
     6, "21 - 15 = 6. The workers planted 6 trees.", {"correctness": 1.0, "format": 1.0}),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in "
     "the parking lot?", 5, "3 + 2 = 5 cars.", {"correctness": 1.0, "format": 1.0}),
    ("Leah has 32 chocolates. Her sister has 42. If they eat 35, how many do they have left?",
     39, "32 + 42 - 35 = 39.", {"correctness": 1.0, "format": 1.0}),
    ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
     "muffins for her friends every day with four. She sells the remainder at the farmers' "
     "market daily for $2 per fresh duck egg. How much in dollars does she make every day at "
     "the farmers' market?", 18, "16 - 3 - 4 = 9 eggs, 9 * 2 = $18.", {"correctness": 1.0, "format": 1.0}),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in "
     "total does it take?", 3, "2 + 1 = 3 bolts.", {"correctness": 1.0, "format": 1.0}),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many "
     "toys does he have now?", 9, "5 + 2 + 2 = 9 toys.", {"correctness": 1.0, "format": 1.0}),
    ("There are 90 people in a ship. If the ship is sinking at a rate of 4 people per hour, "
     "how many people will be on the ship after 5 hours?", 70, "90 - 4*5 = 70.", {"correctness": 1.0, "format": 1.0}),
    ("A train travels at a speed of 60 miles per hour for 3 hours. How far does it travel?",
     180, "60 * 3 = 180 miles.", {"correctness": 1.0, "format": 1.0}),
]


def test_frozen_samples_all_correct():
    for question, golden, completion, expected in SAMPLES:
        comps = gsm8k_rule_verifier(completion, golden)
        assert comps == expected, f"{question[:40]}... -> {comps}"


def test_wrong_answer_extracts_last_number():
    comps = gsm8k_rule_verifier("5 + 3 = 8", 7)
    assert comps == {"correctness": 0.0, "format": 1.0}


def test_no_numeric_completion_unformatted():
    assert gsm8k_rule_verifier("I cannot solve this.", 5) == {"correctness": 0.0, "format": 0.0}
    assert gsm8k_rule_verifier("", 5) == {"correctness": 0.0, "format": 0.0}


def test_decimal_answer_within_tolerance():
    assert gsm8k_rule_verifier("half of 9 is 4.5", 4.5)["correctness"] == 1.0
    assert gsm8k_rule_verifier("half of 9 is 4.5000004", 4.5)["correctness"] == 1.0  # |Δ| < 1e-6
    assert gsm8k_rule_verifier("half of 9 is 4.50001", 4.5)["correctness"] == 0.0   # |Δ| > 1e-6
    assert gsm8k_rule_verifier("half of 9 is 4.6", 4.5)["correctness"] == 0.0


def test_protocol_hash_stable_and_distinct_from_countdown():
    from grpo_guard.adapters.countdown_reward import reward_protocol_sha256 as countdown_sha

    sha = reward_protocol_sha256()
    assert isinstance(sha, str) and len(sha) == 64
    # different task => different protocol identity
    assert sha != countdown_sha()
