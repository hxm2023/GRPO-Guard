"""GSM8K integration: reward adapter -> sealed RewardEvent -> protocol identity.

Shows the framework pipeline is task-agnostic: the same event schema and
protocol binding used by Countdown works with a second deterministic rule
set without touching the validator/store layers.
"""

from __future__ import annotations

import time

from grpo_guard.adapters.gsm8k_reward import Gsm8kRewardAdapter, reward_protocol_sha256
from grpo_guard.schema.artifacts import EventRef
from grpo_guard.schema.events import RewardEvent


def test_gsm8k_reward_event_protocol_binding():
    adapter = Gsm8kRewardAdapter()
    comps = adapter.score("60 * 3 = 180 miles.", 180)
    assert comps == {"correctness": 1.0, "format": 1.0}

    gen = EventRef(uri="", event_id="gen-gsm8k-1", event_sha256="a" * 64)
    rew = RewardEvent(
        event_id="reward-gsm8k-1", event_type="reward_finished",
        run_id="run-gsm8k", component_id="gsm8k_reward",
        lifecycle_seq=1, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        input_events=[gen],
        reward_version="gsm8k-rule-v1",
        evaluator_protocol_sha256=reward_protocol_sha256(),
        source_generation_event=gen,
        components=comps, terminal_status="success", latency_ms=0.0,
    ).seal()

    assert rew.reward_version == "gsm8k-rule-v1"
    assert rew.evaluator_protocol_sha256 == reward_protocol_sha256()
    assert rew.event_sha256  # sealed event is content-addressed like any other
    assert rew.components["correctness"] == 1.0


def test_adapter_protocol_distinct_per_task():
    from grpo_guard.adapters.countdown_reward import CountdownRewardAdapter

    assert (Gsm8kRewardAdapter().protocol_sha256 != CountdownRewardAdapter().protocol_sha256)
