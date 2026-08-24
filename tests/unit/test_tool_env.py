"""P1-3 tool-use trajectory contract tests."""

from __future__ import annotations

from grpo_guard.adapters.tool_env import STATE_VERSION, ToolCall, ToolEnv, ToolResult, TrajectoryTurn, validate_tool_trajectory


def _turn(index, action, calls=None, results=None):
    return TrajectoryTurn(index=index, action_tokens=action, tool_calls=calls, results=results)


def test_valid_trajectory_passes():
    env = ToolEnv()
    call = ToolCall(call_id="c1", name="add", args={"a": 1, "b": 2})
    res = env.execute(call)
    c2 = ToolCall(call_id="c2", name="add", args={"a": 10, "b": 20})
    r2 = ToolResult(call_id="c2", state_version=STATE_VERSION, observation="30")
    turns = [
        _turn(0, [1, 2, 3], calls=[call], results=[res]),
        _turn(1, [4, 5], calls=[c2], results=[r2]),
    ]
    assert validate_tool_trajectory(turns) == []


def test_stale_observation_detected():
    env = ToolEnv()
    call = ToolCall(call_id="c1", name="add", args={"a": 1, "b": 2})
    stale = ToolResult(call_id="c1", state_version="OLD", observation="3")
    turns = [_turn(0, [1], calls=[call], results=[stale])]
    violations = validate_tool_trajectory(turns)
    assert any("STALE_OBSERVATION" in v for v in violations)


def test_orphan_result_detected():
    res = ToolResult(call_id="nobody", state_version=STATE_VERSION, observation="x")
    turns = [_turn(0, [1], results=[res])]
    violations = validate_tool_trajectory(turns)
    assert any("ORPHAN_TOOL_RESULT" in v for v in violations)


def test_duplicate_result_detected():
    env = ToolEnv()
    call = ToolCall(call_id="c1", name="add", args={"a": 1, "b": 2})
    res = env.execute(call)
    turns = [_turn(0, [1], calls=[call], results=[res, res])]
    violations = validate_tool_trajectory(turns)
    assert any("DUPLICATE_TOOL_RESULT" in v for v in violations)


def test_duplicate_call_detected():
    call = ToolCall(call_id="c1", name="add", args={"a": 1, "b": 2})
    turns = [_turn(0, [1], calls=[call, call])]
    violations = validate_tool_trajectory(turns)
    assert any("DUPLICATE_TOOL_CALL" in v for v in violations)


def test_empty_action_detected():
    turns = [_turn(0, [])]
    violations = validate_tool_trajectory(turns)
    assert any("M005_EMPTY_ACTION" in v for v in violations)


def test_env_execution_deterministic():
    env = ToolEnv()
    r1 = env.execute(ToolCall(call_id="a", name="mul", args={"a": 6, "b": 7}))
    r2 = env.execute(ToolCall(call_id="a", name="mul", args={"a": 6, "b": 7}))
    assert r1.observation == "42" and r2.observation == "42"
