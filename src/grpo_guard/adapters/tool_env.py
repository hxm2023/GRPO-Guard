"""Minimal tool-use environment + trajectory contract (P1-3).

A small deterministic tool-use loop: the agent emits
``thought/action/tool_call`` turns, the environment executes the tool
and returns an observation with an explicit STATE VERSION.  The contract
layer validates:

- **action-only loss mask**: only action tokens are trained on (tool
  calls and observations are not).
- **causal order**: a tool result must be generated AFTER its tool call
  (L005-style lifecycle).
- **stale observation**: an observation whose state version is older
  than the environment's current version is stale — consuming it as
  fresh signal is a contract violation.
- **duplicate/reordered results**: the same tool-call id resolved twice,
  or a result arriving out of order, is a contract violation.

This is the Agent-RL infra seam: it connects the guard to real
tool-use trajectories (design doc §2.2 legacy narrative excluded — this
is a fresh, deterministic fixture).
"""

from __future__ import annotations

from dataclasses import dataclass, field

STATE_VERSION = "tool-env-v1"


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    call_id: str
    state_version: str
    observation: str


@dataclass
class TrajectoryTurn:
    index: int
    action_tokens: list[int]          # tokens the agent emitted (trainable)
    tool_calls: list[ToolCall] | None = None
    results: list[ToolResult] | None = None


class ToolEnv:
    """Deterministic tool environment (e.g. a tiny calculator)."""

    def __init__(self, version: str = STATE_VERSION):
        self.version = version
        self._results: dict[str, str] = {}

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name == "add":
            obs = str(call.args.get("a", 0) + call.args.get("b", 0))
        elif call.name == "mul":
            obs = str(call.args.get("a", 0) * call.args.get("b", 0))
        else:
            obs = f"unknown tool {call.name}"
        self._results[call.call_id] = obs
        return ToolResult(call_id=call.call_id, state_version=self.version, observation=obs)


def validate_tool_trajectory(turns: list[TrajectoryTurn], env_version: str = STATE_VERSION) -> list[str]:
    """Contract checks over a tool-use trajectory.  Empty == pass.

    Returns violations (task-agnostic reason-coded strings).
    """
    violations = []
    seen_calls: set[str] = set()
    seen_versions: set[str] = set()
    for turn in turns:
        # action-only mask: every turn must carry trainable action tokens
        if not turn.action_tokens:
            violations.append(f"TURN{turn.index}: M005_EMPTY_ACTION")
        if turn.tool_calls:
            for call in turn.tool_calls:
                if call.call_id in seen_calls:
                    violations.append(f"TURN{turn.index}: DUPLICATE_TOOL_CALL {call.call_id}")
                seen_calls.add(call.call_id)
        if turn.results:
            for res in turn.results:
                if res.call_id not in seen_calls:
                    violations.append(f"TURN{turn.index}: ORPHAN_TOOL_RESULT {res.call_id}")
                if res.state_version != env_version:
                    violations.append(f"TURN{turn.index}: STALE_OBSERVATION {res.call_id} "
                                     f"(version {res.state_version} != {env_version})")
                if res.call_id in seen_versions:
                    violations.append(f"TURN{turn.index}: DUPLICATE_TOOL_RESULT {res.call_id}")
                seen_versions.add(res.call_id)
    return violations
