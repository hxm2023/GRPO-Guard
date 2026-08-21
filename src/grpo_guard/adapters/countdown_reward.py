"""Deterministic Countdown rule verifier (design doc §4.1, §7.6).

The verifier checks whether a completion is a valid arithmetic expression
using exactly the target numbers to reach the goal.  It is deterministic and
versioned: any change to the rule set must bump ``reward_version`` and the
protocol hash, or rewards lose their lineage identity.
"""

from __future__ import annotations

import ast
import hashlib
import operator
import re

from grpo_guard.store.canonical_json import canonical_dumps

REWARD_VERSION = "countdown-rule-v1"

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_EXPR_RE = re.compile(r"^[0-9+\-*/()\s]+$")


def countdown_rule_verifier(completion: str, target_numbers: list[int], goal: int) -> dict:
    """Return reward components for one completion.

    correctness: 1.0 iff the expression uses exactly the target numbers and
    evaluates to the goal; format: 1.0 iff the completion is a single
    well-formed expression (no prose).  Never maps infra errors here — the
    caller decides how to represent them (Rule R002).
    """
    components = {"correctness": 0.0, "format": 0.0}
    expr = completion.strip()
    if not expr or not _EXPR_RE.match(expr):
        return components
    components["format"] = 1.0
    try:
        used = _extract_numbers(expr)
        if sorted(used) != sorted(target_numbers):
            return components
        value = _safe_eval(expr)
        if abs(value - float(goal)) < 1e-6:
            components["correctness"] = 1.0
    except (ValueError, ZeroDivisionError, SyntaxError):
        pass
    return components


def _extract_numbers(expr: str) -> list[int]:
    return [int(tok) for tok in re.findall(r"\d+", expr)]


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return _ALLOWED_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = eval_node(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        raise ValueError(f"unsupported node {type(node).__name__}")

    return eval_node(tree)


def reward_protocol_sha256() -> str:
    """Identity of the verifier rule set; bound into every RewardEvent."""
    return hashlib.sha256(
        canonical_dumps(
            {
                "reward_version": REWARD_VERSION,
                "allowed_ops": ["add", "sub", "mul", "div"],
                "exact_number_use": True,
                "format_rule": "single-expression",
                "float_tolerance": 1e-6,
            }
        )
    ).hexdigest()


class CountdownRewardAdapter:
    """Produce RewardEvents with explicit protocol identity (design doc §7.6)."""

    def __init__(self, protocol_sha256: str | None = None):
        self.protocol_sha256 = protocol_sha256 or reward_protocol_sha256()

    def score(self, completion: str, target_numbers: list[int], goal: int) -> dict:
        return countdown_rule_verifier(completion, target_numbers, goal)
