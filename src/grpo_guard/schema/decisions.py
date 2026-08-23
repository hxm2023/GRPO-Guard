"""Validation decisions and reason codes (design doc §7.8, §8)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["allow", "quarantine", "reject"]

ValidationStage = Literal["identity_pre_reward", "full_pre_update"]

# Identity stage only checks policy/lifecycle/token/mask/logprob lineage plus
# the reward-null invariant; the data/reward rule families run at pre-update.
IDENTITY_STAGE_RULES = {
    "P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009",
    "T001", "T002", "T003", "T004", "T005",
    "M001", "M002", "M003", "M004", "M005",
    "L001", "L002", "L003", "L004", "L005", "L006", "L007", "L008",
    "R004",
}
PRE_UPDATE_STAGE_RULES = IDENTITY_STAGE_RULES | {
    "D001", "D002", "R001", "R002", "R003", "R005",
    # v0.2-preview fault families (F5 split overlap, F6 evaluator alias)
    "D003", "R006",
    # v0.2.1 families (F9 reward injection, F10 data poisoning — D16)
    "D004", "R008",
}


class ValidationDecision(BaseModel):
    decision: Decision
    validation_stage: ValidationStage
    reason_codes: list[str] = Field(default_factory=list)
    checked_ruleset_sha256: str = ""
    checked_event_sha256s: list[str] = Field(default_factory=list)
    checked_artifact_sha256s: list[str] = Field(default_factory=list)
    observed_policy_lag: int = Field(default=0, ge=0)
    validator_version: str = "1.0"
