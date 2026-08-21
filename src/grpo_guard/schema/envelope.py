"""Trajectory envelope: reference-only container (design doc §7.7)."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from grpo_guard.schema.artifacts import EnvelopeRef, EventRef, ManifestRef
from grpo_guard.store.canonical_json import canonical_dumps

authoritative_logprob_sources = ("generation_service", "exact_behavior_scorer")


class TrainingContract(BaseModel):
    """Exactly one authoritative behavior-logprob source must be selected."""

    protocol: Literal["strict_on_policy", "bounded_off_policy"]
    trainer_parent_policy_version: int = Field(ge=0)
    consuming_update_id: str
    max_policy_lag_versions: int = Field(default=0, ge=0)
    importance_correction: str | None = None
    behavior_logprob_source: Literal["generation_service", "exact_behavior_scorer"]
    authoritative_behavior_logprob_event: EventRef
    diagnostic_non_authoritative_logprobs_allowed: bool = False


class TrajectoryEnvelope(BaseModel):
    """Reference-only envelope.  Never carries token/logprob/mask copies."""

    envelope_id: str
    envelope_sha256: str = ""
    envelope_stage: Literal["pre_reward", "pre_update"]
    run_id: str
    request_id: str
    generation_event: EventRef
    scoring_event: EventRef | None = None
    reward_event: EventRef | None = None
    policy_manifest: ManifestRef
    split_manifest: ManifestRef
    parent_envelope_sha256: str | None = None
    parent_identity_decision: EventRef | None = None
    training_contract: TrainingContract
    required_extensions: list[str] = Field(default_factory=list)

    def seal(self) -> "TrajectoryEnvelope":
        if self.envelope_sha256:
            raise ValueError(f"envelope {self.envelope_id} already sealed")
        payload = self.model_dump(mode="json", exclude={"envelope_sha256"})
        self.envelope_sha256 = hashlib.sha256(canonical_dumps(payload)).hexdigest()
        return self

    def verify_seal(self) -> bool:
        if not self.envelope_sha256:
            return False
        payload = self.model_dump(mode="json", exclude={"envelope_sha256"})
        return hashlib.sha256(canonical_dumps(payload)).hexdigest() == self.envelope_sha256

    def ref(self, uri: str = "") -> EnvelopeRef:
        if not self.envelope_sha256:
            raise ValueError("envelope must be sealed before creating a ref")
        return EnvelopeRef(uri=uri, envelope_id=self.envelope_id, envelope_sha256=self.envelope_sha256)
