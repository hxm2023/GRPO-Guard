"""Validation context: everything a rule may read (design doc §8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from grpo_guard.schema.envelope import TrajectoryEnvelope
from grpo_guard.schema.events import EventBase
from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
from grpo_guard.store.artifact_store import ArtifactStore


class ProtocolConfig(BaseModel):
    name: str
    mode: Literal["strict_on_policy", "bounded_off_policy"]
    max_policy_lag_versions: int = 0
    importance_correction: str | None = None
    checked_ruleset_sha256: str = ""


@dataclass
class ValidationContext:
    envelope: TrajectoryEnvelope
    store: ArtifactStore
    events: dict[str, EventBase]
    policy_manifest: PolicyManifest | None
    split_manifest: SplitManifest | None
    protocol: ProtocolConfig
    canary_status: str | None = None  # "pass" | "mismatch" | None (no canary evidence)
    update_input_event: EventBase | None = None
    split_registry: dict[str, SplitManifest] | None = None  # v0.2 F5: all splits by name
    eval_protocol_sha256: str | None = None  # v0.2 F6: the declared eval protocol
    notes: list[str] = field(default_factory=list)

    def event(self, event_id: str) -> EventBase | None:
        return self.events.get(event_id)
