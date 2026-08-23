"""Event schema: EventBase plus all concrete event types (design doc §7.3-§7.8).

Sealing protocol: an event producer pre-allocates an immutable ``event_id``,
writes all artifact bytes and refs, then computes ``event_sha256`` over the
canonical JSON of the event with the ``event_sha256`` field excluded.  After
sealing, events are immutable; any revision creates a new event ID.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from grpo_guard.schema.artifacts import ArtifactRef, EnvelopeRef, EventRef
from grpo_guard.schema.decisions import ValidationDecision
from grpo_guard.store.canonical_json import canonical_dumps

SYNC_EVENT_TYPES = Literal[
    "sync_requested",
    "sync_started",
    "runtime_loaded",
    "canary_passed",
    "sync_unknown",
    "sync_reconciled_canary_passed",
    "sync_retryable_old",
    "sync_quarantined",
    "sync_failed",
    "sync_attempt_superseded",
]

UPDATE_EVENT_TYPES = Literal[
    "update_started",
    "update_prepared",
    "update_unknown",
    "update_committed",
    "update_restored_parent",
    "update_aborted",
    "update_attempt_superseded",
]

# Lifecycle states derivable by the deterministic reducer (§7.3.4)
SYNC_TERMINAL_SUCCESS = frozenset({"canary_passed", "sync_reconciled_canary_passed"})
SYNC_TERMINAL_FAILURE = frozenset({"sync_failed", "sync_quarantined", "sync_retryable_old", "sync_attempt_superseded"})
UPDATE_TERMINAL_SUCCESS = frozenset({"update_committed", "update_restored_parent"})
UPDATE_TERMINAL_FAILURE = frozenset({"update_aborted", "update_attempt_superseded"})


class EventBase(BaseModel):
    schema_version: str = "1.0"
    event_id: str
    event_type: str
    run_id: str
    component_id: str
    lifecycle_seq: int = Field(ge=0)
    created_at_utc: str
    input_events: list[EventRef] = Field(default_factory=list)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    event_sha256: str = ""

    def seal(self) -> "EventBase":
        """Compute the self hash over canonical JSON excluding ``event_sha256``."""
        if self.event_sha256:
            raise ValueError(f"event {self.event_id} already sealed")
        payload = self.model_dump(mode="json", exclude={"event_sha256"})
        self.event_sha256 = hashlib.sha256(canonical_dumps(payload)).hexdigest()
        return self

    def is_sealed(self) -> bool:
        return bool(self.event_sha256)

    def verify_seal(self) -> bool:
        if not self.event_sha256:
            return False
        payload = self.model_dump(mode="json", exclude={"event_sha256"})
        return hashlib.sha256(canonical_dumps(payload)).hexdigest() == self.event_sha256


class SyncEvent(EventBase):
    """One immutable phase of a weight-sync lifecycle (§7.3.1).

    Retries keep the same ``sync_id``/``idempotency_key``, bump ``attempt``,
    and write a superseding event with new event IDs.
    """

    event_type: SYNC_EVENT_TYPES
    sync_id: str
    attempt: int = Field(default=1, ge=1)
    supersedes_attempt: int | None = None
    lease_epoch: int = Field(ge=0)
    idempotency_key: str
    source_policy_version: int = Field(ge=0)
    source_checkpoint_manifest_sha256: str
    target_runtime_id: str
    previous_runtime_load_epoch: int | None = Field(default=None, ge=0)
    observed_runtime_load_epoch: int | None = Field(default=None, ge=0)
    observed_policy_version: int | None = None
    upstream_adapter_id: str
    upstream_operation: str
    compatibility_profile_sha256: str
    status_detail: str | None = None


class UpdateEvent(EventBase):
    """One immutable phase of the guarded optimizer transaction (§7.3.2)."""

    event_type: UPDATE_EVENT_TYPES
    update_id: str
    transaction_id: str
    attempt: int = Field(default=1, ge=1)
    supersedes_attempt: int | None = None
    lease_epoch: int = Field(ge=0)
    idempotency_key: str
    parent_policy_version: int = Field(ge=0)
    output_policy_version: int | None = None
    input_preupdate_envelope_sha256s: list[str] = Field(default_factory=list)
    update_input_event: EventRef | None = None
    gradient_accumulation_microbatches: int = Field(default=1, ge=1)
    optimizer_step_count_delta: int = Field(default=1, ge=1)
    trajectory_use_policy: str = "consume_once_v01"
    checkpoint_manifest_sha256: str | None = None
    optimizer_state_artifact: ArtifactRef | None = None
    failure_code: str | None = None


class UpdateInputEvent(EventBase):
    """Materialization of validated trajectories for the guarded update (§7.3.3)."""

    event_type: Literal["update_input_materialized"] = "update_input_materialized"
    update_id: str
    preupdate_envelope: EnvelopeRef
    preupdate_validation_decision: EventRef
    sequence_token_ids: ArtifactRef
    loss_mask: ArtifactRef
    authoritative_behavior_logprob_event: EventRef
    authoritative_behavior_logprobs: ArtifactRef
    reward_event: EventRef
    materialized_layout_sha256: str
    single_use_nonce_sha256: str
    tokenizer_called: bool = Field(default=False)


class GenerationEvent(EventBase):
    """Authoritative producer record of a rollout completion (§7.4)."""

    event_type: Literal["generation_finished"] = "generation_finished"
    request_id: str
    attempt_id: str
    prompt_id: str
    sample_index: int = Field(ge=0)
    runtime_id: str
    runtime_load_epoch: int = Field(ge=0)
    behavior_policy_version: int = Field(ge=0)
    checkpoint_manifest_sha256: str
    sync_event: EventRef
    sampling_config_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    prompt_span: list[int] = Field(min_length=2, max_length=2)
    completion_span: list[int] = Field(min_length=2, max_length=2)
    padding_spans: list[list[int]] = Field(default_factory=list)
    truncation_applied: bool = False
    terminal_status: Literal["success", "infra_error", "timeout", "invalid"] = "success"
    sequence_token_ids: ArtifactRef
    completion_target_mask: ArtifactRef
    loss_mask: ArtifactRef
    service_behavior_logprobs: ArtifactRef | None = None


class ScoringEvent(EventBase):
    """Exact recomputation of behavior log-probs by a checkpoint-bound scorer (§7.5)."""

    event_type: Literal["behavior_scoring_finished"] = "behavior_scoring_finished"
    source_generation_event: EventRef
    scorer_policy_version: int = Field(ge=0)
    scorer_checkpoint_manifest_sha256: str
    token_artifact_sha256: str
    scoring_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    behavior_logprobs: ArtifactRef


class RewardEvent(EventBase):
    """Rule-verifier reward with explicit protocol identity (§7.6)."""

    event_type: Literal["reward_finished"] = "reward_finished"
    reward_version: str
    evaluator_protocol_sha256: str
    source_generation_event: EventRef
    components: dict[str, float]
    terminal_status: Literal["success", "infra_error", "timeout", "invalid", "task_fail"] = "success"
    latency_ms: float = Field(ge=0.0)


class ValidationDecisionEvent(EventBase):
    """Wraps a ValidationDecision payload so it can be referenced by EventRef (§7.8)."""

    event_type: Literal["validation_decision"] = "validation_decision"
    decision_payload: ValidationDecision


class TrainingStepEvent(EventBase):
    """Per-step training metrics persisted to the event stream (D15 infra).

    Makes a training run fully recoverable from the event log alone
    (PM-2 prevention): loss / ratio / success / weight-delta are no
    longer only in the run log.
    """

    event_type: Literal["training_step_finished"] = "training_step_finished"
    update_id: str
    parent_policy_version: int
    output_policy_version: int
    rollout_sequences: int
    consumed_sequences: int
    success_rate: float
    loss: float
    ratio_p50: float
    ratio_max: float
    clip_fraction: float | None = None
    weight_delta_fp32_vs_v0: float | None = None


def event_from_payload(payload: dict) -> EventBase:
    """Rehydrate a typed event from its canonical JSON payload."""
    kind = payload["event_type"]
    model: type[EventBase]
    if kind == "validation_decision":
        model = ValidationDecisionEvent
    elif kind in {"canary_passed", "sync_requested", "sync_started", "runtime_loaded",
                  "sync_unknown", "sync_reconciled_canary_passed", "sync_retryable_old",
                  "sync_quarantined", "sync_failed", "sync_attempt_superseded"}:
        model = SyncEvent
    elif kind in {"update_started", "update_prepared", "update_committed", "update_unknown",
                  "update_restored_parent", "update_aborted", "update_attempt_superseded"}:
        model = UpdateEvent
    elif kind == "update_input_materialized":
        model = UpdateInputEvent
    elif kind == "generation_finished":
        model = GenerationEvent
    elif kind == "behavior_scoring_finished":
        model = ScoringEvent
    elif kind == "reward_finished":
        model = RewardEvent
    elif kind == "training_step_finished":
        model = TrainingStepEvent
    else:
        raise ValueError(f"unknown event_type {kind}")
    return model(**payload)
