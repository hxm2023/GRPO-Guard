"""Deterministic trajectory factory: sealed events, artifacts, envelopes.

Used by unit/contract tests, frozen fixtures, and fault injectors.  The
factory always produces the canonical happy path; faults mutate exactly one
target field afterwards (design doc §12.3).
"""

from __future__ import annotations

import datetime
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from grpo_guard.adapters.countdown_reward import reward_protocol_sha256
from grpo_guard.schema.artifacts import EnvelopeRef, EventRef, ManifestRef
from grpo_guard.schema.decisions import ValidationDecision
from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
from grpo_guard.schema.events import (
    GenerationEvent,
    RewardEvent,
    ScoringEvent,
    SyncEvent,
    ValidationDecisionEvent,
)
from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.store.canonical_json import canonical_dumps, canonical_sha256

DEFAULT_TOKENIZER_SHA = hashlib.sha256(b"tokenizer-qwen3-4b-v1").hexdigest()
DEFAULT_TEMPLATE_SHA = hashlib.sha256(b"chat-template-qwen3-v1").hexdigest()
DEFAULT_SAMPLING_SHA = hashlib.sha256(b"sampling-grpo-v1").hexdigest()


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ArtifactStoreTmp:
    """Context-managed temp store for tests."""

    def __enter__(self) -> ArtifactStore:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return ArtifactStore(Path(self._tmp.name))

    def __exit__(self, *exc) -> None:
        self._tmp.cleanup()


_GLOBAL_TMP: list[ArtifactStore] = []


def global_store() -> ArtifactStore:
    """Process-lifetime temp store (auto-cleaned at interpreter exit)."""
    if not _GLOBAL_TMP:
        import atexit
        import tempfile

        dir_ = Path(tempfile.mkdtemp(prefix="grpo-guard-tests-"))
        store = ArtifactStore(dir_)
        _GLOBAL_TMP.append(store)

        @atexit.register
        def _cleanup() -> None:
            import shutil

            shutil.rmtree(dir_, ignore_errors=True)

    return _GLOBAL_TMP[0]


def _make_scoring(run_id: str, seq_num: int, policy_version: int, gen: GenerationEvent, seq_ref, lp_ref) -> ScoringEvent:
    return ScoringEvent(
        event_id=f"score-{policy_version}-{seq_num}",
        event_type="behavior_scoring_finished",
        run_id=run_id,
        component_id="behavior_scorer",
        lifecycle_seq=seq_num + 1,
        created_at_utc=now_utc(),
        input_events=[EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256)],
        output_artifacts=[lp_ref],
        source_generation_event=EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256),
        scorer_policy_version=policy_version,
        scorer_checkpoint_manifest_sha256=make_policy_manifest(policy_version).checkpoint_manifest_sha256,
        token_artifact_sha256=seq_ref.sha256,
        scoring_dtype="bf16",
        behavior_logprobs=lp_ref,
    ).seal()


def make_policy_manifest(
    policy_version: int,
    parent_policy_version: int | None = None,
    checkpoint_sha: str | None = None,
    code_commit_sha: str = "abc1234",
) -> PolicyManifest:
    return PolicyManifest(
        manifest_id=f"pm-{policy_version}",
        model_id="Qwen/Qwen3-4B",
        model_revision="rev-fixed",
        policy_version=policy_version,
        parent_policy_version=parent_policy_version,
        weights=[],
        checkpoint_manifest_sha256=checkpoint_sha or hashlib.sha256(f"ckpt-v{policy_version}".encode()).hexdigest(),
        tokenizer_sha256=DEFAULT_TOKENIZER_SHA,
        chat_template_sha256=DEFAULT_TEMPLATE_SHA,
        precision="bf16",
        adapter_kind="full",
        code_commit_sha=code_commit_sha,
        config_sha256=hashlib.sha256(b"grpo-config-v1").hexdigest(),
    )


def make_split_manifest(split_name: Literal["train", "calibration", "held_out"] = "train", prompt_ids: list[str] | None = None) -> SplitManifest:
    return SplitManifest(
        split_id=f"split-{split_name}",
        split_name=split_name,
        prompt_ids=prompt_ids or ["countdown-0001"],
        content_sha256s={},
    )


def make_sync_chain(run_id: str, policy_version: int, lease_epoch: int, start_seq: int = 0) -> list[SyncEvent]:
    """Requested → started → runtime_loaded → canary_passed (terminal success)."""
    sync_id = f"sync-run-{policy_version}-r1"
    checkpoint = make_policy_manifest(policy_version).checkpoint_manifest_sha256
    return [
        SyncEvent(
            event_id=f"sync-{policy_version}-req",
            event_type="sync_requested",
            run_id=run_id,
            component_id="trl_control",
            lifecycle_seq=start_seq,
            created_at_utc=now_utc(),
            sync_id=sync_id,
            attempt=1,
            lease_epoch=lease_epoch,
            idempotency_key=f"{run_id}:{policy_version}:rollout-gpu1",
            source_policy_version=policy_version,
            source_checkpoint_manifest_sha256=checkpoint,
            target_runtime_id="rollout-gpu1",
            upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="profile",
        ).seal(),
        SyncEvent(
            event_id=f"sync-{policy_version}-started",
            event_type="sync_started",
            run_id=run_id,
            component_id="trl_control",
            lifecycle_seq=start_seq + 1,
            created_at_utc=now_utc(),
            sync_id=sync_id,
            attempt=1,
            lease_epoch=lease_epoch,
            idempotency_key=f"{run_id}:{policy_version}:rollout-gpu1",
            source_policy_version=policy_version,
            source_checkpoint_manifest_sha256=checkpoint,
            target_runtime_id="rollout-gpu1",
            upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="profile",
        ).seal(),
        SyncEvent(
            event_id=f"sync-{policy_version}-loaded",
            event_type="runtime_loaded",
            run_id=run_id,
            component_id="trl_control",
            lifecycle_seq=start_seq + 2,
            created_at_utc=now_utc(),
            sync_id=sync_id,
            attempt=1,
            lease_epoch=lease_epoch,
            idempotency_key=f"{run_id}:{policy_version}:rollout-gpu1",
            source_policy_version=policy_version,
            source_checkpoint_manifest_sha256=checkpoint,
            target_runtime_id="rollout-gpu1",
            previous_runtime_load_epoch=policy_version - 1 if policy_version > 0 else 0,
            observed_runtime_load_epoch=policy_version + 1,
            observed_policy_version=policy_version,
            upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="profile",
        ).seal(),
        SyncEvent(
            event_id=f"sync-{policy_version}-canary",
            event_type="canary_passed",
            run_id=run_id,
            component_id="trl_control",
            lifecycle_seq=start_seq + 3,
            created_at_utc=now_utc(),
            sync_id=sync_id,
            attempt=1,
            lease_epoch=lease_epoch,
            idempotency_key=f"{run_id}:{policy_version}:rollout-gpu1",
            source_policy_version=policy_version,
            source_checkpoint_manifest_sha256=checkpoint,
            target_runtime_id="rollout-gpu1",
            observed_runtime_load_epoch=policy_version + 1,
            observed_policy_version=policy_version,
            upstream_adapter_id="trl-vllm-server",
            upstream_operation="update_named_param",
            compatibility_profile_sha256="profile",
            status_detail="canary suite 4/4 within tolerance",
        ).seal(),
    ]


@dataclass
class Trajectory:
    """Everything a validator needs: events, manifests, envelope, artifacts."""

    run_id: str
    events: dict[str, object]
    policy_manifest: PolicyManifest
    split_manifest: SplitManifest
    envelope: TrajectoryEnvelope
    store: ArtifactStore
    sequence: np.ndarray
    target_mask: np.ndarray
    loss_mask: np.ndarray
    logprobs: np.ndarray
    completion_text: str
    goal: int
    target_numbers: list[int]
    reward_components: dict = field(default_factory=dict)
    sync_events: list[SyncEvent] = field(default_factory=list)
    sequence_ref: object = None  # ArtifactRef of the producer sequence
    bogus_sequence_ref: object = None  # F3 held-out: re-encoded sequence

    def lifecycle_seq_of(self, event_id: str) -> int:
        return self.events[event_id].lifecycle_seq

    def next_seq(self) -> int:
        return max(e.lifecycle_seq for e in self.events.values()) + 1


def build_trajectory(
    store: ArtifactStore | None = None,
    run_id: str = "run-test-0001",
    policy_version: int = 0,
    completion_text: str = "(1+2)*3=9",
    target_numbers: list[int] | None = None,
    goal: int = 9,
    prompt_span: list[int] | None = None,
    completion_span: list[int] | None = None,
    padding_spans: list[list[int]] | None = None,
    logprob_source: Literal["generation_service", "exact_behavior_scorer", "none"] = "generation_service",
    diagnostic_non_authoritative_allowed: bool = False,
    stage: Literal["pre_reward", "pre_update"] = "pre_reward",
    parent_identity: ValidationDecisionEvent | None = None,
    parent_envelope_sha256: str | None = None,
    truncation_applied: bool = False,
    seq: np.ndarray | None = None,
) -> Trajectory:
    """Build one canonical valid trajectory with all events sealed.

    Default layout (design doc §7.9): T=12, P=4, C=8.
    """
    if store is None:
        store = global_store()
    if seq is None:
        seq = np.arange(12, dtype=np.int32)
    T = int(seq.shape[0])
    P = (prompt_span or [0, 4])[1]
    C_end = (completion_span or [4, T])[1]
    completion_start = (completion_span or [4, T])[0]

    target = np.zeros(T, dtype=np.int8)
    target[completion_start:C_end] = 1
    for s, e in padding_spans or []:
        target[s:e] = 0
    loss = target[1:].copy()

    target_numbers = target_numbers or [1, 2, 3]
    seq_ref = store.put(seq.tobytes(), "application/octet-stream", f"gen-{policy_version}-prod", dtype="int32", shape=[T])
    target_ref = store.put(target.tobytes(), "application/octet-stream", f"gen-{policy_version}-prod", dtype="int8", shape=[T])
    loss_ref = store.put(loss.tobytes(), "application/octet-stream", f"gen-{policy_version}-prod", dtype="int8", shape=[T - 1])
    C = C_end - completion_start
    logprobs = np.full(C, -0.5, dtype=np.float16)
    lp_ref = store.put(logprobs.tobytes(), "application/octet-stream", f"gen-{policy_version}-prod", dtype="bf16", shape=[C])

    sync_events = make_sync_chain(run_id, policy_version, lease_epoch=1)
    canary_ref = EventRef(uri="event://sync-0-canary", event_id=sync_events[-1].event_id, event_sha256=sync_events[-1].event_sha256)

    events: dict[str, object] = {e.event_id: e for e in sync_events}

    seq_num = 10 + policy_version * 100
    gen = GenerationEvent(
        event_id=f"gen-{policy_version}-{seq_num}",
        event_type="generation_finished",
        run_id=run_id,
        component_id="vllm_runtime",
        lifecycle_seq=seq_num,
        created_at_utc=now_utc(),
        output_artifacts=[seq_ref, target_ref, loss_ref] + ([lp_ref] if logprob_source == "generation_service" else []),
        # service logprobs present iff declared source is generation_service
        request_id=f"req-{policy_version}-{seq_num}",
        attempt_id=f"att-{policy_version}-{seq_num}",
        prompt_id="countdown-0001",
        sample_index=0,
        runtime_id="rollout-gpu1",
        runtime_load_epoch=policy_version + 1,
        behavior_policy_version=policy_version,
        checkpoint_manifest_sha256=make_policy_manifest(policy_version).checkpoint_manifest_sha256,
        sync_event=canary_ref,
        sampling_config_sha256=DEFAULT_SAMPLING_SHA,
        tokenizer_sha256=DEFAULT_TOKENIZER_SHA,
        chat_template_sha256=DEFAULT_TEMPLATE_SHA,
        prompt_span=prompt_span or [0, P],
        completion_span=[completion_start, C_end],
        padding_spans=padding_spans or [],
        truncation_applied=truncation_applied,
        terminal_status="success",
        sequence_token_ids=seq_ref,
        completion_target_mask=target_ref,
        loss_mask=loss_ref,
        service_behavior_logprobs=lp_ref if logprob_source == "generation_service" else None,
    ).seal()
    events[gen.event_id] = gen

    scoring: ScoringEvent | None = None
    if logprob_source == "exact_behavior_scorer":
        scoring = _make_scoring(run_id, seq_num, policy_version, gen, seq_ref, lp_ref)
        events[scoring.event_id] = scoring
    elif logprob_source not in ("generation_service", "none"):
        raise ValueError(logprob_source)

    reward = RewardEvent(
        event_id=f"reward-{policy_version}-{seq_num}",
        event_type="reward_finished",
        run_id=run_id,
        component_id="countdown_reward",
        lifecycle_seq=seq_num + (2 if scoring else 1),
        created_at_utc=now_utc(),
        input_events=[EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256)],
        reward_version="countdown-rule-v1",
        evaluator_protocol_sha256=reward_protocol_sha256(),
        source_generation_event=EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256),
        components={"correctness": 1.0, "format": 1.0},
        terminal_status="success",
        latency_ms=3.2,
    ).seal()
    events[reward.event_id] = reward

    policy_manifest = make_policy_manifest(policy_version)
    split_manifest = make_split_manifest()

    authoritative_event = EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256)
    if scoring is not None:
        authoritative_event = EventRef(uri=f"event://{scoring.event_id}", event_id=scoring.event_id, event_sha256=scoring.event_sha256)

    declared_source = "generation_service" if logprob_source == "none" else logprob_source
    contract = TrainingContract(
        protocol="strict_on_policy",
        trainer_parent_policy_version=policy_version,
        consuming_update_id=f"update-{policy_version + 1}",
        max_policy_lag_versions=0,
        importance_correction=None,
        behavior_logprob_source=declared_source,
        authoritative_behavior_logprob_event=authoritative_event,
        diagnostic_non_authoritative_logprobs_allowed=diagnostic_non_authoritative_allowed,
    )

    envelope = TrajectoryEnvelope(
        envelope_id=f"env-{policy_version}-{seq_num}-{'pre_reward' if stage == 'pre_reward' else 'pre_update'}",
        envelope_stage=stage,
        run_id=run_id,
        request_id=f"req-{policy_version}-{seq_num}",
        generation_event=EventRef(uri=f"event://{gen.event_id}", event_id=gen.event_id, event_sha256=gen.event_sha256),
        scoring_event=(
            EventRef(uri=f"event://{scoring.event_id}", event_id=scoring.event_id, event_sha256=scoring.event_sha256)
            if scoring else None
        ),
        reward_event=(EventRef(uri=f"event://{reward.event_id}", event_id=reward.event_id, event_sha256=reward.event_sha256)
                      if stage == "pre_update" else None),
        policy_manifest=ManifestRef(uri=f"manifest://{policy_manifest.manifest_id}", manifest_id=policy_manifest.manifest_id, sha256=policy_manifest.checkpoint_manifest_sha256),
        split_manifest=ManifestRef(uri=f"manifest://{split_manifest.split_id}", manifest_id=split_manifest.split_id, sha256=canonical_sha256(split_manifest.model_dump(mode="json"))),
        parent_envelope_sha256=parent_envelope_sha256,
        parent_identity_decision=(EventRef(uri=f"event://{parent_identity.event_id}", event_id=parent_identity.event_id, event_sha256=parent_identity.event_sha256) if parent_identity else None),
        training_contract=contract,
    ).seal()

    return Trajectory(
        run_id=run_id,
        events=events,
        policy_manifest=policy_manifest,
        split_manifest=split_manifest,
        envelope=envelope,
        store=store,
        sequence=seq,
        target_mask=target,
        loss_mask=loss,
        logprobs=logprobs,
        completion_text=completion_text,
        goal=goal,
        target_numbers=target_numbers,
        reward_components=reward.components,
        sync_events=sync_events,
        sequence_ref=seq_ref,
    )


def validation_event(run_id: str, decision: ValidationDecision, lifecycle_seq: int) -> ValidationDecisionEvent:
    return ValidationDecisionEvent(
        event_id=f"vdec-{uuid.uuid4().hex[:12]}",
        event_type="validation_decision",
        run_id=run_id,
        component_id="validator",
        lifecycle_seq=lifecycle_seq,
        created_at_utc=now_utc(),
        decision_payload=decision,
    ).seal()
