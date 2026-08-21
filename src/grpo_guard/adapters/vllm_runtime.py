"""vLLM runtime adapter: the ONLY producer of generation events and token
artifacts (design doc §6.1).

Wraps TRL's VLLMGeneration and emits GenerationEvent from the server's OWN
token ids (prompt_ids + completion_ids) and service logprobs — no
re-tokenization anywhere on this path.  The event is sealed and appended
with its artifacts before anything else can reference it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from grpo_guard.schema.artifacts import ArtifactRef, EventRef
from grpo_guard.schema.events import GenerationEvent
from grpo_guard.store.append_log import AppendLog
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.testing import now_utc


class VLLMRuntimeAdapter:
    def __init__(
        self,
        store: ArtifactStore,
        log: AppendLog,
        run_id: str,
        runtime_id: str,
        component_id: str = "vllm_runtime",
        seq_provider=None,
    ):
        self.store = store
        self.log = log
        self.run_id = run_id
        self.runtime_id = runtime_id
        self.component_id = component_id
        self._seq = 0
        self._load_epoch = 0
        self._seq_provider = seq_provider  # run-wide strictly increasing seq

    def next_seq(self) -> int:
        if self._seq_provider is not None:
            return self._seq_provider()
        self._seq += 1
        return self._seq

    def emit_generation(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        service_logprobs: list[float] | None,
        behavior_policy_version: int,
        checkpoint_manifest_sha256: str,
        sync_event: EventRef,
        tokenizer_sha256: str,
        chat_template_sha256: str,
        sampling_config_sha256: str,
        prompt_id: str,
        request_id: str,
        truncation_applied: bool = False,
        terminal_status: str = "success",
        required_epoch: int | None = None,
    ) -> GenerationEvent:
        """Emit a sealed GenerationEvent; the token artifacts come from the
        server's sampled ids (authoritative), masks are canonical §7.9."""
        seq_arr = np.asarray(prompt_ids + completion_ids, dtype=np.int32)
        T = int(seq_arr.shape[0])
        P = len(prompt_ids)
        C = len(completion_ids)

        target = np.zeros(T, dtype=np.int8)
        target[P:T] = 1
        loss = target[1:].copy()

        seq_no = self.next_seq()
        ev_id = f"gen-{behavior_policy_version}-{self.run_id}-{seq_no:04d}"
        seq_ref = self.store.put(seq_arr.tobytes(), "application/octet-stream", ev_id, dtype="int32", shape=[T])
        target_ref = self.store.put(target.tobytes(), "application/octet-stream", ev_id, dtype="int8", shape=[T])
        loss_ref = self.store.put(loss.tobytes(), "application/octet-stream", ev_id, dtype="int8", shape=[T - 1])

        lp_ref: ArtifactRef | None = None
        if service_logprobs is not None:
            lp = np.asarray(service_logprobs, dtype=np.float32)
            if lp.shape[0] != C:
                raise ValueError(f"service logprobs length {lp.shape[0]} != completion length {C}")
            lp_ref = self.store.put(lp.tobytes(), "application/octet-stream", ev_id, dtype="float32", shape=[C])

        event = GenerationEvent(
            event_id=ev_id,
            event_type="generation_finished",
            run_id=self.run_id,
            component_id=self.component_id,
            lifecycle_seq=seq_no,
            created_at_utc=now_utc(),
            output_artifacts=[seq_ref, target_ref, loss_ref] + ([lp_ref] if lp_ref else []),
            request_id=request_id,
            attempt_id=f"att-{request_id}",
            prompt_id=prompt_id,
            sample_index=0,
            runtime_id=self.runtime_id,
            runtime_load_epoch=self._load_epoch,
            behavior_policy_version=behavior_policy_version,
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            sync_event=sync_event,
            sampling_config_sha256=sampling_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
            prompt_span=[0, P],
            completion_span=[P, T],
            padding_spans=[],
            truncation_applied=truncation_applied,
            terminal_status=terminal_status,
            sequence_token_ids=seq_ref,
            completion_target_mask=target_ref,
            loss_mask=loss_ref,
            service_behavior_logprobs=lp_ref,
        ).seal()
        self.log.append(event, required_epoch=required_epoch)
        for ref in event.output_artifacts:
            self.log.append_provenance_edge(event.event_id, ref.sha256, "producer")
        return event

    def set_load_epoch(self, epoch: int) -> None:
        self._load_epoch = epoch
