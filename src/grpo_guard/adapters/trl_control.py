"""Trainer control-plane adapter (design doc §6.1, §7.3.1-§7.3.2).

The ONLY producer of SyncEvent/UpdateEvent.  The runtime cannot self-report
"synced" without a control-plane event; the optimizer cannot claim a commit
without an UpdateCommitted event.  Every phase of a sync/update is its own
immutable event sharing sync_id/update_id.
"""

from __future__ import annotations

from grpo_guard.schema.events import SyncEvent, UpdateEvent
from grpo_guard.store.append_log import AppendLog
from grpo_guard.testing import now_utc


class TrlControlAdapter:
    def __init__(self, log: AppendLog, run_id: str, component_id: str = "trl_control", seq_provider=None):
        self.log = log
        self.run_id = run_id
        self.component_id = component_id
        self._seq = 0
        self._seq_provider = seq_provider

    def next_seq(self) -> int:
        if self._seq_provider is not None:
            return self._seq_provider()
        self._seq += 1
        return self._seq

    def _sync(
        self,
        event_type: str,
        sync_id: str,
        attempt: int,
        lease_epoch: int,
        source_policy_version: int,
        source_checkpoint_sha: str,
        target_runtime_id: str,
        upstream_adapter_id: str,
        upstream_operation: str,
        profile_sha: str,
        required_epoch: int | None,
        observed_load_epoch: int | None = None,
        observed_policy_version: int | None = None,
        status_detail: str | None = None,
    ) -> SyncEvent:
        ev = SyncEvent(
            event_id=f"{sync_id}-{event_type}-{self._seq:04d}",
            event_type=event_type,
            run_id=self.run_id,
            component_id=self.component_id,
            lifecycle_seq=self.next_seq(),
            created_at_utc=now_utc(),
            sync_id=sync_id,
            attempt=attempt,
            lease_epoch=lease_epoch,
            idempotency_key=f"{self.run_id}:{source_policy_version}:{target_runtime_id}",
            source_policy_version=source_policy_version,
            source_checkpoint_manifest_sha256=source_checkpoint_sha,
            target_runtime_id=target_runtime_id,
            observed_runtime_load_epoch=observed_load_epoch,
            observed_policy_version=observed_policy_version,
            upstream_adapter_id=upstream_adapter_id,
            upstream_operation=upstream_operation,
            compatibility_profile_sha256=profile_sha,
            status_detail=status_detail,
        ).seal()
        self.log.append(ev, required_epoch=required_epoch)
        return ev

    def sync_begin(
        self,
        policy_version: int,
        checkpoint_sha: str,
        lease_epoch: int,
        attempt: int = 1,
        required_epoch: int | None = None,
    ) -> list[SyncEvent]:
        """requested → started.  runtime_loaded is written ONLY by
        sync_complete, AFTER the actual per-parameter calls succeeded
        (P0-2: no self-reported load before the fact)."""
        sync_id = f"sync-run-{policy_version}"
        events = [
            self._sync("sync_requested", sync_id, attempt, lease_epoch, policy_version, checkpoint_sha,
                       "rollout-gpu1", "trl-vllm-server", "update_named_param", "profile", required_epoch),
            self._sync("sync_started", sync_id, attempt, lease_epoch, policy_version, checkpoint_sha,
                       "rollout-gpu1", "trl-vllm-server", "update_named_param", "profile", required_epoch),
        ]
        return events

    def sync_complete(
        self,
        policy_version: int,
        checkpoint_sha: str,
        lease_epoch: int,
        sync_id: str,
        observed_sync_calls: int,
        param_digest: str,
        attempt: int = 1,
        required_epoch: int | None = None,
    ) -> SyncEvent:
        """runtime_loaded, recorded ONLY after the caller observed the real
        update_named_param calls (count + name/shape digest)."""
        return self._sync(
            "runtime_loaded", sync_id, attempt, lease_epoch, policy_version, checkpoint_sha,
            "rollout-gpu1", "trl-vllm-server", "update_named_param", "profile", required_epoch,
            observed_load_epoch=policy_version + 1, observed_policy_version=policy_version,
            status_detail=f"observed {observed_sync_calls} update_named_param calls, digest {param_digest}",
        )

    def sync_failed(
        self,
        policy_version: int,
        checkpoint_sha: str,
        lease_epoch: int,
        sync_id: str,
        error: str,
        attempt: int = 1,
        required_epoch: int | None = None,
    ) -> SyncEvent:
        """sync_failed — the sync did NOT reach runtime_loaded."""
        return self._sync(
            "sync_failed", sync_id, attempt, lease_epoch, policy_version, checkpoint_sha,
            "rollout-gpu1", "trl-vllm-server", "update_named_param", "profile", required_epoch,
            status_detail=f"sync failed: {error[:200]}",
        )

    def canary_passed(
        self,
        policy_version: int,
        checkpoint_sha: str,
        lease_epoch: int,
        sync_id: str,
        drift: dict,
        attempt: int = 1,
        required_epoch: int | None = None,
    ) -> SyncEvent:
        return self._sync(
            "canary_passed", sync_id, attempt, lease_epoch, policy_version, checkpoint_sha,
            "rollout-gpu1", "trl-vllm-server", "update_named_param", "profile", required_epoch,
            observed_load_epoch=policy_version + 1, observed_policy_version=policy_version,
            status_detail=f"canary drift {drift}",
        )

    def canary_mismatch(
        self,
        policy_version: int,
        checkpoint_sha: str,
        lease_epoch: int,
        sync_id: str,
        drift: dict,
        attempt: int = 1,
        required_epoch: int | None = None,
    ) -> SyncEvent:
        """canary_mismatch — recorded when the sketch drifted (P0-2: a
        mismatch must never be written as canary_passed)."""
        return self._sync(
            "canary_mismatch", sync_id, attempt, lease_epoch, policy_version, checkpoint_sha,
            "rollout-gpu1", "trl-vllm-server", "update_named_param", "profile", required_epoch,
            observed_load_epoch=policy_version + 1, observed_policy_version=policy_version,
            status_detail=f"canary MISMATCH drift {drift}",
        )

    def update_committed(
        self,
        update_id: str,
        transaction_id: str,
        lease_epoch: int,
        parent_policy_version: int,
        output_policy_version: int,
        input_envelope_sha256s: list[str],
        checkpoint_manifest_sha256: str,
        update_input_event,
        required_epoch: int | None = None,
    ) -> UpdateEvent:
        ev = UpdateEvent(
            event_id=f"upd-commit-{update_id}",
            event_type="update_committed",
            run_id=self.run_id,
            component_id=self.component_id,
            lifecycle_seq=self.next_seq(),
            created_at_utc=now_utc(),
            update_id=update_id,
            transaction_id=transaction_id,
            attempt=1,
            lease_epoch=lease_epoch,
            idempotency_key=f"{self.run_id}:{update_id}",
            parent_policy_version=parent_policy_version,
            output_policy_version=output_policy_version,
            input_preupdate_envelope_sha256s=input_envelope_sha256s,
            update_input_event=update_input_event,
            gradient_accumulation_microbatches=1,
            optimizer_step_count_delta=1,
            trajectory_use_policy="consume_once_v01",
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        ).seal()
        self.log.append(ev, required_epoch=required_epoch)
        return ev
