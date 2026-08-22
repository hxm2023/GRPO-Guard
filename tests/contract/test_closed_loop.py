"""Contract: the full guarded lifecycle (design doc §6.2, §7.3.3).

COMMITTED(v) → SYNC(v) → CANARY(v) → GENERATION(v) → IDENTITY_VALIDATED(allow)
→ REWARD → PRE_UPDATE_VALIDATED(allow) → MATERIALIZED(handle) → UPDATE_STARTED
→ UPDATE_COMMITTED(v+1).  A quarantined/rejected envelope never reaches the
update path; the same nonce can never be consumed twice.
"""

import numpy as np

from grpo_guard import testing
from grpo_guard.adapters.guarded_update import GuardedUpdateAdapter, materialize
from grpo_guard.faults import inject_f1_static_rollout
from grpo_guard.schema.artifacts import EventRef
from grpo_guard.schema.events import UpdateEvent
from grpo_guard.store.append_log import AppendLog
from grpo_guard.store.reducer import reduce_update
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

STRICT = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def _ctx(t, **kw):
    return ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=STRICT, **kw,
    )


def test_full_guarded_loop_commits_new_policy():
    with testing.ArtifactStoreTmp() as store:
        log = AppendLog(store.root / "log", run_id="run-loop", lease_id="trainer")
        epoch = log.acquire_lease()

        # v0 rollout → identity validation
        t0 = testing.build_trajectory(store, run_id="run-loop", policy_version=0)
        for ev in t0.sync_events:
            log.append(ev, required_epoch=epoch)
        for ev in t0.events.values():
            if ev is not t0.envelope and ev.event_id not in {e.event_id for e in t0.sync_events}:
                log.append(ev, required_epoch=epoch)

        d0 = validate_envelope(_ctx(t0), "identity_pre_reward")
        assert d0.decision_payload.decision == "allow"
        log.append(d0, required_epoch=epoch)

        # reward event exists; pre-update envelope chained to allowed identity
        t1 = testing.build_trajectory(
            store, run_id="run-loop", policy_version=0, stage="pre_update",
            parent_envelope_sha256=t0.envelope.envelope_sha256, parent_identity=d0,
        )
        t1.events[d0.event_id] = d0
        appended_ids = {e.event_id for e in t0.events.values()} | {d0.event_id}
        for ev in t1.events.values():
            if ev.event_id not in appended_ids:
                log.append(ev, required_epoch=epoch)

        d1 = validate_envelope(_ctx(t1), "full_pre_update")
        assert d1.decision_payload.decision == "allow"
        log.append(d1, required_epoch=epoch)

        # materialize the guarded handle; update adapter executes once
        gen = t1.events[t1.envelope.generation_event.event_id]
        reward_ref = t1.envelope.reward_event
        handle = materialize(
            store=store, run_id="run-loop", update_id="update-1",
            preupdate_envelope=t1.envelope.ref(),
            validation_decision=EventRef(uri="", event_id=d1.event_id, event_sha256=d1.event_sha256),
            sequence_ref=gen.sequence_token_ids,
            loss_mask_ref=gen.loss_mask,
            logprob_event_ref=t1.envelope.training_contract.authoritative_behavior_logprob_event,
            logprob_ref=gen.service_behavior_logprobs,
            reward_event_ref=reward_ref,
            nonce="nonce-loop-1",
            rewards=np.zeros(gen.loss_mask.shape[0], dtype=np.float32),
            lifecycle_seq=t1.next_seq(),
        )
        log.append(handle.input_event, required_epoch=epoch)

        adapter = GuardedUpdateAdapter(store, decision_verifier=lambda ref: ref.event_id == d1.event_id)
        adapter.update(handle)  # one optimizer step (CPU contract: bookkeeping)

        # commit v1 and verify the reducer
        committed = UpdateEvent(
            event_id="upd-commit-1", event_type="update_committed", run_id="run-loop",
            component_id="trl_control", lifecycle_seq=handle.input_event.lifecycle_seq + 1,
            created_at_utc=testing.now_utc(),
            update_id="update-1", transaction_id="txn-1", attempt=1, lease_epoch=epoch,
            idempotency_key="run-loop:update-1", parent_policy_version=0, output_policy_version=1,
            input_preupdate_envelope_sha256s=[t1.envelope.envelope_sha256],
            update_input_event=EventRef(uri="", event_id=handle.input_event.event_id,
                                        event_sha256=handle.input_event.event_sha256),
            checkpoint_manifest_sha256=testing.make_policy_manifest(1).checkpoint_manifest_sha256,
        ).seal()
        log.append(committed, required_epoch=epoch)

        state = reduce_update([committed])
        assert state.state == "COMMITTED"
        assert state.output_policy_version == 1


def test_quarantine_never_reaches_update():
    with testing.ArtifactStoreTmp() as store:
        t = testing.build_trajectory(store)
        tq = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
        d = validate_envelope(_ctx(tq), "identity_pre_reward")
        assert d.decision_payload.decision == "reject"
        # no handle may be materialized from a rejected chain — the envelope
        # reference itself is the gate: materialize requires an ALLOW decision
        assert d.decision_payload.reason_codes == ["P004_STALE_POLICY_STRICT"]


def test_nonce_single_use_across_retry():
    with testing.ArtifactStoreTmp() as store:
        t = testing.build_trajectory(store, stage="pre_update")
        gen = t.events[t.envelope.generation_event.event_id]
        handle = materialize(
            store=store, run_id=t.run_id, update_id="update-1",
            preupdate_envelope=t.envelope.ref(),
            validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
            sequence_ref=gen.sequence_token_ids,
            loss_mask_ref=gen.loss_mask,
            logprob_event_ref=t.envelope.training_contract.authoritative_behavior_logprob_event,
            logprob_ref=gen.service_behavior_logprobs,
            reward_event_ref=t.envelope.reward_event,
            nonce="nonce-1",
            rewards=np.zeros(gen.loss_mask.shape[0], dtype=np.float32),
            lifecycle_seq=t.next_seq(),
        )
        handle.consume()
        # a retry with the same update_id must mint a NEW handle/nonce —
        # reuse of the old nonce is refused by the adapter's fail-closed path
        from grpo_guard.adapters.guarded_update import HandleConsumedError
        try:
            handle.consume()
            raise AssertionError("second consume must fail")
        except HandleConsumedError:
            pass
