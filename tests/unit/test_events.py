"""Event sealing and immutability (design doc §7.3)."""

import pytest

from grpo_guard import testing


def test_seal_excludes_own_hash_field():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    assert gen.event_sha256
    # verify_seal recomputes with the field excluded and matches
    assert gen.verify_seal()


def test_seal_is_immutable():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    gen.created_at_utc = "tampered"
    assert gen.verify_seal() is False


def test_seal_rejects_double_seal():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    with pytest.raises(ValueError):
        gen.seal()


def test_sync_chain_has_unique_event_ids():
    t = testing.build_trajectory()
    ids = [e.event_id for e in t.sync_events]
    assert len(ids) == len(set(ids))
    seqs = [e.lifecycle_seq for e in t.sync_events]
    assert seqs == sorted(seqs)


def test_envelope_seal_and_ref():
    t = testing.build_trajectory()
    assert t.envelope.verify_seal()
    ref = t.envelope.ref()
    assert ref.envelope_sha256 == t.envelope.envelope_sha256
