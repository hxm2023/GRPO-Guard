"""Artifact store contract (design doc §7.1, §6.1)."""

import tempfile
from pathlib import Path

import pytest

from grpo_guard.store.artifact_store import ArtifactStore, HashMismatchError


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield ArtifactStore(Path(tmp))


def test_put_get_roundtrip(store):
    ref = store.put(b"hello", "application/octet-stream", "evt-1", dtype="int8", shape=[5])
    assert ref.num_bytes == 5
    assert store.get(ref) == b"hello"


def test_mutation_detected(store):
    ref = store.put(b"original", "application/octet-stream", "evt-1")
    path = store.blobs / ref.sha256
    path.write_bytes(b"tampered!")
    with pytest.raises(HashMismatchError):
        store.get(ref)
    assert store.verify(ref) is False


def test_idempotent_put_same_bytes(store):
    ref1 = store.put(b"same", "application/octet-stream", "evt-1")
    ref2 = store.put(b"same", "application/octet-stream", "evt-2")
    assert ref1.sha256 == ref2.sha256
    assert ref1.producer_event_id != ref2.producer_event_id


def test_missing_blob(store):
    from grpo_guard.schema.artifacts import ArtifactRef

    ref = ArtifactRef(uri="artifact://nope", media_type="x", num_bytes=1, sha256="0" * 64, producer_event_id="e")
    with pytest.raises(FileNotFoundError):
        store.get(ref)
    assert store.verify(ref) is False
