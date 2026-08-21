"""Common artifact reference types (design doc §7.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EventRef(BaseModel):
    """Reference to a sealed event, never to an unsealed draft."""

    uri: str
    event_id: str
    event_sha256: str


class EnvelopeRef(BaseModel):
    """Reference to a sealed trajectory envelope."""

    uri: str
    envelope_id: str
    envelope_sha256: str


class ManifestRef(BaseModel):
    """Reference to a PolicyManifest or split manifest document."""

    uri: str
    manifest_id: str
    sha256: str


class ArtifactRef(BaseModel):
    """Content-addressed reference to a byte artifact.

    ``producer_event_id`` is the pre-allocated (unsealed) event ID so that
    event hashes never contain artifact refs that reference the event hash
    itself (no self-reference cycle, design doc §7.1).
    """

    uri: str
    media_type: str
    dtype: str | None = None
    shape: list[int] | None = None
    num_bytes: int = Field(ge=0)
    sha256: str
    producer_event_id: str
