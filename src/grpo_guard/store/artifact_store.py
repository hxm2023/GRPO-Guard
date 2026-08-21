"""Content-addressed artifact store (design doc §7.1, §6.1).

Artifacts are addressed by SHA-256 of their bytes; writing is idempotent
(an existing blob must have the identical hash or it is a collision error).
Validation must read bytes back from this store and recompute hashes —
never trust the envelope's self-report.
"""

from __future__ import annotations

from pathlib import Path

from grpo_guard.schema.artifacts import ArtifactRef
from grpo_guard.store.canonical_json import sha256_bytes


class HashMismatchError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        data: bytes,
        media_type: str,
        producer_event_id: str,
        dtype: str | None = None,
        shape: list[int] | None = None,
    ) -> ArtifactRef:
        digest = sha256_bytes(data)
        path = self.blobs / digest
        if path.exists():
            existing = path.read_bytes()
            if existing != data:
                raise HashMismatchError(f"sha256 collision at {digest}")
        else:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return ArtifactRef(
            uri=f"artifact://{digest}",
            media_type=media_type,
            dtype=dtype,
            shape=shape,
            num_bytes=len(data),
            sha256=digest,
            producer_event_id=producer_event_id,
        )

    def get(self, ref: ArtifactRef) -> bytes:
        path = self.blobs / ref.sha256
        if not path.exists():
            raise FileNotFoundError(f"artifact blob {ref.sha256} missing")
        data = path.read_bytes()
        if sha256_bytes(data) != ref.sha256:
            raise HashMismatchError(
                f"artifact {ref.sha256} failed hash check (got {sha256_bytes(data)})"
            )
        if len(data) != ref.num_bytes:
            raise HashMismatchError(
                f"artifact {ref.sha256} size mismatch: ref says {ref.num_bytes}, actual {len(data)}"
            )
        return data

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            self.get(ref)
            return True
        except (FileNotFoundError, HashMismatchError):
            return False
