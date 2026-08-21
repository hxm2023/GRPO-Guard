"""Append-only event log with single-writer lease + fencing (design doc §6.2, §7.3.4).

State is derived from immutable events via a deterministic reducer; the last
log line never overwrites anything.  A lease marker carries (lease_id, epoch);
every write checks the marker and fails closed if the epoch moved on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypeVar

from pydantic import BaseModel

from grpo_guard.schema.events import EventBase
from grpo_guard.store.canonical_json import canonical_dumps

T = TypeVar("T", bound=BaseModel)


class LeaseError(RuntimeError):
    pass


class EventAlreadyAppended(ValueError):
    pass


class EventSealError(ValueError):
    pass


class AppendLog:
    def __init__(self, root: Path, run_id: str, lease_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.lease_id = lease_id
        self.events_dir = self.root / "events"
        self.edges_dir = self.root / "edges"
        self.lease_file = self.root / "lease.json"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.edges_dir.mkdir(parents=True, exist_ok=True)

    # ---- lease / fencing -------------------------------------------------

    def acquire_lease(self) -> int:
        """Acquire the single-writer lease; returns the new fencing epoch.

        Re-acquiring with the same lease_id bumps the epoch; a different
        lease_id while held raises LeaseError.
        """
        marker = self._read_lease()
        if marker is None:
            epoch = 1
        elif marker["lease_id"] == self.lease_id:
            epoch = marker["epoch"] + 1
        else:
            raise LeaseError(
                f"lease held by {marker['lease_id']} (epoch {marker['epoch']})"
            )
        self._write_lease(self.lease_id, epoch)
        return epoch

    def current_epoch(self) -> int | None:
        marker = self._read_lease()
        return marker["epoch"] if marker else None

    def _read_lease(self) -> dict | None:
        if not self.lease_file.exists():
            return None
        return json.loads(self.lease_file.read_text(encoding="utf-8"))

    def _write_lease(self, lease_id: str, epoch: int) -> None:
        self.lease_file.write_text(
            json.dumps({"lease_id": lease_id, "epoch": epoch}), encoding="utf-8"
        )

    # ---- event writes ----------------------------------------------------

    def append(self, event: EventBase, required_epoch: int | None = None) -> EventBase:
        """Append a sealed event.  Idempotent by event_id; never overwrites."""
        if not event.is_sealed():
            raise EventSealError(f"event {event.event_id} not sealed")
        if event.run_id != self.run_id:
            raise ValueError(
                f"event run_id {event.run_id} != log run_id {self.run_id}"
            )
        marker = self._read_lease()
        if marker is None or marker["lease_id"] != self.lease_id:
            raise LeaseError(f"lease not held by {self.lease_id}")
        if required_epoch is not None and marker["epoch"] != required_epoch:
            raise LeaseError(
                f"stale writer: lease epoch is {marker['epoch']}, required {required_epoch}"
            )
        path = self.events_dir / f"{event.event_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("event_sha256") == event.event_sha256:
                return event  # idempotent success
            raise EventAlreadyAppended(
                f"event_id {event.event_id} already appended with different payload"
            )
        path.write_text(canonical_dumps(event.model_dump(mode="json")).decode("utf-8"), encoding="utf-8")
        return event

    def append_provenance_edge(self, event_id: str, artifact_sha256: str, role: str) -> None:
        """Append-only provenance edges (event → output artifact), design doc §7.3.

        One event may own many artifacts; the edge list is append-only.  A
        conflict means the same (event, artifact) was registered with a
        different role — a genuine inconsistency, never silently merged.
        """
        path = self.edges_dir / f"{event_id}.json"
        edge = {"event_id": event_id, "artifact_sha256": artifact_sha256, "role": role}
        existing: list[dict] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            for e in existing:
                if e["artifact_sha256"] == artifact_sha256:
                    if e["role"] != role:
                        raise ValueError(f"conflicting provenance edge for {event_id}:{artifact_sha256}")
                    return  # idempotent: identical edge already recorded
        existing.append(edge)
        path.write_text(canonical_dumps(existing).decode("utf-8"), encoding="utf-8")

    # ---- reads -----------------------------------------------------------

    def get(self, event_id: str, model: type[T] | None = None) -> dict:
        path = self.events_dir / f"{event_id}.json"
        if not path.exists():
            raise KeyError(f"event {event_id} not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if model is not None:
            return model(**payload)
        return payload

    def iterate(self) -> Iterator[dict]:
        for path in sorted(self.events_dir.glob("*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))
