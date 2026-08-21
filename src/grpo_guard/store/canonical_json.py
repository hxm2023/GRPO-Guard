"""Canonical JSON serialization (design doc §7).

Canonical form: UTF-8, sorted keys, compact separators, no NaN/Infinity.
Stable across Python versions and platforms; the byte sequence is the hash
input for all content-addressing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(obj: Any) -> bytes:
    """Serialize ``obj`` to canonical UTF-8 JSON bytes.

    ``allow_nan=False`` makes NaN/Infinity a hard error so no event or
    artifact manifest can silently carry them.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(obj: Any) -> str:
    return sha256_bytes(canonical_dumps(obj))
