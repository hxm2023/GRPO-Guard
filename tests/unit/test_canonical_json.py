"""Canonical JSON contract (design doc §7)."""

import json

import pytest

from grpo_guard.store.canonical_json import canonical_dumps, canonical_sha256, sha256_bytes


def test_canonical_stability_across_key_order():
    a = {"z": 1, "a": [2, {"b": 3}]}
    b = {"a": [2, {"b": 3}], "z": 1}
    assert canonical_dumps(a) == canonical_dumps(b)


def test_canonical_utf8_and_compact():
    out = canonical_dumps({"k": "中文", "n": 1})
    assert out == b'{"k":"\xe4\xb8\xad\xe6\x96\x87","n":1}'


def test_canonical_rejects_nan_and_infinity():
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("inf")})


def test_sha256_stable():
    assert canonical_sha256({"b": 1, "a": 2}) == canonical_sha256({"a": 2, "b": 1})
    assert len(sha256_bytes(b"x")) == 64


def test_canonical_roundtrip():
    obj = {"a": [1, 2.5, "s"], "b": {"c": None}}
    assert json.loads(canonical_dumps(obj).decode("utf-8")) == obj
