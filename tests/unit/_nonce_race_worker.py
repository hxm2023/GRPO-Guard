"""Multiprocessing worker for the nonce race test (P0-2).

Kept torch-free on purpose: Windows spawn re-imports this module in every
child process, and importing torch per-child is slow/resource-heavy.
"""

from __future__ import annotations


def try_consume(db_path: str, nonce: str) -> int:
    from grpo_guard.adapters.guarded_update import NonceRegistry

    try:
        NonceRegistry(db_path).consume(nonce)
        return 1
    except Exception:
        return 0
