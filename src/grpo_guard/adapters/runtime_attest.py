"""Dual-source runtime attestation (P1-1).

TRL's vLLM serve exposes ``/get_sequence_logprobs``: the logprobs the
server's LOADED weights assign to fixed token sequences.  The trainer
computes the same quantity from its own model on the same frozen canary
sequences.  If the server serves stale weights, the two fingerprints
diverge — observed at the API level (server-attested), not just
caller-reported sync calls.

Honest scope: this is a behavioral fingerprint (logprob values over a
few frozen sequences), not a byte-level parameter digest — identical
weights give near-identical logprobs, different weights give measurably
different ones within a small noise tolerance.  It complements the
caller-observed 398 update_named_param calls with an independent
server-observed signal.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request

import numpy as np

CANARY = [
    ("Use the numbers [4, 5, 6] exactly once to reach 40.\nReturn only the arithmetic expression.", 17),
    ("Use the numbers [1, 2, 3] exactly once to reach 9.\nReturn only the arithmetic expression.", 17),
    ("Use the numbers [3, 4, 7] exactly once to reach 32.\nReturn only the arithmetic expression.", 17),
    ("A farmer has 12 cows and buys 7 more. How many cows does he have?\nAnswer with only the final number.", 14),
    ("Tom reads 5 pages a day for a whole week. How many pages does he read?\nAnswer with only the final number.", 16),
]


def server_logprob_fingerprint(host: str, port: int, sequences: list[list[int]],
                               prompt_lengths: list[int],
                               top_logprobs: int = 16, timeout: float = 60.0) -> dict:
    """POST /get_sequence_logprobs and return {digest, arrays, info}."""
    payload = {
        "sequences": [list(s) for s in sequences],
        "prompt_lengths": [int(p) for p in prompt_lengths],
        "top_logprobs": top_logprobs,
        "temperature": 1.0,
        "response_format": "json",
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/get_sequence_logprobs/",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    logprobs = data.get("logprobs") or []
    arrays = []
    for seq_lp in logprobs:
        rows = []
        for pos in seq_lp:
            topk = [float(x) if x is not None else None for x in (pos or [])]
            rows.append(topk)
        arrays.append(np.asarray(rows, dtype=np.float64))
    return {"digest": _digest(arrays), "arrays": arrays,
            "n_sequences": len(arrays), "n_positions": [a.shape[0] for a in arrays]}


def model_logprob_fingerprint(model, tokenizer, sequences: list[list[int]],
                              prompt_lengths: list[int],
                              top_logprobs: int = 16) -> dict:
    """Trainer-side fingerprint: top-k logprobs from the model's own weights."""
    import torch

    model.eval()
    arrays = []
    with torch.no_grad():
        for seq, P in zip(sequences, prompt_lengths):
            ids = torch.tensor([seq], device=next(model.parameters()).device)
            out = model(ids)
            logits = out[0] if isinstance(out, (tuple, list)) else out.logits
            logits = logits[0].float()  # [seq_len, vocab]
            comp = logits[P - 1:-1]  # logprobs of completion tokens t at positions t-1
            topk = torch.topk(comp, k=min(top_logprobs, comp.shape[-1]), dim=-1)
            rows = topk.values.cpu().numpy() - torch.logsumexp(comp, dim=-1, keepdim=True).cpu().numpy()
            arrays.append(np.asarray(rows, dtype=np.float64))
    model.train()
    return {"digest": _digest(arrays), "arrays": arrays,
            "n_sequences": len(arrays), "n_positions": [a.shape[0] for a in arrays]}


def _digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.round(np.asarray(a, dtype=np.float64), 6).tobytes())
    return h.hexdigest()


def drift(server: dict, model: dict) -> dict:
    """Max abs logprob difference between the two fingerprints + verdict.

    Verdict threshold: 1e-2 — identical weights give ~1e-6 noise (same
    math on the same tensors); a stale/partially-updated runtime shows
    materially larger drift on at least one canary position.
    """
    max_drift = 0.0
    for sa, ma in zip(server["arrays"], model["arrays"]):
        if sa.shape != ma.shape:
            continue  # shape mismatch itself is a strong signal
        max_drift = max(max_drift, float(np.abs(sa - ma).max()))
    n = min(len(server["arrays"]), len(model["arrays"]))
    shape_match = all(s.shape == m.shape for s, m in zip(server["arrays"], model["arrays"]))
    return {
        "max_abs_logprob_drift": round(max_drift, 6),
        "n_sequences": n,
        "shape_match": shape_match,
        "verdict": "CONSISTENT" if (shape_match and max_drift < 1e-2) else "STALE_RUNTIME_SUSPECTED",
    }
