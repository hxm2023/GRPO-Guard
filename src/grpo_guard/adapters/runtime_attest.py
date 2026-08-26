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
        rows = [[float(x) if x is not None else 0.0 for x in (pos or [])]
                for pos in seq_lp]
        k = max((len(r) for r in rows), default=0)
        rect = np.zeros((len(rows), k), dtype=np.float64)
        for j, r in enumerate(rows):
            rect[j, :len(r)] = r
        arrays.append(rect)
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
    """Max abs top-1 logprob difference between the fingerprints + verdict.

    Compares the top-1 (argmax) logprob per completion position — the
    strongest weight-sensitive signal, robust to top-k width differences
    between the server's ragged response and the model's uniform topk.
    Verdict threshold: 0.15 — measured same-weights noise (fp16 server vs
    bf16 trainer) is ~0.06-0.08; a stale/partially-updated runtime shows
    materially larger drift on at least one canary position.  The E1/E2/F1
    runners additionally use the RELATIVE verdict (late drift vs the
    same-weights baseline) for the final stale_detected flag.
    """
    max_drift = 0.0
    for sa, ma in zip(server["arrays"], model["arrays"]):
        n = min(sa.shape[0], ma.shape[0])
        if n == 0:
            continue
        max_drift = max(max_drift, float(np.abs(sa[:n, 0] - ma[:n, 0]).max()))
    return {
        "max_abs_logprob_drift": round(max_drift, 6),
        "n_sequences": min(len(server["arrays"]), len(model["arrays"])),
        "verdict": "CONSISTENT" if max_drift < 0.15 else "STALE_RUNTIME_SUSPECTED",
    }
