"""P1-1 runtime attestation: server-observed vs trainer-computed logprob
fingerprints (dual-source, HTTP-level)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from grpo_guard.adapters.runtime_attest import (
    _digest,
    drift,
    model_logprob_fingerprint,
    server_logprob_fingerprint,
)


class _FakeServe(BaseHTTPRequestHandler):
    """Minimal /get_sequence_logprobs stand-in returning canned logprobs."""

    logprobs: list = []

    def do_POST(self):
        if not self.path.startswith("/get_sequence_logprobs"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"logprobs": _FakeServe.logprobs}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeServe)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def test_server_fingerprint_parses_and_digests(fake_server):
    _FakeServe.logprobs = [[[0.1, 0.05, -0.2], [-0.3, -0.4, -0.5]]]
    fp = server_logprob_fingerprint("127.0.0.1", fake_server, [[1, 2, 3, 4]], [2])
    assert fp["n_sequences"] == 1
    assert fp["n_positions"] == [2]
    assert len(fp["digest"]) == 64
    assert _digest(fp["arrays"]) == fp["digest"]


def test_drift_verdicts():
    same = {"arrays": [np.full((2, 3), -0.5), np.full((2, 3), -0.5)]}
    assert drift(same, same)["verdict"] == "CONSISTENT"
    diff = {"arrays": [np.full((2, 3), -0.5), np.full((2, 3), -0.5)]}
    stale = {"arrays": [np.full((2, 3), -1.8), np.full((2, 3), -0.5)]}
    verdict = drift(diff, stale)
    assert verdict["verdict"] == "STALE_RUNTIME_SUSPECTED"
    assert verdict["max_abs_logprob_drift"] > 1e-2


def test_model_fingerprint_runs_on_tiny_model():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, ids):
            return (self.head(self.embed(ids)),)

    model = Tiny()
    fp = model_logprob_fingerprint(model, None, [[1, 2, 3, 4]], [2], top_logprobs=8)
    assert fp["n_sequences"] == 1
    assert fp["arrays"][0].shape == (2, 8)
    # same weights -> stable fingerprint
    fp2 = model_logprob_fingerprint(model, None, [[1, 2, 3, 4]], [2], top_logprobs=8)
    assert _digest(fp["arrays"]) == _digest(fp2["arrays"])
