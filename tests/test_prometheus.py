"""Prometheus-format metrics tests."""

from __future__ import annotations

import json
from pathlib import Path

from grpo_guard.prometheus import render_metrics, serve


def _mk_events(tmp_path: Path) -> Path:
    d = tmp_path / "events"
    d.mkdir()
    events = [
        {"event_id": "gen-1", "event_type": "generation_finished"},
        {"event_id": "vdec-1", "event_type": "validation_decision",
         "decision_payload": {"decision": "reject", "reason_codes": ["P004_STALE_POLICY_STRICT"]}},
        {"event_id": "vdec-2", "event_type": "validation_decision",
         "decision_payload": {"decision": "allow", "reason_codes": []}},
        {"event_id": "can-1", "event_type": "canary_passed", "drift": {"max_token_drift": 2}},
        {"event_id": "ts-1", "event_type": "training_step_finished", "success_rate": 0.66},
    ]
    for e in events:
        (d / f"{e['event_id']}.json").write_text(json.dumps(e), encoding="utf-8")
    return d


def test_render_metrics_counts(tmp_path: Path):
    text = render_metrics(_mk_events(tmp_path))
    assert 'grpo_guard_events_total{event_type="generation_finished"} 1' in text
    assert 'grpo_guard_decisions_total{decision="reject"} 1' in text
    assert 'grpo_guard_reason_codes_total{code="P004_STALE_POLICY_STRICT"} 1' in text
    assert "grpo_guard_canary_drift 2.0" in text
    assert "grpo_guard_training_steps_total 1" in text
    assert "grpo_guard_training_success_rate_last 0.66" in text


def test_serve_endpoint(tmp_path: Path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    events = _mk_events(tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = render_metrics(events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/metrics") as resp:
            body = resp.read().decode()
        assert "grpo_guard_decisions_total" in body
    finally:
        server.shutdown()
