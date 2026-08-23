"""Operational monitor: event search + alert webhook tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from grpo_guard.monitor import alert_non_allow, event_search, scan_decisions, send_webhook


@pytest.fixture()
def event_dir(tmp_path: Path) -> Path:
    events = [
        {
            "event_id": "gen-1", "event_type": "generation_finished", "component_id": "rollout",
            "prompt_id": "countdown-0001", "created_at_utc": "t",
        },
        {
            "event_id": "vdec-reject-1", "event_type": "validation_decision",
            "component_id": "validator", "run_id": "run-x", "created_at_utc": "t",
            "decision_payload": {"decision": "reject", "reason_codes": ["P004_STALE_POLICY_STRICT"]},
        },
        {
            "event_id": "vdec-allow-1", "event_type": "validation_decision",
            "component_id": "validator", "run_id": "run-x", "created_at_utc": "t",
            "decision_payload": {"decision": "allow", "reason_codes": []},
        },
        {
            "event_id": "vdec-quar-1", "event_type": "validation_decision",
            "component_id": "validator", "run_id": "run-x", "created_at_utc": "t",
            "decision_payload": {"decision": "quarantine", "reason_codes": ["R006_EVALUATOR_ALIAS"]},
        },
    ]
    d = tmp_path / "events"
    d.mkdir()
    for e in events:
        (d / f"{e['event_id']}.json").write_text(json.dumps(e), encoding="utf-8")
    return d


def test_event_search_by_type_and_code(event_dir: Path):
    gen = event_search(event_dir, event_type="generation_finished")
    assert len(gen) == 1 and gen[0]["prompt_id"] == "countdown-0001"
    p004 = event_search(event_dir, event_type="validation_decision", reason_code="P004_STALE_POLICY_STRICT")
    assert len(p004) == 1 and p004[0]["event_id"] == "vdec-reject-1"


def test_scan_decisions_finds_non_allow(event_dir: Path):
    hits = scan_decisions(event_dir)
    ids = {h["event_id"] for h in hits}
    assert ids == {"vdec-reject-1", "vdec-quar-1"}


class _WebhookHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _WebhookHandler.received.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_alert_non_allow_posts_webhook(event_dir: Path):
    server = HTTPServer(("127.0.0.1", 0), _WebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/hook"
        summary = alert_non_allow(event_dir, url)
        assert summary == {"scanned": 2, "sent": 2, "failed": 0}
        decisions = {r["decision"] for r in _WebhookHandler.received}
        assert decisions == {"reject", "quarantine"}
    finally:
        server.shutdown()


def test_send_webhook_failure_returns_false(event_dir: Path):
    assert send_webhook({"text": "x"}, "http://127.0.0.1:1/nope") is False
