"""Operational monitor: event search + guard alert webhooks.

Two operational surfaces for the evidence chain:

- ``event_search`` — query the append-only event log by type / component /
  reason code / prompt; returns canonical event payloads.
- ``scan_decisions`` — collect every non-ALLOW validation decision in an
  event dir (reject/quarantine), ready for alerting.
- ``send_webhook`` — POST a decision payload to an HTTP endpoint
  (Slack-compatible or any generic JSON receiver).  Failures are logged,
  never raised: alerting must not crash the training loop.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("grpo_guard.monitor")

DECISION_CODES = {"reject", "quarantine"}


def load_events(event_dir: str | Path) -> list[dict]:
    """Read every canonical event json in an append-log events dir."""
    out = []
    for p in sorted(Path(event_dir).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.warning("unreadable event file %s", p)
            continue
    return out


def event_search(
    event_dir: str | Path,
    event_type: str | None = None,
    component_id: str | None = None,
    reason_code: str | None = None,
    prompt_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Search the event log.  reason_code matches validation decisions;
    prompt_id matches generation events."""
    out = []
    for e in load_events(event_dir):
        if event_type and e.get("event_type") != event_type:
            continue
        if component_id and e.get("component_id") != component_id:
            continue
        if reason_code:
            codes = ((e.get("decision_payload") or {}).get("reason_codes") or [])
            if reason_code not in codes:
                continue
        if prompt_id and e.get("prompt_id") != prompt_id:
            continue
        out.append(e)
        if len(out) >= limit:
            break
    return out


def scan_decisions(event_dir: str | Path) -> list[dict]:
    """Every non-ALLOW validation decision in the event dir."""
    out = []
    for e in load_events(event_dir):
        if e.get("event_type") != "validation_decision":
            continue
        payload = e.get("decision_payload") or {}
        if payload.get("decision") in DECISION_CODES:
            out.append(e)
    return out


def send_webhook(payload: dict, webhook_url: str, timeout: float = 5.0) -> bool:
    """POST a payload to a webhook endpoint.  Returns True on 2xx."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as exc:
        log.warning("webhook failed: %s", exc)
        return False


def alert_non_allow(event_dir: str | Path, webhook_url: str) -> dict:
    """Scan + alert every non-ALLOW decision.  Returns a summary dict."""
    decisions = scan_decisions(event_dir)
    sent, failed = 0, 0
    for d in decisions:
        payload = {
            "text": f"[grpo-guard] {d['event_id']} "
                    f"{(d.get('decision_payload') or {}).get('decision')} "
                    f"{(d.get('decision_payload') or {}).get('reason_codes', [])[:3]}",
            "event_id": d["event_id"],
            "decision": (d.get("decision_payload") or {}).get("decision"),
            "reason_codes": (d.get("decision_payload") or {}).get("reason_codes", []),
            "run_id": d.get("run_id"),
            "created_at_utc": d.get("created_at_utc"),
            "sent_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if send_webhook(payload, webhook_url):
            sent += 1
        else:
            failed += 1
    return {"scanned": len(decisions), "sent": sent, "failed": failed}
