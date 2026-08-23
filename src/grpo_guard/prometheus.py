"""Prometheus-format metrics for the guard (production observability).

``grpo-guard metrics --dir <events>`` scans the event log once and emits
Prometheus text format (zero dependencies, standard http.server for the
``--serve`` mode).  A Prometheus scraper can point at the endpoint; the
metrics are derived from the append-only event stream, so they cannot
lie about history.

Metrics:
- grpo_guard_events_total{event_type}       events by type
- grpo_guard_decisions_total{decision}      validation decisions
- grpo_guard_reason_codes_total{code}       reason-code occurrences
- grpo_guard_canary_drift                   last canary drift (gauge)
- grpo_guard_training_steps_total           training_step_finished count
- grpo_guard_training_success_rate_last     last step success rate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def render_metrics(events_dir: Path) -> str:
    events: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    canary_drift: float | None = None
    training_steps = 0
    success_last: float | None = None

    for p in sorted(events_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        events[payload.get("event_type", "unknown")] += 1
        if payload.get("event_type") == "validation_decision":
            dp = payload.get("decision_payload") or {}
            decisions[dp.get("decision", "unknown")] += 1
            for c in dp.get("reason_codes", []):
                codes[c] += 1
        if payload.get("event_type") == "canary_passed":
            drift = (payload.get("drift") or {}).get("max_token_drift")
            if drift is not None:
                canary_drift = float(drift)
        if payload.get("event_type") == "training_step_finished":
            training_steps += 1
            success_last = float(payload.get("success_rate", 0.0))

    lines = ["# HELP grpo_guard_events_total Events by type in the append-only log.",
             "# TYPE grpo_guard_events_total counter"]
    for k, v in sorted(events.items()):
        lines.append(f'grpo_guard_events_total{{event_type="{k}"}} {v}')
    lines.append("# TYPE grpo_guard_decisions_total counter")
    for k, v in sorted(decisions.items()):
        lines.append(f'grpo_guard_decisions_total{{decision="{k}"}} {v}')
    lines.append("# TYPE grpo_guard_reason_codes_total counter")
    for k, v in sorted(codes.items()):
        lines.append(f'grpo_guard_reason_codes_total{{code="{k}"}} {v}')
    if canary_drift is not None:
        lines.append("# TYPE grpo_guard_canary_drift gauge")
        lines.append(f"grpo_guard_canary_drift {canary_drift}")
    lines.append("# TYPE grpo_guard_training_steps_total counter")
    lines.append(f"grpo_guard_training_steps_total {training_steps}")
    if success_last is not None:
        lines.append("# TYPE grpo_guard_training_success_rate_last gauge")
        lines.append(f"grpo_guard_training_success_rate_last {success_last}")
    return "\n".join(lines) + "\n"


def serve(events_dir: Path, port: int = 9100) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            body = render_metrics(events_dir).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"grpo-guard metrics on :{port}/metrics (dir={events_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="events dir")
    ap.add_argument("--serve", action="store_true", help="serve /metrics over HTTP")
    ap.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()
    events_dir = Path(args.dir)
    if not events_dir.is_dir():
        print(f"no events dir: {events_dir}")
        return 1
    if args.serve:
        serve(events_dir, args.port)
        return 0
    print(render_metrics(events_dir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
