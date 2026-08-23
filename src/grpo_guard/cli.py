"""grpo-guard CLI (design doc §14.1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _cmd_contract_check(args) -> int:
    from grpo_guard.contract_check import run_contract_check

    manifest = run_contract_check(Path(args.cases), args.out)
    return 0 if manifest["summary"]["passed"] == manifest["summary"]["total"] else 1


def _cmd_freeze_cases(args) -> int:
    from grpo_guard.frozen import write_case
    from grpo_guard import testing

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    from grpo_guard.day3 import INJECTORS

    for case in cfg["cases"]:
        injector = INJECTORS[case["fault"]]
        for variant in case["variants"]:
            t = testing.build_trajectory(policy_version=0)
            ft = injector(t, variant)
            expected = variant.get("expected_decision", case["expected_decision"])
            required = variant.get("required_reason_codes", case["required_reason_codes"])
            path = write_case(
                root, f"{case['id']}_{variant['name']}",
                expected, required, ft,
                notes=json.dumps(variant),
            )
            written.append(str(path))
    for i in range(cfg["normal_cases"]["count"]):
        t = testing.build_trajectory(policy_version=0, run_id=f"run-normal-{i}")
        path = write_case(root, f"normal_{i}", "allow", [], t, notes="normal neighbor")
        written.append(str(path))
    for spec in cfg.get("boundary_cases", []):
        t = testing.build_trajectory(policy_version=0, run_id=f"run-{spec['id']}", **spec.get("kwargs", {}))
        path = write_case(root, spec["id"], spec["expected_decision"], spec.get("required_reason_codes", []), t,
                          notes="boundary")
        written.append(str(path))
    print(f"froze {len(written)} cases under {root}")
    return 0


def _cmd_fault_matrix(args) -> int:
    from grpo_guard.matrix import run_fault_matrix

    matrix = run_fault_matrix(
        Path(args.config), Path(args.out), Path(args.freeze) if args.freeze else None, args.guard_mode
    )
    s = matrix["summary"]
    print(f"fault matrix: {s['passed']}/{s['total']} matched; "
          f"canonical {s['canonical_faults']}/{s['canonical_total']}; "
          f"normal allow {s['normal_allow_count']}/{s['normal_total']}")
    return 0 if s["passed"] == s["total"] else 1


def _cmd_smoke(args) -> int:
    from grpo_guard.smoke import run_smoke

    manifest = run_smoke(Path(args.config), args.out, server=args.server)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest.get("official_smoke_passed") else 1


def _cmd_replay(args) -> int:
    # prefer the committed real-model replay; fall back to the CPU surrogate
    committed = Path(args.manifest).parent / "replay" / "gradient_replay.json"
    if committed.exists():
        payload = json.loads(committed.read_text(encoding="utf-8"))
        for p in payload["pairs"]:
            print(f"{p['fault_kind']:16s} cos={p['gradient_cosine']} rL2={p['relative_l2']:.4f} "
                  f"norm_c={p['control_update_norm']:.3e} norm_f={p['fault_update_norm']:.3e}")
        print(f"source: {committed}")
        return 0
    from grpo_guard.replay.gradient_probe import run_replay

    manifest = run_replay(Path(args.manifest))
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_day3(args) -> int:
    from grpo_guard.day3 import run_day3_matrix

    matrix = run_day3_matrix(Path(args.loop_dir), Path(args.config), Path(args.out))
    s = matrix["summary"]
    print(f"day3 matrix: canonical {s['canonical_matched']}/{s['canonical_total']}; "
          f"normal allow {s['normal_allow']}/{s['normal_total']} "
          f"(q={s['normal_quarantine']}, r={s['normal_reject']}); "
          f"boundary {s['boundary_matched']}/{s['boundary_total']}")
    print(f"GATE: {'PASS' if s['gate_pass'] else 'FAIL'}")
    return 0 if s["gate_pass"] else 1


def _cmd_v02(args) -> int:
    from grpo_guard.matrix_v02 import run_v02_matrix

    matrix = run_v02_matrix(Path(args.loop_dir), Path(args.config), Path(args.out))
    s = matrix["summary"]
    print(f"v0.2 matrix: {s['matched']}/{s['total']} matched; normal allow {s['normal_allow']}/{s['normal_total']}")
    print(f"GATE: {'PASS' if s['gate_pass'] else 'FAIL'}")
    return 0 if s["gate_pass"] else 1


def _cmd_events(args) -> int:
    from grpo_guard.monitor import event_search

    hits = event_search(args.dir, event_type=args.type, component_id=args.component,
                        reason_code=args.code, prompt_id=args.prompt)
    for e in hits:
        print(f"{e.get('event_type')} {e.get('event_id')} {e.get('component_id')}")
    print(f"{len(hits)} events")
    return 0


def _cmd_alert_scan(args) -> int:
    from grpo_guard.monitor import alert_non_allow

    summary = alert_non_allow(args.dir, args.webhook)
    print(f"scanned={summary['scanned']} sent={summary['sent']} failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


def _cmd_doctor(args) -> int:
    from grpo_guard.doctor import check_checkpoint, run_doctor

    report = run_doctor(Path(args.profile))
    if args.checkpoint:
        failures, warnings = check_checkpoint(Path(args.checkpoint))
        for f in failures:
            report.fail(f)
        for w in warnings:
            report.warn(w)
        if not failures:
            report.ok(f"checkpoint {args.checkpoint} verified (no corrupted shards)")
    for line in report.findings:
        print(f"[doctor] {line}")
    print(f"[doctor] {'OK' if not report.failures else f'{len(report.failures)} failures'}")
    return 0 if not report.failures else 1


def _cmd_metrics(args) -> int:
    from grpo_guard.prometheus import render_metrics, serve

    events_dir = Path(args.dir)
    if not events_dir.is_dir():
        print(f"no events dir: {events_dir}")
        return 1
    if args.serve:
        serve(events_dir, args.port)
        return 0
    print(render_metrics(events_dir), end="")
    return 0


def _cmd_resume(args) -> int:
    from grpo_guard.resume import write_recovery_plan

    plan = write_recovery_plan(Path(args.events), Path(args.out))
    print(f"last completed step: {plan['last_step']} (next: {plan['next_step']})")
    print(f"checkpoint: {plan['checkpoint_dir']} sha={plan['checkpoint_manifest_sha256']}")
    print(f"recovery plan -> {args.out}")
    return 0 if plan["ok"] else 1


def _cmd_verify(args) -> int:
    from grpo_guard.verify import verify_artifact_dir

    report = verify_artifact_dir(Path(args.artifact_dir), Path(args.events) if args.events else None)
    for c in report.checks:
        print(f"[verify] {c}")
    for f in report.failures:
        print(f"[verify] FAIL: {f}")
    print(f"[verify] {'OK' if report.ok else f'{len(report.failures)} failures'}")
    return 0 if report.ok else 1


def _cmd_report(args) -> int:
    from grpo_guard.report import build_report

    report = build_report(Path(args.artifact_dir), commit=args.commit, update=args.update)
    print(f"wrote {report['report_md']} + SHA256SUMS")
    return 0


def _cmd_profile(args) -> int:
    from grpo_guard.profile import freeze_compatibility_profile

    path = freeze_compatibility_profile(Path(args.out), server=args.server)
    print(f"compatibility profile frozen at {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grpo-guard", description="Trajectory contract, lineage and fault-injection framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("contract-check", help="run frozen cases through the validator")
    p.add_argument("--cases", required=True, help="directory of frozen cases")
    p.add_argument("--out", default="artifacts/v0.1.0", help="output dir")
    p.set_defaults(fn=_cmd_contract_check)

    p = sub.add_parser("freeze-cases", help="generate frozen F1-F4 cases (no-overwrite)")
    p.add_argument("--config", required=True, help="faults config yaml")
    p.add_argument("--out", required=True, help="output dir for frozen cases")
    p.set_defaults(fn=_cmd_freeze_cases)

    p = sub.add_parser("fault-matrix", help="run the F1-F4 reason-coded matrix")
    p.add_argument("--config", required=True, help="faults config yaml")
    p.add_argument("--guard-mode", default="strict_on_policy", choices=["strict_on_policy", "bounded_off_policy"])
    p.add_argument("--out", default="artifacts/v0.1.0", help="output dir")
    p.add_argument("--freeze", default=None, help="also write frozen cases here")
    p.set_defaults(fn=_cmd_fault_matrix)

    p = sub.add_parser("smoke", help="official TRL+vLLM server-mode smoke (Compatibility Gate)")
    p.add_argument("--config", required=True, help="workload config yaml")
    p.add_argument("--out", default="artifacts/v0.1.0", help="output dir")
    p.add_argument("--server", default="autodl2", help="ssh target for GPU runs")
    p.set_defaults(fn=_cmd_smoke)

    p = sub.add_parser("replay", help="deterministic paired gradient replay (Day 4)")
    p.add_argument("--manifest", required=True, help="run manifest json")
    p.set_defaults(fn=_cmd_replay)

    p = sub.add_parser("day3-matrix", help="F1-F4 matrix over real loop artifacts (Correctness Gate)")
    p.add_argument("--loop-dir", required=True, help="Day 2 loop evidence dir")
    p.add_argument("--config", default="configs/faults/f1_f4_v01.yaml")
    p.add_argument("--out", default="artifacts/v0.1.0")
    p.set_defaults(fn=_cmd_day3)

    p = sub.add_parser("v02-matrix", help="v0.2 F5-F8 matrix over real loop artifacts")
    p.add_argument("--loop-dir", required=True)
    p.add_argument("--config", default="configs/faults/f5_f8_v02.yaml")
    p.add_argument("--out", default="artifacts/v0.2.0-dev")
    p.set_defaults(fn=_cmd_v02)

    p = sub.add_parser("events", help="search the append-only event log")
    p.add_argument("--dir", required=True, help="events dir (canonical json files)")
    p.add_argument("--type", default=None, help="event_type filter")
    p.add_argument("--component", default=None, help="component_id filter")
    p.add_argument("--code", default=None, help="reason code filter (validation decisions)")
    p.add_argument("--prompt", default=None, help="prompt_id filter (generation events)")
    p.set_defaults(fn=_cmd_events)

    p = sub.add_parser("alert-scan", help="scan non-ALLOW decisions and POST to a webhook")
    p.add_argument("--dir", required=True, help="events dir")
    p.add_argument("--webhook", required=True, help="webhook URL (generic JSON or Slack-compatible)")
    p.set_defaults(fn=_cmd_alert_scan)

    p = sub.add_parser("doctor", help="environment self-check vs compatibility profile")
    p.add_argument("--profile", default="compatibility_profile.yaml", help="compatibility profile yaml")
    p.add_argument("--checkpoint", default=None, help="checkpoint dir to verify (PolicyManifest weights)")
    p.set_defaults(fn=_cmd_doctor)

    p = sub.add_parser("metrics", help="Prometheus-format guard metrics (scan or --serve /metrics)")
    p.add_argument("--dir", required=True, help="events dir")
    p.add_argument("--serve", action="store_true", help="serve over HTTP")
    p.add_argument("--port", type=int, default=9100)
    p.set_defaults(fn=_cmd_metrics)

    p = sub.add_parser("resume", help="training recovery plan from the event log")
    p.add_argument("--events", required=True, help="events dir")
    p.add_argument("--out", default="recovery_plan.json", help="output recovery plan json")
    p.set_defaults(fn=_cmd_resume)

    p = sub.add_parser("verify", help="verify the evidence chain (checksums + event seals/order/refs)")
    p.add_argument("--artifact-dir", required=True, help="artifact dir with SHA256SUMS")
    p.add_argument("--events", default=None, help="events dir for seal/order/ref checks")
    p.set_defaults(fn=_cmd_verify)

    p = sub.add_parser("report", help="build run_manifest.json + REPORT.md + SHA256SUMS")
    p.add_argument("--artifact-dir", default="artifacts/v0.1.0")
    p.add_argument("--commit", default="")
    p.add_argument("--update", action="store_true", help="merge manifest entry + regenerate SHA256SUMS")
    p.set_defaults(fn=_cmd_report)

    p = sub.add_parser("profile", help="freeze the compatibility profile (design doc §4.1.1)")
    p.add_argument("--out", default="compatibility_profile.yaml")
    p.add_argument("--server", default="autodl2")
    p.set_defaults(fn=_cmd_profile)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
