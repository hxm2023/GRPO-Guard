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
