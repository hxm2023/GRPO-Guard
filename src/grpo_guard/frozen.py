"""Frozen case writer (design doc §15.4).

Frozen cases are never overwritten by test code: writing an existing case
dir raises.  Updates create a new version directory (e.g. f1_f4_v02) and
keep the old results.  Each case contains case.json (spec), inputs/
(artifacts + events), expected_decision, and SHA256SUMS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from grpo_guard import testing
from grpo_guard.faults import (
    inject_f1_static_rollout,
    inject_f2_misbound_logprob,
    inject_f3_retokenization,
    inject_f3_retokenized_sequence,
    inject_f3_template_variant,
    inject_f4_mask_shift,
)
from grpo_guard.store.canonical_json import canonical_dumps


class FrozenCaseExists(FileExistsError):
    pass


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_case(
    root: Path,
    case_id: str,
    expected_decision: str,
    required_reason_codes: list[str],
    t: testing.Trajectory,
    notes: str = "",
) -> Path:
    """Serialize a trajectory as a frozen case with no-overwrite semantics."""
    case_dir = root / case_id
    if case_dir.exists():
        raise FrozenCaseExists(f"frozen case {case_id} already exists (no overwrite)")
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)

    for name, ev in sorted(t.events.items()):
        (inputs_dir / f"event_{name}.json").write_text(
            canonical_dumps(ev.model_dump(mode="json")).decode("utf-8"), encoding="utf-8"
        )
    (inputs_dir / "envelope.json").write_text(
        canonical_dumps(t.envelope.model_dump(mode="json")).decode("utf-8"), encoding="utf-8"
    )
    (inputs_dir / "policy_manifest.json").write_text(
        canonical_dumps(t.policy_manifest.model_dump(mode="json")).decode("utf-8"), encoding="utf-8"
    )
    (inputs_dir / "split_manifest.json").write_text(
        canonical_dumps(t.split_manifest.model_dump(mode="json")).decode("utf-8"), encoding="utf-8"
    )
    for blob in sorted(t.store.blobs.glob("*")):
        if blob.suffix == ".tmp":
            continue
        (inputs_dir / f"artifact_{blob.name}").write_bytes(blob.read_bytes())
    if getattr(t, "bogus_sequence_ref", None) is not None:
        from grpo_guard.schema.artifacts import ArtifactRef

        if isinstance(t.bogus_sequence_ref, ArtifactRef):
            (inputs_dir / "bogus_sequence_ref.json").write_text(
                canonical_dumps(t.bogus_sequence_ref.model_dump(mode="json")).decode("utf-8"), encoding="utf-8"
            )
    # v0.2: validation-time context (split registry, eval protocol) for F5/F6
    ctx_extra = {}
    if getattr(t, "split_registry", None):
        ctx_extra["split_registry"] = {
            name: sm.model_dump(mode="json") for name, sm in t.split_registry.items()
        }
    if getattr(t, "eval_protocol_sha256", None):
        ctx_extra["eval_protocol_sha256"] = t.eval_protocol_sha256
    if getattr(t, "reward_verifier_registry", None):
        ctx_extra["reward_verifier_registry"] = t.reward_verifier_registry
    if getattr(t, "requires_update_input", False):
        ctx_extra["requires_update_input"] = True
    if ctx_extra:
        (inputs_dir / "context.json").write_text(
            canonical_dumps(ctx_extra).decode("utf-8"), encoding="utf-8"
        )

    sha_lines = sorted(
        f"{_hash_file(p)}  {p.relative_to(case_dir).as_posix()}" for p in inputs_dir.iterdir()
    )
    (case_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    spec = {
        "case_id": case_id,
        "expected_decision": expected_decision,
        "required_reason_codes": required_reason_codes,
        "envelope_id": t.envelope.envelope_id,
        "run_id": t.run_id,
        "notes": notes,
        "generation_event_id": t.envelope.generation_event.event_id,
    }
    (case_dir / "case.json").write_text(
        canonical_dumps(spec).decode("utf-8"), encoding="utf-8"
    )
    return case_dir


FAULTS = {
    "f1_static_rollout": lambda t, v: inject_f1_static_rollout(t, v["runtime_version"], v["claimed_parent"]),
    "f2_misbound_logprob": lambda t, v: inject_f2_misbound_logprob(t, v["scorer_policy_version"]),
    "f3_retokenization": lambda t, v: (
        inject_f3_retokenized_sequence(t, "b" * 64) if v.get("kind") == "sequence"
        else (inject_f3_template_variant(t) if v.get("kind") == "template" else inject_f3_retokenization(t))
    ),
    "f4_mask_shift": lambda t, v: inject_f4_mask_shift(t, v["shift"]),
}
