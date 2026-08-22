"""P008 canary-mismatch online verification on autodl2 (design doc §10).

1. vLLM server loads the REAL v0 weights; a greedy canary sketch is taken
   (the calibration baseline).
2. A deterministically perturbed checkpoint is saved (seed 7, sigma 0.05 —
   large enough to move greedy tokens in bf16) and the server reloads it.
3. The second sketch drifts beyond the frozen tolerance (0) → the canary
   reports "mismatch".
4. A real envelope is validated with canary_status="mismatch" → the
   validator must reject with P008_CANARY_MISMATCH (fail closed).

Output: <out>/canary_mismatch_online.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/canary_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8004"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51219"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
PERTURBED_DIR = Path(os.environ.get("GRPO_GUARD_PERTURBED_CKPT",
                                    "/root/autodl-tmp/grpo-guard/canary_ckpt_perturbed"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))


def log(msg: str) -> None:
    print(f"[canary-check] {msg}", flush=True)


def main() -> int:
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server
    from grpo_guard.canary import CanarySuite
    from grpo_guard.schema.artifacts import EventRef, ManifestRef
    from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
    from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
    from grpo_guard.store.canonical_json import canonical_sha256
    from grpo_guard.validators.context import ProtocolConfig, ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suite = CanarySuite()

    def generate_fn(client):
        return lambda prompts, **kw: (
            client.generate(prompts, n=1, temperature=0.0, top_p=1.0, top_k=1,
                            max_tokens=8, logprobs=0)
        )

    # ---- phase 1: baseline on the REAL v0 weights --------------------------
    server = start_server(OUT_DIR / "vllm_server_v0.log", port=VLLM_PORT)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)
        baseline = suite.sketch(generate_fn(client))
        log("baseline canary sketch taken")
    finally:
        stop_server(server)

    # ---- phase 2: deterministic perturbed checkpoint -----------------------
    log(f"building perturbed checkpoint at {PERTURBED_DIR} (seed=7, sigma=0.05)")
    shutil.rmtree(PERTURBED_DIR, ignore_errors=True)
    PERTURBED_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    torch.manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data).mul_(0.05))
    from safetensors.torch import save_file

    keys = list(model.state_dict().keys())
    shard_size = max(1, len(keys) // 4)
    for i in range(0, len(keys), shard_size):
        shard = {k: model.state_dict()[k].float().contiguous() for k in keys[i:i + shard_size]}
        save_file(shard, PERTURBED_DIR / f"model-{i // shard_size + 1:05d}-of-{len(keys) // shard_size + 1:05d}.safetensors")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        src_file = Path(MODEL_PATH) / name
        if src_file.exists():
            shutil.copy(src_file, PERTURBED_DIR / name)
    del model
    torch.cuda.empty_cache()
    log("perturbed checkpoint saved")

    # ---- phase 3: canary on the perturbed weights --------------------------
    server2 = start_server(OUT_DIR / "vllm_server_perturbed.log", port=VLLM_PORT)
    try:
        client2 = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                             connection_timeout=300)
        check = suite.check(generate_fn(client2), policy_version=0,
                            baseline=baseline, tolerance=0)
        log(f"canary verdict: {check.verdict}, drift={check.drift}")
    finally:
        stop_server(server2)

    # ---- phase 4: validator must reject with P008 --------------------------
    from grpo_guard import testing

    t = testing.build_trajectory()
    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=ProtocolConfig(name="strict_v01", mode="strict_on_policy"),
        canary_status="mismatch" if check.verdict == "mismatch" else "pass",
    )
    d = validate_envelope(ctx, "identity_pre_reward").decision_payload
    p008 = "P008_CANARY_MISMATCH" in d.reason_codes

    result = {
        "run_id": f"canary-{int(time.time())}",
        "baseline_sketches": baseline,
        "perturbed_verdict": check.verdict,
        "perturbed_drift": check.drift,
        "validator_decision": d.decision,
        "validator_reason_codes": d.reason_codes[:4],
        "p008_fired": p008,
        "expectation": "canary mismatch -> validator reject P008",
        "matched": p008 and check.verdict == "mismatch",
    }
    (OUT_DIR / "canary_mismatch_online.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"result: verdict={check.verdict} validator={d.decision} p008={p008}")
    log("CANARY CHECK DONE")
    return 0 if result["matched"] else 1


if __name__ == "__main__":
    sys.exit(main())
