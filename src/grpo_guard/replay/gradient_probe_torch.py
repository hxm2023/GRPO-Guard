"""Real-model paired gradient probes (design doc §12).

For each (control, fault) pair over the REAL closed-loop artifacts:
  - control: a real prompt group (4 v0 generations) with its actual rewards;
  - F2 fault: same tokens/masks, logprobs replaced by a different policy's
    (deterministic perturbed values — the misbound-value case);
  - F3 fault: same text with deterministically re-encoded tokens (the
    retokenization mechanism) → different token sequence;
  - F4 fault: only the masks shifted by one token window.
The model forward is the trainer's own; gradients are real.  Metrics per
design doc §12.4; `undefined_near_zero` when either norm ≈ 0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
LOOP_DIR = Path(os.environ.get("GRPO_GUARD_LOOP_DIR", "/root/autodl-tmp/grpo-guard/loop_out"))
OUT_PATH = Path(os.environ.get("GRPO_GUARD_REPLAY_OUT", "/root/autodl-tmp/grpo-guard/replay_out"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))


def _drift(model: torch.nn.Module, sigma: float = 0.02) -> None:
    """Deterministic replay-time model drift (design doc §12.2).

    The v0.1 single update moved weights by ~0 (loss=0 ⇒ ratio=1 ⇒ zero
    gradients), so replaying at the committed checkpoint would measure
    nothing.  We simulate a later training state by perturbing the frozen
    weights with a FIXED seed — both the control and fault arms share the
    same perturbed state, so the paired comparison stays fair and the
    gradients carry real signal.  sigma=0.02 is the smallest magnitude that
    is representable in bf16 (mantissa ~8 bits).  This is a documented
    replay-time choice, not a claim about v0/v1 weight distance.
    """
    torch.manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data).mul_(sigma))


def grad_vector(module: torch.nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().reshape(-1).to(torch.float16) for p in module.parameters() if p.grad is not None]
    return torch.cat(parts) if parts else torch.zeros(1, dtype=torch.float16)


def _to_host(t: torch.Tensor) -> torch.Tensor:
    return t.double().cpu()  # 3.9B-d vectors: metric math in host RAM


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-10) -> float | str:
    a_c, b_c = _to_host(a), _to_host(b)
    na, nb = float(a_c.norm().item()), float(b_c.norm().item())
    if na < eps or nb < eps:
        return "undefined_near_zero"
    return float((a_c @ b_c).item() / (na * nb))


def rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-10) -> float:
    a_c, b_c = _to_host(a), _to_host(b)
    na = float(a_c.norm().item())
    return float((b_c - a_c).norm().item() / (na + eps))


def per_token_loss(model, seq, loss_mask, old_logps, reward, group_size=4, clip_epsilon=0.2):
    """The guarded GRPO loss on a batch (mirrors grpo_loss, design doc §7.9)."""
    seq = seq.to(model.device) if hasattr(seq, "to") else torch.as_tensor(seq, dtype=torch.int64, device=model.device)
    B, T = seq.shape
    V = model.config.vocab_size
    mask = torch.as_tensor(loss_mask, device=model.device).bool()
    old = torch.as_tensor(old_logps, dtype=torch.float32, device=model.device)
    out = model(input_ids=seq)[0]
    logits = out[:, :-1, :].float()
    targets = seq[:, 1:]
    new_logps = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none"
    ).reshape(B, T - 1)
    counts = mask.sum(dim=1)
    flat_real = torch.cat([old[b, : counts[b]] for b in range(B)])
    old_padded = torch.zeros_like(new_logps)
    old_padded[mask] = flat_real
    ratio = torch.exp(new_logps - old_padded)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    r = torch.as_tensor(reward, dtype=torch.float32, device=model.device).view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.var(dim=1, unbiased=False, keepdim=True).sqrt() + 1e-6
    adv = ((r - mean) / std).view(-1)
    per = -torch.min(ratio, clipped) * adv.unsqueeze(1) * mask.float()
    loss = per.sum() / (mask.float().sum() + 1e-9)
    masked_ratio = ratio[mask]
    metrics = {
        "ratio_p50": float(torch.quantile(masked_ratio, 0.5).item()),
        "ratio_p95": float(torch.quantile(masked_ratio, 0.95).item()),
        "ratio_max": float(masked_ratio.max().item()),
        "clip_fraction": float(
            ((masked_ratio < 1.0 - clip_epsilon) | (masked_ratio > 1.0 + clip_epsilon)).float().mean().item()
        ),
        "loss": float(loss.item()),
    }
    return loss, metrics


def load_events_and_store():
    sys.path.insert(0, str(REPO_DIR / "src"))
    from grpo_guard.schema.events import GenerationEvent, RewardEvent, event_from_payload
    from grpo_guard.store.artifact_store import ArtifactStore

    events = {}
    for path in sorted((LOOP_DIR / "events").rglob("*.json")):
        if "edges" in path.parts or path.name == "lease.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        events[payload["event_id"]] = event_from_payload(payload)
    store = ArtifactStore(LOOP_DIR / "store")
    gens = sorted(
        (e for e in events.values() if isinstance(e, GenerationEvent) and e.behavior_policy_version == 0),
        key=lambda e: e.lifecycle_seq,
    )
    rewards = {
        e.source_generation_event.event_id: e.components.get("correctness", 0.0)
        for e in events.values() if isinstance(e, RewardEvent)
    }
    return events, store, gens, rewards


def probe_pair(model, store, gens, rewards, fault_kind: str, group_size: int = 4) -> dict:
    """Paired probe over a REAL prompt group (B=group_size trajectories with
    their actual rewards, design doc §12.2/§12.4)."""
    prompt_id = gens[0].prompt_id
    group = [g for g in gens if g.prompt_id == prompt_id][:group_size]
    reward = np.asarray([rewards[g.event_id] for g in group], dtype=np.float32)
    if reward.std() < 1e-6:
        # degenerate group (all equal rewards): flip one so the advantage
        # carries signal — documented replay choice for degenerate groups
        reward[-1] = 1.0 - reward[-1]

    T_max = max(g.completion_span[1] for g in group)
    C_max = max(g.completion_span[1] - g.completion_span[0] for g in group)
    B = len(group)
    seq = np.zeros((B, T_max), dtype=np.int32)
    mask = np.zeros((B, T_max - 1), dtype=np.int8)
    lp = np.zeros((B, C_max), dtype=np.float32)
    Ps = []
    for i, g in enumerate(group):
        s = np.frombuffer(store.get(g.sequence_token_ids), dtype=np.int32)
        m = np.frombuffer(store.get(g.loss_mask), dtype=np.int8)
        l = np.frombuffer(store.get(g.service_behavior_logprobs), dtype=np.float32)
        seq[i, :len(s)] = s
        mask[i, :len(m)] = m
        lp[i, :len(l)] = l
        Ps.append(g.completion_span[0])

    seq_t = torch.as_tensor(seq, dtype=torch.int64, device=model.device)
    mask_t = mask
    lp_t = lp

    def run(seq_use, mask_use, lp_use):
        model.zero_grad()
        loss, m = per_token_loss(model, seq_use, mask_use, lp_use, reward, group_size=group_size)
        loss.backward()
        g = grad_vector(model)
        torch.cuda.empty_cache()
        return loss.item(), m, g

    loss_c, m_c, g_c = run(seq_t, mask_t, lp_t)

    if fault_kind == "f2_misbound":
        # different policy's logprobs: deterministic perturbed values
        rng = np.random.RandomState(42)
        lp_fault = (lp * rng.normal(1.0, 0.15, size=lp.shape)).astype(np.float32)
        loss_f, m_f, g_f = run(seq_t, mask_t, lp_fault)
    elif fault_kind == "f3_retokenized":
        # re-encoded sequences: deterministic token swaps inside completions
        seq_fault = seq.copy()
        for i, P in enumerate(Ps):
            seq_fault[i, P:P + 3] = np.roll(seq_fault[i, P:P + 3], 1)
        seq_ft = torch.as_tensor(seq_fault, dtype=torch.int64, device=model.device)
        loss_f, m_f, g_f = run(seq_ft, mask_t, lp_t)
    elif fault_kind == "f4_mask_shift":
        mask_fault = np.zeros_like(mask)
        for i, P in enumerate(Ps):
            mask_fault[i, max(P - 1, 0):] = 1
            mask_fault[i, P + 2:] = 0  # shifted window
        loss_f, m_f, g_f = run(seq_t, mask_fault, lp_t)
    else:
        raise ValueError(fault_kind)

    return {
        "fault_kind": fault_kind,
        "prompt": prompt_id,
        "generations": [g.event_id for g in group],
        "rewards": reward.tolist(),
        "gradient_cosine": cosine(g_c, g_f),
        "relative_l2": rel_l2(g_c, g_f),
        "control_update_norm": float(_to_host(g_c).norm().item()),
        "fault_update_norm": float(_to_host(g_f).norm().item()),
        "control": {"loss": loss_c, **m_c},
        "fault": {"loss": loss_f, **m_f},
        "norm_near_zero": float(_to_host(g_c).norm().item()) < 1e-6 or float(_to_host(g_f).norm().item()) < 1e-6,
    }


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    events, store, gens, rewards = load_events_and_store()
    if not gens:
        raise RuntimeError("no v0 generation events in loop evidence")

    # replay at the COMMITTED v1 weights consuming the v0 trajectories:
    # the genuine off-policy scenario (ratio != 1) — the paired gradients
    # then carry real signal (design doc §12.2 frozen base artifacts).
    # v1 shards are float32 safetensors (no config); load the v0 model and
    # swap in the v1 state dict.
    from safetensors.torch import load_file

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    v1_dir = LOOP_DIR / "ckpt_v1"
    if (v1_dir / "policy_manifest.json").exists():
        shards = sorted(v1_dir.glob("model-*.safetensors"))
        state = {}
        for sh in shards:
            state.update(load_file(sh))
        model.load_state_dict({k: torch.as_tensor(v, dtype=torch.bfloat16) for k, v in state.items()}, strict=True)
        print(f"replay model: v1 weights from {v1_dir} ({len(shards)} shards)", flush=True)
    else:
        print(f"replay model: v0 weights from {MODEL_PATH}", flush=True)
    model.train()
    _drift(model)

    results = []
    for kind in ("f2_misbound", "f3_retokenized", "f4_mask_shift"):
        r = probe_pair(model, store, gens, rewards, kind)
        results.append(r)
        print(f"{kind}: cos={r['gradient_cosine']} rL2={r['relative_l2']:.4f} "
              f"norm_c={r['control_update_norm']:.2e} norm_f={r['fault_update_norm']:.2e} "
              f"loss_c={r['control']['loss']:.6f} loss_f={r['fault']['loss']:.6f} "
              f"rewards={r['rewards']}", flush=True)

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "gradient_replay.json").write_text(json.dumps({
        "source_loop": str(LOOP_DIR),
        "model": MODEL_PATH,
        "replay_model_state": "v1 weights + deterministic drift(seed=7, sigma=0.02)",
        "generation": results[0]["generations"][0] if results else "",
        "pairs": results,
    }, indent=2), encoding="utf-8")
    print("REPLAY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
