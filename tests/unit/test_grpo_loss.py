"""Guarded GRPO loss contract (design doc §7.3.3, §9.1)."""

import numpy as np
import pytest
import torch

from grpo_guard import testing
from grpo_guard.adapters.guarded_update import ValidatedBatchHandle, materialize
from grpo_guard.adapters.grpo_loss import grpo_loss
from grpo_guard.schema.artifacts import ArtifactRef, EnvelopeRef, EventRef


class TinyModel(torch.nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.config = type("C", (), {"vocab_size": vocab})()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return (self.head(self.embed(input_ids)),)


def _make_handle(run_id, seq, mask, logprobs, rewards, nonce):
    from grpo_guard.adapters.guarded_update import MaterializedBatch
    from grpo_guard.schema.events import UpdateInputEvent

    refs = []
    for name, arr in (("seq", seq), ("mask", mask), ("lp", logprobs)):
        refs.append(ArtifactRef(
            uri=f"artifact://{name}", media_type="application/octet-stream",
            dtype="int32" if name == "seq" else ("int8" if name == "mask" else "float32"),
            shape=list(arr.shape), num_bytes=arr.nbytes, sha256="0" * 64, producer_event_id="gen-x",
        ))
    seq_ref, mask_ref, lp_ref = refs
    ev = UpdateInputEvent(
        event_id=f"uinput-{nonce}", run_id=run_id, component_id="materializer",
        lifecycle_seq=1, created_at_utc=testing.now_utc(),
        update_id="update-1",
        preupdate_envelope=EnvelopeRef(uri="", envelope_id="env-x", envelope_sha256="0" * 64),
        preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
        sequence_token_ids=seq_ref, loss_mask=mask_ref,
        authoritative_behavior_logprob_event=EventRef(uri="", event_id="gen-x", event_sha256="0" * 64),
        authoritative_behavior_logprobs=lp_ref,
        reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64,
        single_use_nonce_sha256="0" * 64,
        tokenizer_called=False,
    ).seal()
    batch = MaterializedBatch(
        sequence_token_ids=seq, loss_mask=mask, behavior_logprobs=logprobs,
        rewards=rewards, layout_sha256="0" * 64,
    )
    return ValidatedBatchHandle(ev, batch)


def test_grpo_loss_runs_and_masks_padding():
    torch.manual_seed(0)
    model = TinyModel()
    # B=2, T_max=8: row0 has completion [P=3, T=7), row1 completion [P=2, T=6)
    seq = np.zeros((2, 8), dtype=np.int32)
    seq[0, :7] = [1, 2, 3, 4, 5, 6, 7]
    seq[1, :6] = [2, 3, 4, 5, 6, 7]
    mask = np.zeros((2, 7), dtype=np.int8)
    mask[0, 2:6] = 1  # positions [P-1, T-1) = [2, 6)
    mask[1, 1:5] = 1  # positions [1, 5)
    lp = np.full((2, 4), -0.5, dtype=np.float32)
    rewards = np.asarray([1.0, 0.0], dtype=np.float32)

    handle = _make_handle("run-x", seq, mask, lp, rewards, "n1")
    result = grpo_loss(model, handle, group_size=2)
    assert torch.isfinite(result.loss)
    assert result.metrics["B"] == 2 and result.metrics["T"] == 8
    assert 0.0 <= result.metrics["clip_fraction"] <= 1.0
    assert result.metrics["ratio_max"] > 0.0
    assert result.metrics["ratio_p50"] > 0.0


def test_grpo_loss_masked_positions_only():
    torch.manual_seed(1)
    model = TinyModel()
    seq = np.zeros((1, 6), dtype=np.int32)
    seq[0, :5] = [1, 2, 3, 4, 5]
    mask = np.zeros((1, 5), dtype=np.int8)
    mask[0, 2:4] = 1  # only two completion positions
    lp = np.full((1, 2), -0.5, dtype=np.float32)
    rewards = np.asarray([1.0], dtype=np.float32)

    handle = _make_handle("run-x", seq, mask, lp, rewards, "n2")
    result = grpo_loss(model, handle, group_size=1)
    # with ratio=1 everywhere and advantage=0 (single group), loss must be ~0
    assert abs(result.loss.item()) < 1e-4


def test_grpo_loss_rejects_text_path():
    with pytest.raises(TypeError):
        grpo_loss(TinyModel(), "prompt+completion text", group_size=1)
