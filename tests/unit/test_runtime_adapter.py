"""Runtime adapter contract (design doc §6.1, §7.4)."""

from grpo_guard import testing
from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
from grpo_guard.schema.artifacts import EventRef
from grpo_guard.store.append_log import AppendLog


def _adapter(tmp_path):
    from grpo_guard.store.artifact_store import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    log = AppendLog(tmp_path / "log", run_id="run-x", lease_id="rt")
    epoch = log.acquire_lease()
    rt = VLLMRuntimeAdapter(store, log, "run-x", "rollout-gpu1",
                            seq_provider=lambda: max([e["lifecycle_seq"] for e in log.iterate()] + [-1]) + 1)
    sync = EventRef(uri="", event_id="sync-x", event_sha256="0" * 64)
    return store, log, rt, epoch, sync


def test_emit_generation_shapes_and_seqs(tmp_path):
    store, log, rt, epoch, sync = _adapter(tmp_path)
    gen1 = rt.emit_generation(
        prompt_ids=[1, 2, 3], completion_ids=[4, 5, 6, 7],
        service_logprobs=[-0.1, -0.2, -0.3, -0.4],
        behavior_policy_version=0, checkpoint_manifest_sha256="c" * 64,
        sync_event=sync, tokenizer_sha256="t" * 64, chat_template_sha256="m" * 64,
        sampling_config_sha256="s" * 64, prompt_id="p1", request_id="r1",
        required_epoch=epoch,
    )
    gen2 = rt.emit_generation(
        prompt_ids=[1, 2], completion_ids=[5, 6],
        service_logprobs=[-0.5, -0.6],
        behavior_policy_version=0, checkpoint_manifest_sha256="c" * 64,
        sync_event=sync, tokenizer_sha256="t" * 64, chat_template_sha256="m" * 64,
        sampling_config_sha256="s" * 64, prompt_id="p1", request_id="r2",
        required_epoch=epoch,
    )
    assert gen1.event_id != gen2.event_id
    assert gen1.lifecycle_seq < gen2.lifecycle_seq
    assert gen1.completion_span == [3, 7]
    assert gen1.prompt_span == [0, 3]
    # authoritative sequence artifact matches the server token ids
    seq = store.get(gen1.sequence_token_ids)
    import numpy as np

    assert np.frombuffer(seq, dtype=np.int32).tolist() == [1, 2, 3, 4, 5, 6, 7]
    # service logprobs artifact length == completion length
    lp = store.get(gen1.service_behavior_logprobs)
    assert np.frombuffer(lp, dtype=np.float32).shape[0] == 4


def test_emit_rejects_logprob_length_mismatch(tmp_path):
    store, log, rt, epoch, sync = _adapter(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        rt.emit_generation(
            prompt_ids=[1, 2, 3], completion_ids=[4, 5, 6, 7],
            service_logprobs=[-0.1, -0.2],  # wrong length
            behavior_policy_version=0, checkpoint_manifest_sha256="c" * 64,
            sync_event=sync, tokenizer_sha256="t" * 64, chat_template_sha256="m" * 64,
            sampling_config_sha256="s" * 64, prompt_id="p1", request_id="r3",
            required_epoch=epoch,
        )


def test_emit_events_are_sealed_and_typed(tmp_path):
    store, log, rt, epoch, sync = _adapter(tmp_path)
    gen = rt.emit_generation(
        prompt_ids=[1, 2, 3], completion_ids=[4, 5, 6, 7],
        service_logprobs=[-0.1, -0.2, -0.3, -0.4],
        behavior_policy_version=0, checkpoint_manifest_sha256="c" * 64,
        sync_event=sync, tokenizer_sha256="t" * 64, chat_template_sha256="m" * 64,
        sampling_config_sha256="s" * 64, prompt_id="p1", request_id="r4",
        required_epoch=epoch,
    )
    from grpo_guard.schema.events import GenerationEvent, event_from_payload

    assert gen.verify_seal()
    rehydrated = event_from_payload(log.get(gen.event_id))
    assert isinstance(rehydrated, GenerationEvent)
    assert rehydrated.behavior_policy_version == 0
