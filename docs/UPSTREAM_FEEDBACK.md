# Upstream feedback log

GRPO-Guard's adapter patches and findings mapped against upstream TRL/vLLM
issues — honest contribution signal, no duplicate issue reports.

## 1. PyNcclCommunicator device assertion (server mode)

- **Our finding** (2026-08-22): TRL 1.10.0 server mode —
  `VLLMClient.init_communicator(device=accelerator.device)` passes an
  UNINDEXED `torch.device('cuda')` into vLLM's `PyNcclCommunicator`, whose
  warm-up `all_reduce` asserts `in_tensor.device == self.device` and fails
  (`'cuda' != 'cuda:0'`) before any rollout.
- **Our patch**: `examples/countdown/closed_loop.py` normalizes the device
  index before init (version-guarded, fail-closed; DECISION_LOG D2).
- **Upstream status**: huggingface/trl **#3774** "Fix pynccl communicator
  assertion error with VLLMClient" (merged) and **#3762** "Prevent NCCL
  Device Conflicts Between vLLM Server and Trainers" (merged) address this
  class of bug.  Our patch remains a documented workaround for the pinned
  matrix (trl 1.10.0 / vllm 0.26.0) until we verify the merged fixes are
  present in a release we pin.
- **Still-open verification**: whether the merged fixes cover the exact
  "unindexed `accelerator.device` passed by the trainer client" path in a
  released TRL.  If a future pinned TRL still reproduces it, we file an
  issue with a minimal repro.

## 2. torchcodec `.so` load failure (environment, not upstream bug)

- `torchcodec 0.16.0`'s `libtorchcodec_image.so` failed to load in the
  autodl2 venv (CUDA 12.8 / torch 2.11.0+cu130), blocking GRPOTrainer
  import.  Removed (no reverse deps).  Environment-specific; no upstream
  issue filed.

## 3. TRL `VLLMClient.generate` return shape

- The client returns a **dict** keyed `prompt_ids/completion_ids/logprobs/
  logprob_token_ids`; tuple-unpacking yields dict keys (the constant-sketch
  canary bug, fixed in our `canary.py`).  The signature documents the dict
  return, so this is a caller-side pitfall — we documented it in code and
  added a regression test rather than filing an upstream issue.

## Contribution posture

- No duplicate issues filed.  We map our findings to existing upstream
  fixes (#3774/#3762), keep workarounds version-guarded, and will file a
  new issue ONLY if the merged fix does not cover our scenario on a pinned
  future release (with a minimal repro from this repo).
