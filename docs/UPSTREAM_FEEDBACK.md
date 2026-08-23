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
- **Verification (2026-08-23)**: checked huggingface/trl PR **#3774** diff
  directly — it changes exactly the failing call in `grpo_trainer.py`:
  `init_communicator(device=self.accelerator.device)` →
  `device=torch.cuda.current_device()`.  That is the same normalization our
  patch applies, so the merged fix covers our scenario as a strict
  replacement.  The pinned trl 1.10.0 still contains the unpatched call
  (`trl/generation/vllm_generation.py`), so our workaround stays for the
  pinned matrix and can be dropped when a release containing #3774 is
  pinned.  No new issue needed.

## 2. torchcodec `.so` load failure (environment, not upstream bug)

- `torchcodec 0.16.0`'s `libtorchcodec_image.so` failed to load in the
  autodl2 venv (CUDA 12.8 / torch 2.11.0+cu130), blocking GRPOTrainer
  import.  Removed (no reverse deps).  Environment-specific; no upstream
  issue filed.

## 3. fastapi/starlette version trap in `trl vllm-serve` (2026-08-23)

- `trl 1.10.0`'s `scripts/vllm_serve.py` calls `FastAPI(lifespan=...)` in a
  way that breaks with **starlette >= 1.0** (`TypeError: Router.__init__()
  got an unexpected keyword argument 'on_startup'`) — starlette 1.x
  removed that kwarg.  Root cause (2026-08-23): the venv carried fastapi
  0.110.3 (pins starlette<0.38) while vllm 0.26.0 declares
  `fastapi>=0.133,<0.137` + `starlette>=1.0.1` — a stale fastapi under a
  re-resolved vllm silently produces the broken pair.
- **Upstream PR (2026-08-23)**: opened huggingface/trl **#6873** — pin
  `fastapi>=0.133` in the `[vllm]` extra (trl main still declares an
  unconstrained `fastapi`), so resolvers in dirty venvs upgrade fastapi
  instead of leaving the broken combination.  Local fix meanwhile: upgrade
  fastapi to 0.136.3 (satisfies vllm's floor); `trl vllm-serve` verified
  working on fastapi 0.136.3 + starlette 1.6.0.

## 4. TRL `VLLMClient.generate` return shape

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
