# GRPO-Guard Experiment Audit

**Date**: 2026-08-25  
**Auditor**: Codex same-context local review  
**Review independence**: `same-context-local`  
**Acceptance status**: `provisional`  
**Project snapshot**: `hxm2023/GRPO-Guard@d8c650e4edc8c1a9c8a856cd41eb4078ac5740aa`

## Overall Verdict: FAIL

## Integrity Status: fail

`FAIL` 针对“当前仓库全部已发布/已引用 claim 可由不可变制品链支持”这一整体主张，不表示所有实验无效。当前 `artifacts/v0.1.0` 快照的 327 个 checksum 和 event verify 通过，多项 claim 可在限定范围内使用；但 README 引用的 `artifacts/v0.2.0-dev/fault_matrix.json` checksum 失配，且 released `v0.1.0` 目录在后续 release 后仍被持续修改，因此整体 evidence-release integrity gate 不应判 PASS。

## Checks

### A. Ground Truth Provenance: PASS with qualifier

- F1–F10 使用预冻结 deterministic injection oracle，适合作为 validator regression ground truth；README 也已将 256-rollout 结果写成“256 real rollouts + per-trajectory injected faults”（`README.md:138`）。
- `batch_online_256` 自报 scope 为“256 real rollouts, per-generation injection”（`artifacts/v0.1.0/batch_online_256/batch_online_matrix.json:3`），结果为 normal 256、F1–F4 1024、F5–F8 1024（同文件 `:18450-18455`）。
- 这不是自然线上 fault distribution 的 ground truth；2048 是派生判定，不是 2048 个独立真实事故。
- held-out 只有 8 个手写 disjoint prompts，base/trained 都为 0.25（`artifacts/v0.1.0/heldout/heldout_eval.json:2-7,62,114`），不能承担模型能力结论。

### B. Score Normalization: PASS

- 未发现用模型自身 max/min/mean 对 accuracy、fault hit 或 gradient cosine 做欺骗性归一化。
- fault matrix 报告直接计数；gradient cosine/relative L2 由原始梯度计算，near-zero 单独标记而非伪装成 0。
- 需要限定的不是归一化，而是样本来源：paired replay 使用 deterministic weight drift，退化 reward group 会按文档翻转一个值；因此属于 mechanism probe，不是自然事故效应。

### C. Result File Existence and Integrity: FAIL

- 当前快照 `artifacts/v0.1.0/SHA256SUMS` 的 327/327 条目经本次 `sha256sum -c` 通过，`grpo-guard verify` 也通过。
- `artifacts/v0.2.0-dev/SHA256SUMS:1-2` 记录的两个 hash 与当前 `contract_check.json`、`fault_matrix.json` 不一致；仅第三个 online matrix 通过。
- README 仍把 `v0.2.0-dev/fault_matrix.json` 用作正式 v0.2 variant evidence（`README.md:145`），所以不能把失配包完全视为无关临时文件。
- CI 只校验 `artifacts/v0.1.0`（`.github/workflows/ci.yml:42-47`），绿色 CI 不覆盖上述 dev mismatch。
- 顶层 run manifest 初始 commit 为空、platform 为 Windows/Python 3.14.3、`stages={}`（`artifacts/v0.1.0/run_manifest.json:2-17`），不足以证明主要 Linux/GPU run 的完整 provenance。
- `v0.3.0` tag 为 `31bb20d`，当前 main 比其多 59 commits；之后仍向 `artifacts/v0.1.0` 追加 P0/P1 evidence。当前 checksum 只能证明当前 bytes 自洽，不能证明历史 release pack 不可变。

### D. Dead Code / Actual Path Detection: WARN

- custom RL loop 的 guard-on arm确实调用 `guarded_optimizer_step`（`examples/countdown/rl_training_loop.py:437-472`），不是完全 dead code。
- 但同仓库多个旧 example 仍直接调用 `optimizer.step()`；“repository-wide single unbypassable entry”不成立。
- `guarded_optimizer_step` 在 nonce/handle consume 后才执行 loss/backward/step/commit（`src/grpo_guard/adapters/guarded_update.py:290-314`），没有 post-step rollback。
- actual rewards 是未 content-address 的 NumPy 参数（同文件 `:174-205`），group size/model identity 也由 caller 另传；validated event 没有完全绑定最终 update semantics。
- official `GuardedGRPOTrainer._guard_pre_update` 不读取 `inputs`，只检查 completion 总长度大于 0（`src/grpo_guard/adapters/guarded_grpo_trainer.py:171-185`）；generation mask/logprob 使用零 hash placeholder（同文件 `:142-168`），commit hook只设置 digest 属性（同文件 `:187-202`）。official-path 1-step smoke 不能证明 strict guard 真接入 optimizer input。

### E. Scope Assessment: WARN

- 真实范围：Qwen3-4B、TRL/vLLM server、单机 2×RTX 6000D、短 run、手写 math/countdown 类任务。
- README 正确披露 no multi-node/DDP（`README.md:28-34`）。仓库无 FSDP/ZeRO、真实 NPU、HCCL、真实 Agent benchmark 证据。
- 19-step 旧 run 的 success 首 0.2812、峰 0.7812、末 0.0938，参数距离 10.415（`artifacts/v0.1.0/rl_training/rl_training.json:299-305`）；该曲线支持“更新与 collapse 均真实发生”，不支持能力提升。
- guard on/off 仅 3 seeds × 10 steps、8 个重复训练 prompts；最多支持小型 smoke 中初步未见明显 degradation。

### F. Evaluation Type

`real_rollout_with_targeted_synthetic_fault_injection_and_mechanism_replay`

证据应分层描述：

1. real Qwen3-4B/vLLM rollout；
2. deterministic synthetic fault mutations；
3. real-model gradient probe with documented synthetic weight/reward perturbation；
4. real custom-loop optimizer updates/recovery；
5. small frozen held-out evaluator probe；
6. official Trainer seam smoke，而非完整 official guarded update。

## Claim Impact

| Claim | Impact | Safe replacement |
|---|---|---|
| 256 real rollout 上 2048 次 F1–F8 判定匹配 frozen oracle | needs qualifier | 保留“256 real rollout + 2048 targeted injected decisions” |
| 2048/4096 个真实线上故障全部检测 | unsupported | 删除 |
| 24 对梯度 probe 量化 F2/F3/F4 | needs qualifier | 加“v1 + deterministic drift；机制探针” |
| GSM8K 28%→78% 能力提升 | unsupported | 写训练内 reward 峰值后 collapse；held-out probe 25%→25% |
| guarded update 对任何失败事务原子 | unsupported | 只声称 contract preconditions fail before backward |
| official GRPOTrainer optimizer input 已被严格守护 | unsupported | 写 1-step seam instrumentation smoke |
| 当前 CI 覆盖率精确 87% | needs evidence | 当前公开证据只保证 core coverage ≥80% |
| 当前 main 即 released v0.3.0 | unsupported | main 标为 0.4.0-dev，重新冻结并发布 |
| 当前 v0.1.0 snapshot checksum/event verify | supported | 明确“current snapshot”，不声称历史不可变 |

## Action Items

1. 立即修复/隔离 `v0.2.0-dev` mismatch，并让 CI 遍历全部 published packs。
2. 停止修改历史版本目录；发布 immutable `v0.4.0` pack 和完整 run provenance。
3. 删除 atomic/unbypassable 强声明，完成 WAL、transactional nonce、worker recovery 和 failpoint matrix。
4. 绑定 reward tensor、group membership/size、model identity 与 actual official Trainer inputs。
5. 将 normal non-inferiority 与 fault-injected survival 分成预注册实验，使用 official dataset、paired seeds 和 held-out metric。

详细求职定位、简历 bullet、面试话术和实施路线见 `GRPO-Guard_核心项目评估与优化建议_2026-08-25.md`。
