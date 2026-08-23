# GRPO-Guard 项目介绍

> **一句话定位**：给在线 LLM 后训练（TRL GRPO + vLLM）加一条**机器可验证的证据链**
> —— 训练曲线正常不代表训练是对的，GRPO-Guard 证明"训练消费的确实是正确的轨迹"。

---

## 1. 背景：一次真实事故

在做 Agent-RL 训练时，训练曲线一切正常：loss、KL、成功率都在合理范围。
但事后发现 **rollout service 从不加载 trainer 的新权重** —— 静态 rollout：
trainer 每步更新权重，runtime 一直用旧策略生成轨迹，报告的成功率只是
静态策略的批次波动。同时 trainer 会把文本重新 tokenize，并用另一个
policy 重算的值充当 behavior old-logprob。

**旧成功率不能证明模型按声称的算法学习** —— 我停止使用这些结论，
并把事故抽象成框架：训练系统的可靠性不能靠"曲线看起来对"，要靠
**每一跳都可验证的契约**。

## 2. 解决方案：证据链框架

```
Dataset/Split Manifest → TRL GRPO Trainer
  → committed policy v+1 → Checkpoint + PolicyManifest (content-hashed)
  → 上游权重同步（观测 update_named_param）→ vLLM Runtime Adapter
  → GenerationEvent + token artifacts（server 自己的 token ids）
  → Append-only Event/Artifact Store
  → Pre-reward Envelope → Identity Validator → ALLOW → Reward
  → Pre-update Envelope → Pre-update Validator → ALLOW
  → Guarded Batch Materializer → 单次 ValidatedBatchHandle
  → GRPO update（真实 optimizer step）→ UpdateCommitted
  → canary 检查 → v+1 rollout
```

五个核心设计：

| 设计 | 内容 | 防什么 |
|---|---|---|
| **Producer ownership** | runtime 是 token 的唯一生产者；materializer 是 update input 的唯一生产者；任何组件不能改写他人输出 | 静默重 tokenize、轨迹篡改 |
| **No re-tokenization** | optimizer 消费 server 采样的原始 token ids；文本只是 reward verifier 的只读视图 | retokenization 导致的训练/采样不一致 |
| **Reason-coded validation** | P/T/M/L/D/R 六族规则，allow/quarantine/reject 带机器可读原因码 | 无法定位的静默错误 |
| **Guarded update** | 只接受单次 ValidatedBatchHandle；文本 fallback、tokenizer 重调用、nonce 复用都在 optimizer 前 fail closed | 绕过验证的更新 |
| **Deterministic paired replay** | 同一 producer 工件确定性派生故障对，梯度 cosine/L2/update norm 量化影响 | 无法评估的故障影响 |

## 3. 验证结果（全部 gate-passed，可追溯）

| 维度 | 数字 | 证据位置 |
|---|---|---|
| 官方 TRL+vLLM server-mode 冒烟 | ✅ 1 次 committed step、398 次权重同步观测 | `artifacts/v0.1.0/smoke/` |
| 真实闭环 | 32/32 身份 ALLOW + 32/32 预更新 ALLOW → 真实更新 → canary pass | `loop/` |
| 故障注入矩阵 | 基于 **256 条真实 vLLM rollout**，对 F1-F8 八类确定性合成故障逐条注入，**2048 次判定全部符合冻结 oracle**；同批正常轨迹 256/256 ALLOW（0 误拒） | `batch_online_256/` |
| 故障家族 | F1-F4（静态策略/错绑 logprob/重 tokenize/mask 平移）+ F5-F8（split 泄漏/evaluator 别名/事件乱序/工件篡改）+ F9-F10（奖励注入/prompt 投毒） | `tests/frozen/` |
| 梯度影响 | 24 对**离线梯度 probe**（真实模型权重 + 文档化确定性漂移）：F2 cos 0.93、F3 cos 0.53、F4 cos 0.19 | `replay_all/` |
| 真实 RL 训练 | bounded off-policy GRPO：19 次 committed 更新，**loss 非零、参数真实移动（‖θ_v19−θ_v0‖=10.4 fp32 实测）**，guard 每步 ALLOW；成功率曲线如实报告（首 28%、峰值 78%、末 9% —— 训练内 rollout reward，非 held-out 评测，未作能力提升声明） | `rl_training/` |
| 多步闭环 | 3× committed update-sync-rollout、3× canary pass、1876-token 边界 ALLOW | `multi_step/` |
| Guard 开销 | 1.02 ms/envelope（n=256） | `batch_online_256/` |
| 可迁移性 | 第二个任务适配器（GSM8K 数学 QA）不碰 validator/store 层 | `adapters/gsm8k_reward.py` |

## 4. 工程化（production-oriented prototype）

**CI（每次 push 全跑，py3.11/3.12）**：
测试（含 CPU torch 的核心训练路径：grpo_loss、guarded step、sync 状态机）+
core coverage 门禁（≥80%，实测 87%）→ 冻结故障契约检查（F1-F4/F5-F8/F9-F10）
→ Correctness/v0.2 门禁在真实 artifacts 上复验 → SHA256SUMS 完整性 →
证据链校验（seal/顺序/引用）。GPU 闭环不在日常 CI 中重跑（复验已提交
artifacts），release 阶段单独执行。

**生产运维工具链**（全部 CPU、全部进 CI）：

| 工具 | 功能 |
|---|---|
| `grpo-guard verify` | 证据链 attest：checksums + 事件 seal 自洽 + lifecycle 顺序 + 引用完整性 |
| `grpo-guard resume` | 训练恢复：从事件日志生成恢复计划（最后步/checkpoint/下一步）+ `--resume` 续训 |
| `grpo-guard metrics` | Prometheus 指标（决策/原因码/canary/训练成功率），`--serve /metrics` |
| `grpo-guard doctor` | 环境自检：版本 vs 兼容矩阵 + 端口/残留进程 + checkpoint 完整性 |
| `grpo-guard events` / `alert-scan` | 事件检索 + 非 ALLOW 决策 webhook 告警 |
| Streamlit 面板 | 决策分布/血缘追踪/运行健康，现场 demo |

**工程纪律**：决策日志 D1-D17（每个偏离在正式运行前记录）；混沌/模糊测试
（随机变异永不崩溃 + 确定性）；validator 延迟回归基准；三份真实事故
postmortem（canary 误杀训练、engine 死亡恢复、CI checksum 漂移）；
Docker CPU 演示栈。

## 5. 技术栈与规模

- **模型/任务**：Qwen3-4B、Countdown + GSM8K、规则 verifier（确定性）
- **框架**：TRL 1.10.0 server-mode + vLLM 0.26.0、PyTorch 2.11、transformers 5.15
- **硬件**：2×RTX 6000D 84GB，预算 80 GPU·h 硬上限（已用 ~79）
- **规模**：证据链 300+ 个 artifact 文件、SHA256SUMS 全量可验证、CI 每次 push 全绿

## 6. 上游贡献

- huggingface/trl **#6876（open）**：`trl[vllm]` extra 的 fastapi 版本约束
  修复（starlette 1.x 兼容性，真实环境事故驱动；未合并，如实标注）
- 上游 bug 映射：TRL #3774（device 归一化，已验证覆盖我们的场景）、#3762
- 完整记录：`docs/UPSTREAM_FEEDBACK.md`

## 7. 诚实边界（面试必答）

1. v0.1 更新消费自身策略轨迹（loss≈0）—— 梯度影响证据来自配对回放，
   如实标注；on-policy 更新在 bf16 下权重无法移动是数学约束（D14）。
2. 真实权重移动来自 off-policy RL 训练（D15）；成功率曲线是**训练内
   rollout reward**（8 个手写问题、无 seed 重复、无 held-out 评测）——
   证明"loss 非零、参数移动、训练循环真实执行"，**不**作"GSM8K 能力
   提升"声明；曲线含崩溃与事件日志恢复（`recovered: true`）。
3. canary 是行为 sketch（greedy tokens），非逐字节证明；训练中为漂移
   监视器（D17），mismatch 记录为 canary_mismatch 事件（P0-2）；
   P008 fail-closed 保留给非训练场景。
4. 不抵抗恶意伪造；解决的是研发环境的静默接线错误（设计文档 §5.3）。
5. 系统是 production-oriented prototype：单机双卡验证、CPU CI、无多机
   /DDP 长期运行、无外部用户接入；Docker 未本地构建（诚实标注）。

## 8. 仓库结构

```
src/grpo_guard/     schema、store、validators、adapters、faults、replay、CLI、infra 工具
examples/countdown/ 真实闭环、RL 训练、恢复脚本
examples/monitor/   Streamlit 监控面板
configs/            冻结 workload / protocol / fault-matrix
tests/              单元 + 契约 + 属性/混沌 + 冻结用例（145 passed）
artifacts/v0.1.0/   全部 gate 证据（SHA256SUMS 可验证）
docs/               设计文档、决策日志、postmortem、面试材料、上游反馈
DECISION_LOG.md     D1-D17 全部决策记录
```

**复现**：`uv sync --frozen && uv run pytest tests/ && uv run grpo-guard verify --artifact-dir artifacts/v0.1.0 --events artifacts/v0.1.0/loop/events/events`

---

*Release：v0.1.0 / v0.2.0 / v0.3.0（GitHub Releases，含全部证据摘要）。*
