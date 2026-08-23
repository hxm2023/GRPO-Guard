# GRPO-Guard — resume-ready results (updated 2026-08-23)

Every number below is gate-passed and traces to `artifacts/` + commit +
SHA256SUMS (design doc §16, §20: 简历每个数字可追到 report cell 和 raw
artifact).

## 简历 bullet（精炼版，1-2 行）

**中文版：**
> GRPO-Guard：在线 GRPO 轨迹一致性审计与故障注入框架｜PyTorch、TRL、vLLM
> - 从 Agent-RL 静态 rollout 事故中抽象 policy version / server token / behavior logprob / loss mask / reward lineage 契约，设计 content-addressed event/artifact store、分阶段 validator 与确定性 paired replay
> - 基于 256 条 Qwen3-4B/vLLM 真实生成轨迹，对 F1-F8 八类预定义接线故障逐条注入，2048 次判定全部符合冻结 oracle，正常轨迹 256/256 通过；24 对离线梯度 probe 量化 retokenization 与 mask shift 对更新方向的影响
> - 发布 Apache-2.0 开源仓库、版本化 artifacts/SHA256 验证与 CPU contract CI（core coverage 87%），并向 Hugging Face TRL 提交依赖兼容修复 PR #6876（open）

**English version:**
> **GRPO-Guard** — online GRPO trajectory-consistency audit & fault-injection framework | PyTorch, TRL, vLLM
> - Abstracted policy-version / server-token / behavior-logprob / loss-mask / reward-lineage contracts from a real static-rollout incident; designed a content-addressed event/artifact store, staged validator and deterministic paired replay
> - On 256 real Qwen3-4B/vLLM rollouts, injected all 8 predefined wiring faults and got 2048/2048 decisions matching the frozen oracle, 256/256 normal trajectories allowed; 24 paired gradient probes quantify retokenization/mask-shift impact on update direction
> - Released an Apache-2.0 repo with versioned artifacts/SHA256 verification and CPU contract CI (core coverage 87%); opened huggingface/trl #6876 (open) dependency-compat fix

## Resume bullets (设计文档 §20.3 长版 — Release Gate passed)

**GRPO-Guard：在线 GRPO 轨迹契约与故障注入框架｜PyTorch、TRL、vLLM、Qwen3-4B**

- 在 Qwen3-4B / Countdown / 2×RTX 6000D 上集成并观测 TRL/vLLM 上游
  weight-sync 生命周期（**398 次 update_named_param 调用/次同步**），实现
  content-addressed trajectory envelope（事件链 + SHA-256 + 单写者 lease），
  完成真实 `update → commit → sync → new rollout` 闭环：32 条 v0 轨迹
  **32/32 身份验证 ALLOW**、**32/32 预更新验证 ALLOW**、1 次真实 optimizer
  step、canary v1 pass（5 次重载校准，tolerance 0）。
- 对预先冻结的 8 类故障（F1-F4 canonical + F5-F8 v0.2）在**真实 server
  rollout 上**完成注入矩阵：批量在线矩阵 32 rollouts —— F1-F4 128/128
  reject、F5-F8 128/128 reject/quarantine、normal 32/32 ALLOW（0 false
  reject）；扩展至 64 rollouts —— F1-F4 256/256 reject、F5-F8 256/256
  reject/quarantine、normal 64/64 ALLOW**；单点在线矩阵 4/4 + 4/4；
  v0.2 变体矩阵 12/12。
- 从同一 producer artifact 确定性派生 fault pairs，量化梯度影响（**24 对
  配对梯度**，v1 权重 + 确定性漂移）：F2 misbound logprobs 平均 cosine
  **0.93**（值相近时梯度几乎不动——合同检测的边界被如实量化）、F3
  retokenization **0.53**、F4 mask shift **0.19**（部分组方向翻转，min
  -0.03）；F1 guard-off update norm 0.0 如实报告（fp32 精度测得）。
- **bounded off-policy 在线闭环**（§9.2）：lag=1 消费 v0 轨迹（界内 +
  声明 correction → ALLOW），一次 committed update + **398 参数同步 +
  v1 提交**；F1 梯度影响如实报告 undefined_near_zero（合同故障在 P004
  optimizer 前拦截，不造 cosine）。
- Guard 开销：在线 validator **0.59–0.63 ms/envelope**（n=32/64）；离线
  3 次重复 raw+mean±sd 全报；P008 canary-mismatch 在线验证（drift 32
  tokens → reject）。

## 门控状态

| Gate | 状态 | 关键数字 |
|---|---|---|
| Day 1 Compatibility | ✅ | 官方 TRL+vLLM server-mode 冒烟通过；398 sync calls；release commit 复现重跑通过 |
| Day 2 闭环 | ✅ | 32/32 + 32/32 ALLOW；1 次真实更新；v1 rollout 验证 |
| Day 3 Correctness | ✅ | canonical 4/4；12/12 变体；32/32 normal；boundary 4/4；stale acceptance 0 |
| Day 4 Impact/Overhead | ✅ | 24 对梯度分布；overhead 40.4 ms/batch（3 重复）；F1 update norm 0.0 |
| Day 5 Release | ✅ | fresh clone + uv sync --frozen + 全量测试；README/demo/REPORT/SHA256SUMS；tag v0.1.0/v0.2.0 |
| v0.2（F5-F8 正式化） | ✅ | 注入协议冻结；在线 4/4；变体 12/12；P008 在线 reject |
| v0.2.1（F9-F10） | ✅ | reward 注入 R008 + prompt 投毒 D004；frozen 3/3 + normal 4/4 GATE PASS |
| 真实 RL 训练（D15/D17） | ✅ | 19 committed updates；loss 非零；权重 delta 10.4（fp32 实测）；全程 ALLOW；成功率曲线如实报告（训练内 reward，非 held-out） |
| 多步闭环（D14） | ✅ | 3× committed update-sync-rollout；3× canary pass；1876-token 边界 ALLOW |
| 最大真实负载（D13） | ✅ | 256 rollouts：normal 256/256；F1-F4 1024/1024；F5-F8 1024/1024 |
| Infra 工具链 | ✅ | verify（证据链校验）/ resume（训练恢复）/ metrics（Prometheus）/ doctor（环境自检）/ alert-scan |

## 诚实性声明（面试必答）

- v0.1 更新消费自身策略轨迹（loss≈0、ratio≈1）——梯度影响证据来自 Day 4
  配对回放（v1 权重 + 文档化确定性漂移），如实标注。
- on-policy 更新在 bf16 下权重无法移动（数学约束，D14 如实记录）；真实权重
  移动来自 off-policy RL 训练（D15，||θ_v19−θ_v0||=10.4 实测）。
- RL 训练曲线尾段下滑（小 batch GRPO 不稳定）——如实报告含崩溃；
  vLLM engine 第 20 步死亡后从事件日志恢复（recovered: true，无伪造）。
- canary 是行为 sketch（greedy tokens），非逐字节证明（设计文档 §5.3）；
  训练中为漂移监视器（D17），非训练场景保持 fail-closed（P008）。
- 无密码学防篡改；检测的是研发环境静默接线错误，不是恶意攻击者。
- canary.py 曾有一个常量 sketch bug（dict 解包），已修复并添加回归测试；
  Day 2 闭环路径经核实未受影响——全部如实披露在 REPORT.md 与
  docs/POSTMORTEMS.md。
