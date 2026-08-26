# GRPO-Guard — resume-ready results (updated 2026-08-23)

Every number below is gate-passed and traces to `artifacts/` + commit +
SHA256SUMS (design doc §16, §20: 简历每个数字可追到 report cell 和 raw
artifact).

## 简历 bullet（精炼版，1-2 行）

**中文版：**
> GRPO-Guard：在线 GRPO 轨迹一致性审计与故障注入框架｜PyTorch、TRL、vLLM
> - 从 Agent-RL 静态 rollout 事故中抽象 policy version / server token / behavior logprob / loss mask / reward lineage 契约，设计 content-addressed event/artifact store、分阶段 validator 与确定性 paired replay
> - 基于 256 条 Qwen3-4B/vLLM 真实生成轨迹，对 F1-F8 八类预定义接线故障逐条注入，2048 次判定全部符合冻结 oracle，正常轨迹 256/256 通过；24 对离线梯度 probe 量化 retokenization 与 mask shift 对更新方向的影响
> - 发布 Apache-2.0 开源仓库、不可变证据包（SHA256 校验 + RELEASE_MANIFEST + 双版本 CI，coverage 报告作为 CI artifact 可下载，门槛 ≥80%），并向 Hugging Face TRL 提交依赖兼容修复 PR #6876（open）

**English version:**
> **GRPO-Guard** — online GRPO trajectory-consistency audit & fault-injection framework | PyTorch, TRL, vLLM
> - Abstracted policy-version / server-token / behavior-logprob / loss-mask / reward-lineage contracts from a real static-rollout incident; designed a content-addressed event/artifact store, staged validator and deterministic paired replay
> - On 256 real Qwen3-4B/vLLM rollouts, injected all 8 predefined wiring faults and got 2048/2048 decisions matching the frozen oracle, 256/256 normal trajectories allowed; 24 paired gradient probes quantify retokenization/mask-shift impact on update direction
> - Released an Apache-2.0 repo with immutable evidence packs (SHA256 verification + RELEASE_MANIFEST) and dual-version CPU contract CI (coverage gate >=80%, report downloadable as CI artifact); opened huggingface/trl #6876 (open) dependency-compat fix

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
| v0.4.0-dev 审计整改（2026-08-25） | ✅（CPU 验证） | SQLite 事务 nonce（32 进程竞争恰好一个成功）；WAL PREPARED→COMMITTED + 五类 failpoint；reward/group/model 实际输入绑定；官方 Trainer 实际 inputs 校验；冻结 v0.4.0 证据包 + RELEASE_MANIFEST + CI 遍历全部 packs |
| v0.2（F5-F8 正式化） | ✅ | 注入协议冻结；在线 4/4；变体 12/12；P008 在线 reject |
| v0.2.1（F9-F10） | ✅ | reward 注入 R008 + prompt 投毒 D004；frozen 3/3 + normal 4/4 GATE PASS |
| 真实 RL 训练（D15/D17） | ✅ | 19 committed updates；loss 非零；权重 delta 10.4（fp32 实测）；全程 ALLOW；成功率曲线如实报告（训练内 reward，非 held-out） |
| 多步闭环（D14） | ✅ | 3× committed update-sync-rollout；3× canary pass；1876-token 边界 ALLOW |
| 最大真实负载（D13） | ✅ | 256 rollouts：normal 256/256；F1-F4 1024/1024；F5-F8 1024/1024 |
| Infra 工具链 | ✅ | verify（证据链校验）/ resume（训练恢复）/ metrics（Prometheus）/ doctor（环境自检）/ alert-scan |
| P1-1 官方 Trainer 包裹 | ✅ | GuardedGRPOTrainer：官方 TRL GRPOTrainer 三 seam + 实际张量校验；**20 步真实 server-mode run**（每步校验，F3 注入 step 10/19 均 T001 阻断并恢复，run_packs/p0_4_official_trl/）；optimizer.step 已 capability-gated；双源 runtime attestation（server logprob 指纹 vs trainer 前向） |
| P1-2 guard on/off 3-seed | ✅ | guard-on 0.698±0.049 vs off 0.709±0.050（差异 <1σ，guard 不劣化）；held-out 25%→25%（训练内提升不泛化，诚实结论） |
| E1 故障注入训练生存 | ✅ | 官方路径 30 步训练，step 10/20 注入 F3 retokenization + F2 logprob misbinding：guard-on **0/4 坏更新被接受、检测延迟 0 步、浪费 0 步**（F3→T001、F2→L004 均在 backward 前阻断并恢复）；guard-off 4/4 坏更新应用、30 步浪费；step 时间开销 ~1%（run_packs/e1_fault_survival/） |
| E2 non-inferiority | ✅ | 5 seeds × on/off，16-prompt 冻结 held-out：on 0.250±0.06 vs off 0.288±0.11，delta -0.0375 **在预注册 margin（≤1/16）内**且 on 方差更小；训练内 reward on 0.273 vs off 0.263；step 开销 +2.3%（run_packs/e2_non_inferiority/） |
| P1-3 tool-use 契约 | ✅ | action-only mask、stale/duplicate/orphan observation 检测（确定性环境） |

## 诚实性声明（面试必答）

- v0.1 更新消费自身策略轨迹（loss≈0、ratio≈1）——梯度影响证据来自 Day 4
  配对回放（v1 权重 + 文档化确定性漂移），如实标注。
- on-policy 更新在 bf16 下权重无法移动（数学约束，D14 如实记录）；真实权重
  移动来自 off-policy RL 训练（D15，||θ_v19−θ_v0||=10.4 实测）。
- RL 训练曲线尾段下滑（小 batch GRPO 不稳定）——如实报告含崩溃；
  vLLM engine 第 20 步死亡后从事件日志恢复（recovered: true，无伪造）。
- canary 是行为 sketch（greedy tokens），非逐字节证明（设计文档 §5.3）；
  训练中为漂移监视器（D17），非训练场景保持 fail-closed（P008）。
- 2026-08-25 审计后：`guarded_optimizer_step` 的"原子"表述已降级为精确
  claim——所有 precondition 在 backward 前失败（参数不变）；step 后的
  失败走 crash recovery（WAL + 丢弃 worker + 从 last committed checkpoint
  续跑），不做内存回滚。failpoint 测试与 32 进程 nonce 竞争测试已入库
  （docs/audits/ 保留完整审计链）。
- 无密码学防篡改；检测的是研发环境静默接线错误，不是恶意攻击者。
- canary.py 曾有一个常量 sketch bug（dict 解包），已修复并添加回归测试；
  Day 2 闭环路径经核实未受影响——全部如实披露在 REPORT.md 与
  docs/POSTMORTEMS.md。
