# GRPO-Guard 面试准备包

配套 `docs/RESUME_RESULTS.md`（gate-passed 数字）。所有数字可追溯到
artifacts + commit + SHA256SUMS（设计文档 §20 规则：简历/面试只讲
gate-passed 数字，不扩大范围）。

## 90 秒故事（技术面，设计文档 §20.4 框架）

> 我做 Agent-RL credit assignment 时，训练曲线能跑，loss 和 KL 都正常，
> 但后来发现 rollout service 没有加载 trainer 更新后的策略——**静态
> rollout**：trainer 每步更新权重，runtime 一直用旧策略，报告的成功率
> 只是静态策略的批次波动。trainer 还把文本重新 tokenize，并用另一个
> policy 重算的值当 behavior old-logprob。旧成功率不能证明模型按声称的
> 算法学习，我主动停止使用这些结论。
>
> 我把事故抽象成 GRPO-Guard：在 TRL + vLLM 的真实 GRPO 闭环里，把
> behavior policy、原始 token/logprob/mask、数据、reward 封装成不可随意
> 改写的 trajectory envelope——runtime/scorer 各自产生事件和
> content-addressed artifacts，assembler 只引用，validator 按
> policy/token/mask/lifecycle 规则判定 allow/quarantine/reject。
> optimizer 只接受单次 ValidatedBatchHandle，任何文本 fallback、
> tokenizer 重调用、nonce 复用都在 optimizer 前 fail closed。
>
> 项目结果：Qwen3-4B 真实闭环（32/32 身份 + 32/32 预更新 ALLOW、1 次
> 真实提交更新、398 次权重同步观测、canary v1 pass）；基于 256 条真实
> rollout 对 F1-F8 八类故障逐条注入，2048 次判定全部符合冻结 oracle、
> normal 256/256 ALLOW；24 对离线梯度 probe 量化了故障的梯度影响；
> guard 开销 1 ms/条。最后我让框架自己跑了一次**真实 RL 训练**
> （bounded off-policy GRPO，19 次 committed 更新，loss 非零、权重
> 真实移动），训练中断后从事件日志恢复——guard 全程守护。它不抵抗
> 恶意伪造，解决的是研发环境里的静默接线错误。

## 主管面故事（设计文档 §20.5 框架）

1. **发现证据链不闭合后停止使用受影响结论**——旧成功率、ρ=0.735 全部作废，不洗白。
2. **保留失败 run**——没有删掉不漂亮结果，作为事故档案。
3. **用 exact oracle 区分实现错误与方法本身无效**——先审计估计对象，机制对照发现 adaptive mapping 退化成固定 K，继续关闭算法 headline。
4. **预先定义 Gate 和预算上限**——五道门 + 80 GPU·h 硬上限，避免无止境调参；决策日志（D1–D17）在正式运行前记录。
5. **把一次事故变成团队可复用的测试和 release 规范**——事件密封、内容哈希、no-overwrite、SHA256SUMS、reason-coded 矩阵，任何训练系统都能复用。

## 常见追问与回答要点

**Q1: 你的 guard 和直接加 assert 有什么区别？**
- assert 是单点检查；guard 是**身份链**：event → artifact → envelope → decision
  → handle → update 每一跳都有内容哈希和 producer 归属，任何一环被绕过
  都会在下一环暴露（P/T/M/L/D/R reason codes）。
- 关键设计：**producer ownership**——runtime 是 token 唯一生产者，
  trainer 不能偷偷重 tokenize；materializer 是 update input 唯一生产者，
  optimizer 只接受它。

**Q2: 为什么 loss=0？这是不是说明更新是假的？**
- 不是假的——optimizer step 真实执行（backward + step + checkpoint 提交 +
  398 参数同步 + canary）。loss=0 是因为 v0.1 的更新消费**自身策略**的
  轨迹（behavior==new → ratio≈1）——这是设计选择（严格 on-policy），
  梯度影响由 Day 4 配对回放量化（F2/F3/F4 的 24 对 cosine 分布）。
- 面试加分点：主动说"这个数字看起来可疑，我验证过它为什么是 0"。

**Q3: 可以检测 100% 的故障吗？**
- 不能，也从不声称。F1-F4 canonical 4/4 是**冻结用例**上的结果；held-out
  变体如实报告（P003/L006/T004/M004 分支）。检测率只针对预注册的注入
  集合，不泛化为 100%（设计文档 §11 明确禁止）。

**Q4: 你的 validator 会不会误伤正常轨迹？**
- normal 集合：32/32 和 64/64 ALLOW（0 false reject）；boundary 4/4 按
  预注册期望处理（空 completion → quarantine M005；padding/truncation/
  dual-source-diagnostic → allow）。

**Q5: TRL/vLLM 版本那么具体，换版本怎么办？**
- adapter 层隔离：schema/validator 不依赖 TRL 私有对象；版本变化集中在
  adapter（compatibility_profile.yaml 冻结矩阵；两个小 patch 带版本
  assert，fail-closed）。换版本 = 更新 profile + 冒烟，核心不变量不动。

**Q6: 和 Agent-RL Credit Auditor 什么关系？**
- Guard 管"这条轨迹和这次更新的身份链闭合吗"（在线）；Auditor 管
  "credit estimator 到底估计什么"（离线 exact）。共用 trajectory
  artifact 格式但独立仓库（§1.3 边界）。

**Q7: 为什么不是 PPO/DPO/其他？**
- 固定 workload 是设计文档锁定的（GRPO + Countdown + 规则 verifier）；
  项目的产出是**可靠性框架**不是新算法——换算法不换 guard 的机制。

**Q8: 你跑过真实 RL 训练吗？成功率真的提升了吗？**
- 跑过。bounded off-policy（FIFO lag-1 缓冲）GRPO，Qwen3-4B + GSM8K：
  19 次 committed 更新，**loss 非零（off-policy ratio≠1）、权重真实移动
  （||θ_v19−θ_v0||=10.4 fp32 实测）**；guard 每步 identity + pre-update
  ALLOW、398 参数同步、canary 漂移监视。
- 诚实点：成功率曲线是训练内 rollout reward（8 个手写问题、无 seed
  重复、无 held-out 评测）—— 可以证明"训练循环真实执行、参数真实
  移动"，**不能**证明"GSM8K 能力提升"，所以我不把 78% 峰值放进简历；
  曲线尾段回落如实报告；vLLM engine 在第 20 步死亡后从事件日志恢复
  （recovered: true）。教训：canary 的"权重不变"语义在训练中不适用
  （D17 改为漂移监视器），mismatch 记录为 canary_mismatch 事件（P0-2）。

**Q9: 训练崩了怎么办？生产上怎么保证证据可信？**
- 事件流是真相源：每步指标持久化为 training_step 事件（不依赖 run
  log），`grpo-guard resume` 从事件日志生成恢复计划（最后完成步 +
  checkpoint + 下一步），训练脚本 `--resume` 加载 checkpoint 继续。
- `grpo-guard verify` 校验证据链（SHA256SUMS + 事件 seal 自洽 +
  lifecycle 顺序 + 引用完整性）；`doctor` 环境自检（版本 vs
  compatibility profile + 端口/残留进程）；`metrics` 暴露 Prometheus
  指标（决策/原因码/canary/训练成功率）；`alert-scan` 非 ALLOW 决策
  webhook 告警。全部在 CI 里跑。

**Q12: 和官方 TRL Trainer 的关系？**
- `GuardedGRPOTrainer`（P1-1）：官方 `trl.GRPOTrainer` 的子类 mixin，在
  rollout（`_generate_and_score_completions`）、step（`training_step`）、
  commit（`_save_checkpoint`）三个 seam 插桩 —— 每个 completion 做
  align/logprob 契约检查并记录 GenerationEvent，step 前一致性检查，
  checkpoint 内容哈希。真实 1 步 server-mode 实跑通过（4 rollouts +
  events + commit sha）。诚实边界：这是官方路径的契约插桩，严格
  envelope/handle 管线是正式训练路径。
- 训练内曲线（34%→80% rollout reward）经 held-out 评测（25%→25%）
  证实不泛化 —— 所以我从不声称"能力提升"，只讲"训练循环真实执行 +
  契约全程守护"。

**Q10: optimizer 真的只能消费验证后的数据吗？怎么证明不可绕过？**
- `guarded_optimizer_step` 是 capability-gated 的 optimizer 入口：
  验证（decision ALLOW + artifact hash + reward hash + group/model 身份 +
  tokenizer_called）→ 事务化 nonce 消费（SQLite，跨进程 exactly-once）→
  handle 消费 → loss → backward → step → commit。所有 precondition 失败
  都发生在 backward 之前，参数可证明不变（测试断言）。
- handle 只能由 materializer 铸造（issuer token，外部构造 → TypeError）；
  nonce 跨进程持久化（SQLite 唯一约束，复用 → NonceReuseError；
  32 进程竞争测试恰好一个成功）。
- 诚实边界（2026-08-25 审计后）：step 之后的失败（checkpoint/commit）不做
  内存参数回滚——WAL 记录 PREPARED→APPLIED→CHECKPOINTED→COMMITTED，
  恢复语义是"丢弃 worker，从 last committed checkpoint 续跑"（crash
  consistency，不是数据库式事务原子）。failpoint 测试覆盖
  loss/backward/step/checkpoint/commit 五处。
- 真实 GPU 验证：20 步 RL 训练全部走此入口（每步 loss 非零、B=64
  微批次化防 OOM）；被共享服务器中断 3 次后 `--resume` 从事件日志无缝续跑。

**Q11: 怎么证明 runtime 真的加载了新权重（静态 rollout 检测）？**
- 事件链：runtime_loaded 只在真实 398 次 update_named_param 调用后记录
  （带调用数 + name digest，P0-2）；canary mismatch 记录为
  canary_mismatch（绝不写 canary_passed）。
- 数据面：server-vs-trainer greedy sketch 对比 —— 实测 server 服务 v0
  而 trainer 持训练后 v20 时 drift=7 → **STALE RUNTIME DETECTED**
  （原始静态 rollout 事故的机器检测，guard 在 optimizer 前阻断）。

## 演示要点（3-5 分钟）

`uv run python examples/countdown/demo.py`（CPU）：
happy path ALLOW → F1-F4 reject（各代码）→ F5-F8 → 文本输入被拒
（TypeError）。讲"loss 看起来正常但合同失败"的反例（控制 loss vs
F2-fault loss 几乎相同，合同 reject）。

追加演示（可选）：
- `uv run grpo-guard verify --artifact-dir artifacts/v0.1.0 --events
  artifacts/v0.1.0/loop/events/events` → OK（证据链 attest）
- `uv run grpo-guard resume --events <rl_events> --out /tmp/plan.json`
  → 恢复计划（最后步/checkpoint/下一步）
- `uv run grpo-guard metrics --dir <events>` → Prometheus 指标
- `streamlit run examples/monitor/panel.py` → 监控面板（决策/血缘）
