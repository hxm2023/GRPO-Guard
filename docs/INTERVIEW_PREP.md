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
> 项目结果：Qwen3-4B + Countdown 真实闭环（32/32 身份 + 32/32 预更新
> ALLOW、1 次真实提交更新、398 次权重同步观测、canary v1 pass）；8 类
> 故障在 64 个真实 rollout 上 512/512 拒绝/隔离、normal 64/64 ALLOW；
> 24 对配对梯度量化了故障的梯度影响；guard 开销 0.6 ms/条。它不抵抗
> 恶意伪造，解决的是研发环境里的静默接线错误。

## 主管面故事（设计文档 §20.5 框架）

1. **发现证据链不闭合后停止使用受影响结论**——旧成功率、ρ=0.735 全部作废，不洗白。
2. **保留失败 run**——没有删掉不漂亮结果，作为事故档案。
3. **用 exact oracle 区分实现错误与方法本身无效**——先审计估计对象，机制对照发现 adaptive mapping 退化成固定 K，继续关闭算法 headline。
4. **预先定义 Gate 和预算上限**——五道门 + 80 GPU·h 硬上限，避免无止境调参；决策日志（D1–D10）在正式运行前记录。
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

## 演示要点（3-5 分钟）

`uv run python examples/countdown/demo.py`（CPU）：
happy path ALLOW → F1-F4 reject（各代码）→ F5-F8 → 文本输入被拒
（TypeError）。讲"loss 看起来正常但合同失败"的反例（控制 loss vs
F2-fault loss 几乎相同，合同 reject）。
