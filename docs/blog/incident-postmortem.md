# 我如何发现 GRPO 训练曲线是假的

> 事故复盘 · 2026-08 · 首发于 GRPO-Guard 仓库（[zhihu/github 可转载]）
> 配套开源框架：[GRPO-Guard](https://github.com/hxm2023/GRPO-Guard)（Apache-2.0）

## 一、训练曲线很正常，模型却没有在学习

去年我在做 Agent 场景的 GRPO credit assignment。训练日志长这样：

```
step 120  loss=0.42  kl=0.31  reward=0.61  success=0.72
step 121  loss=0.41  kl=0.30  reward=0.63  success=0.74
```

loss 在降，reward 在涨，KL 有界，曲线漂亮。没有人怀疑什么——直到我拿训练后的 checkpoint 去跑 evaluation，发现行为跟**训练开始前**几乎一样。

排查到最后，原因平淡得让人后怕：**rollout service 没有加载 trainer 更新后的策略**。trainer 每步都在更新权重，runtime 却一直用旧策略生成轨迹。训练日志里那些"成功率的提升"，只是**静态策略在不同 prompt 批次上的随机波动**——不是模型在学，是噪声在跳舞。

更隐蔽的是第二层：trainer 把 server 返回的文本重新 tokenize，用另一个 policy 重算的 logprob 当 behavior old-logprob。即使 rollout 是对的，loss 的分母也可能是错的。只看 loss 数值，永远发现不了。

## 二、为什么"记录数值"发现不了这类错误

事后看，问题不是"没记录"，而是**记录的东西没有身份**：

- 日志里的 loss 属于哪个 policy version 的 rollout？不知道。
- 这条 trajectory 的 token ids 是 server 采样的，还是 trainer 重 tokenize 的？不知道。
- 这个 old-logprob 是行为策略产生的，还是另一套模型前向算的？不知道。
- optimizer 消费的 reward 和 reward event 里声明的是同一批值吗？不知道。

普通 logging/W&B 记录的是"某个值"，不绑定 producer、policy version、artifact bytes 和消费关系。而 GRPO/PPO 这类 off-policy 算法，**数值形状对上了不等于语义对上了**——错误策略的 logprob 可能数值很接近，loss 看不出异常，但 estimator 已经失去正确含义。

## 三、把"谁生成了什么、谁消费了什么"变成机器可验证的契约

我把这次事故抽象成一个可靠性框架（不是新算法，是正确性层）：

1. **Producer ownership**：runtime 是 token 的唯一生产者，trainer 不能偷偷重 tokenize；materializer 是 optimizer input 的唯一生产者，optimizer 只接受它铸造的单次 handle。
2. **Content-addressed 证据链**：policy/checkpoint → sync → generation → scoring → reward → pre-update → update，每一跳生成不可变 event + artifact hash。任何一环被绕过，下一环对不上。
3. **Reason-coded validator**：P/T/M/L/D/R 规则表，`allow`/`quarantine`/`reject` + 机器可读原因码（如 `P004_STALE_POLICY_STRICT`、`M004_CANONICAL_MASK_MISMATCH`）。
4. **Fail closed**：所有前置条件（非 ALLOW、artifact 篡改、reward 替换、group 顺序错、模型对象错、nonce 复用）在 backward 之前拒绝，参数可证明不变。
5. **Crash consistency**：WAL `PREPARED→APPLIED→CHECKPOINTED→COMMITTED`，step 之后的失败不假装回滚，恢复 = 丢弃 worker、从 last committed checkpoint 续跑。

关键洞察：**检查的不是"loss 是否正常"，而是"这条轨迹的身份链是否闭合"**。

## 四、怎么验证一个"正确性框架"是对的

正确性框架最大的坑是自证。我的做法是**在真实闭环里注入故障**：

- 真实 Qwen3-4B + TRL + vLLM server 闭环（2×RTX 6000D）；
- 256/512 条真实 rollout，逐条注入 F1–F8 八类预定义接线故障（静态 rollout、logprob 错绑、retokenization、mask shift……），2048/4096 次判定全部匹配冻结 oracle；正常轨迹 100% ALLOW，0 false reject；
- 24 对配对梯度探针量化"这些错误会怎样改变更新方向"：F2 错绑 logprobs 的梯度 cosine 高达 0.93——**数值几乎不变，正是只看 loss 发现不了的原因**；F4 mask shift 低到 0.19，部分组方向翻转；
- 官方 TRL GRPOTrainer 路径 20 步真实 run，逐步校验实际消费的张量，在 step 10/19 注入 retokenization，两次都在 backward 前被拒并恢复。

## 五、负结果和边界（这部分更重要）

- 冻结的 8 个 held-out prompt 上，训练前后都是 25%——**训练内 reward 提升不泛化**，所以我从不声称"能力提升"，只声称"训练循环真实执行、契约全程守护"。
- 我最初把 guarded step 叫做"事务原子"，审计后降级为精确表述：前置条件在 backward 前失败；step 后的失败走 crash recovery，**不做内存回滚**。failpoint 测试全部入库。
- canary 只是行为 sketch（greedy tokens），不是字节级证明。
- 不防恶意 producer——解决的是研发环境里的静默接线错误。

## 六、启示

训练曲线不能证明训练正确。"看起来正常"是最危险的失败模式——loss 在降，reward 在涨，而模型什么都没学到。

如果你的训练系统里，rollout 和 trainer 是异步的，server 返回的 token 要经过文本再编码，reward 和 optimizer input 之间隔着好几层——那么"这条轨迹是哪个策略生成的、optimizer 到底消费了什么"不应该是一个靠人肉核对的问题。把它变成契约。

---

*代码、证据包（全部可校验的 checksum）、审计记录见 [github.com/hxm2023/GRPO-Guard](https://github.com/hxm2023/GRPO-Guard)。本文欢迎转载，注明出处即可。*
