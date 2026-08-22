# Demo 录屏脚本（3-5 分钟）

配套 `examples/countdown/demo.py`（CPU 运行，无 GPU 需求）。录屏工具：
OBS / ScreenRec 等，16:9，终端放大到可读。

## 分段脚本

**开场（0:00-0:20）— 仓库与定位**
- 打开 github.com/hxm2023/GRPO-Guard，滚动 README 到架构图
- 旁白：*"GRPO-Guard 给在线 RL 后训练加一条可机器验证的证据链——训练曲线正常不代表训练是对的。"*

**第一幕（0:20-1:10）— 事故动机**
- 打开 `docs/INTERVIEW_PREP.md` 的 90 秒故事段落
- 旁白：*"这个项目的起点是一次线上事故：rollout service 从不加载 trainer 的新权重——静态 rollout，loss 和 KL 都正常，但成功率是静态策略的批次波动。"*
- 切到 `DECISION_LOG.md`：*"所以我把事故抽象成框架，并预先写下决策日志和反证条件——所有设计决定在正式运行前记录。"*

**第二幕（1:10-2:30）— 运行 demo**
- 终端执行：`uv run python examples/countdown/demo.py`
- 逐段讲解输出：
  - happy path：*"正常轨迹——27 条规则全部检查通过，ALLOW。"*
  - F1-F4：*"四个 canonical 故障，每个都带机器可读的原因码——静态 rollout 是 P004、错绑 logprob 是 L003……"*
  - F5-F8：*"v0.2 的四个家族——split 泄漏、评估器别名、事件乱序、工件篡改。"*
  - 最后一行：*"注意这里——文本输入直接被拒。optimizer 只接受验证过的 handle，任何 fallback 都在更新前 fail closed。"*

**第三幕（2:30-3:30）— 真实闭环证据**
- 打开 `artifacts/v0.1.0/loop/run_manifest.json`
- 旁白：*"这是真实闭环的产物——32/32 身份验证 ALLOW、32/32 预更新 ALLOW、一次真实提交更新、398 次权重同步被观测、canary v1 通过。每个数字都有 SHA-256 绑定的事件链。"*
- 打开 `artifacts/v0.1.0/batch_online_64/batch_online_matrix.json` 的 summary：
  *"64 个真实 rollout 上，8 类故障 512/512 被拒绝，normal 64/64 无误拒。"*

**第四幕（3:30-4:30）— 梯度影响**
- 打开 `artifacts/v0.1.0/replay_all/gradient_replay.json` 的 summary
- 旁白：*"24 对配对梯度回放量化了'故障被静默消费'时的真实影响——F2 错绑 logprob 的梯度几乎不动（cosine 0.93，合同才是检测手段），F4 mask 平移会翻转更新方向（cosine 0.19）。"*

**收尾（4:30-5:00）— 诚实边界**
- 旁白：*"诚实的边界：v0.1 更新消费自身策略轨迹所以 loss≈0——梯度影响来自配对回放；canary 是行为 sketch 不是逐字节证明；框架解决研发环境的静默接线错误，不抵抗恶意伪造。仓库里 REPORT、SHA256SUMS、CI 绿标都在。"*

## 录屏检查清单
- [ ] 终端字体 ≥ 14pt、无敏感信息
- [ ] 每个数字在画面上可读且与文档一致
- [ ] 旁白 < 5 分钟、无口误
- [ ] 结尾展示 CI badge（github.com/hxm2023/GRPO-Guard/actions 绿标）
