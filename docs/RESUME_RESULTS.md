# GRPO-Guard — resume-ready results (2026-08-22)

Every number below is gate-passed and traces to `artifacts/` + commit +
SHA256SUMS (design doc §16, §20: 简历每个数字可追到 report cell 和 raw
artifact).

## 简历 bullet（精炼版，1-2 行）

**中文版：**
> GRPO-Guard：在线 GRPO 轨迹契约与故障注入框架｜PyTorch、TRL、vLLM
> - 在 Qwen3-4B 真实闭环中实现可审计的训练证据链（内容寻址事件 + 单次守卫更新），8 类故障在 64 个真实 rollout 上 512/512 拒绝、normal 64/64 无误拒
> - 确定性配对回放量化故障梯度影响（24 对）；guard 开销 0.6 ms/条；CI 全绿（py3.11/3.12）

**English version:**
> **GRPO-Guard** — online GRPO trajectory-contract & fault-injection framework | PyTorch, TRL, vLLM
> - Built a machine-verifiable evidence chain (content-addressed events + single-use guarded updates) in a real Qwen3-4B/Countdown closed loop; 8 fault families rejected 512/512 on 64 live rollouts with 0 false rejects
> - Quantified fault impact via 24 deterministic paired gradient probes (cosine distributions); guard overhead 0.6 ms/trajectory; CI green on Python 3.11/3.12

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

## 诚实性声明（面试必答）

- v0.1 更新消费自身策略轨迹（loss≈0、ratio≈1）——梯度影响证据来自 Day 4
  配对回放（v1 权重 + 文档化确定性漂移），如实标注。
- canary 是行为 sketch（greedy tokens），非逐字节证明（设计文档 §5.3）。
- 无密码学防篡改；检测的是研发环境静默接线错误，不是恶意攻击者。
- canary.py 曾有一个常量 sketch bug（dict 解包），已修复并添加回归测试；
  Day 2 闭环路径经核实未受影响——全部如实披露在 REPORT.md。
