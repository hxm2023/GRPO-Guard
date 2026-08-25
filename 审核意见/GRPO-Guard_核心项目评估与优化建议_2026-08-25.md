# GRPO-Guard 能否作为大模型后训练求职核心项目：代码、证据与简历定位审计

> 审计日期：2026-08-25（Asia/Shanghai）  
> 仓库：[hxm2023/GRPO-Guard](https://github.com/hxm2023/GRPO-Guard)  
> 审计快照：[`d8c650e4edc8c1a9c8a856cd41eb4078ac5740aa`](https://github.com/hxm2023/GRPO-Guard/commit/d8c650e4edc8c1a9c8a856cd41eb4078ac5740aa)  
> 最近正式 Release：[`v0.3.0` / `31bb20d`](https://github.com/hxm2023/GRPO-Guard/releases/tag/v0.3.0)  
> 审查方式：README、设计文档、Git 历史、核心实现、测试、CI、冻结用例、原始 JSON 制品与 SHA-256 交叉核查；没有只按项目介绍打分  
> `review_independence: same-context-local`  
> `acceptance_status: provisional`（本次是同一模型上下文内的严格审查，不冒充独立人类/跨模型验收）

---

## 1. 结论先行

### 1.1 一句话判断

**能作为核心项目，但应把它定位为“LLM 后训练训练正确性 / RL 系统可靠性工程项目”，而不是新的 GRPO 算法项目；当前版本适合放进核心项目区，但不建议在完成 P0 整改前把它作为算法岗简历唯一的第一主项目。**

结合你的背景，最优组合仍然是：

1. GRPO reward hacking 投稿项目：证明研究问题发现、机制分析和算法实验能力；
2. Agent 场景 GRPO credit assignment 投稿项目：证明 Agent-RL、trajectory 与 credit assignment 能力；
3. GRPO-Guard：证明你能从失败事故出发，把 online rollout、token/logprob/mask/reward lineage、optimizer update、恢复和证据链真正工程化。

这三个项目拼起来，比再做一个同质化的 GRPO loss 项目更完整。GRPO-Guard 的价值不在“指标 SOTA”，而在于让面试官相信：**你不仅会改 loss 和跑 benchmark，也理解后训练系统中数据到底由谁生成、被谁消费、哪一个 policy version 产生了哪一个 token，以及训练出错后如何定位和恢复。**

### 1.2 当前是否可直接上简历

| 使用方式 | 当前建议 | 原因 |
|---|---:|---|
| 作为三个核心项目之一 | **可以** | 有真实 Qwen3-4B / TRL / vLLM / 2-GPU 闭环、代码、CI、制品和负结果，不是 PPT 项目 |
| 作为后训练算法工程岗第一主项目 | **有条件可以** | 需使用本文“当前安全版”表述，不能写 production-ready、事务原子、不可绕过 |
| 作为纯算法研究岗第一主项目 | **不建议** | 没有新的优化目标/算法，也没有可信的 held-out 能力提升；研究投稿应排在它前面 |
| 作为 RLHF/Agent-RL infra、训练可靠性岗位第一主项目 | **接近可以** | 方向高度匹配，但官方 Trainer 严格接线、事务恢复和发布治理仍有 P0 缺口 |
| 作为昇腾/NPU 计算产品线第一主项目 | **当前不够** | 仓库没有 `torch_npu`、MindSpeed-RL、HCCL、vLLM-Ascend 或真实 NPU 运行证据 |
| 宣传成“生产级后训练平台” | **不可以** | 当前是单机 2-GPU、单模型、小任务、短时验证的 production-oriented prototype |

### 1.3 综合评分

以下分数针对“2027 届大模型后训练算法/算法工程求职”，不是论文评分：

| 维度 | 当前评分 | 说明 |
|---|---:|---|
| 方向相关性 | 9/10 | policy sync、behavior logprob、mask、reward、GRPO update 都是后训练真问题 |
| 工程纵深 | 8/10 | schema、content-addressed store、event log、validator、CLI、CI、恢复、监控较完整 |
| 真实运行证据 | 7.5/10 | 有真实 Qwen3-4B/vLLM rollout 和 optimizer step；但规模、模型和任务仍窄 |
| 算法创新 | 4/10 | 主要是训练契约与可靠性系统，不是新 credit assignment 或新 policy objective |
| 实验证据严谨性 | 6.5/10 | 冻结 oracle、paired replay、负结果很好；外部效度、版本冻结和统计力度不足 |
| 生产成熟度 | 5/10 | 单机、短时、无多节点；关键“原子性”和官方 Trainer seam 尚未真正闭合 |
| 可复现/可审计性 | 7/10 | 稳定制品包和当前 CI 可复核；但 release/tag 与 main 漂移、版本目录被持续修改 |
| 面试可讲性 | 9/10 | 有事故、误判风险、架构权衡、负结果、性能和恢复，可自然展开大量追问 |
| 外部认可 | 3/10 | 仓库刚公开，暂无社区采用；TRL PR 仍 open、无 reviewer，不能当 merged contribution |

**当前总体判断：A− 级“后训练算法工程补强项目”，B 级“纯算法创新项目”。完成本文 P0 后，可提升为 A/A+ 级训练可靠性核心项目。**

---

## 2. 项目实际上做了什么

GRPO-Guard 不是另一个 TRL/verl，也不是重新实现 PPO/GRPO。它在 online post-training 的组件之间加入可验证契约：

```text
Trainer policy version / checkpoint
            ↓ real weight sync
        vLLM runtime
            ↓ server token ids + behavior logprobs
 GenerationEvent + content-addressed artifacts
            ↓
 pre-reward identity validation
            ↓
 RewardEvent / verifier identity
            ↓
 pre-update validation
            ↓
 materialized update input / single-use handle
            ↓
 GRPO loss → backward → optimizer step → checkpoint/event
```

它试图检测十类研发期静默接线错误，其中最有后训练含金量的是：

- rollout server 长期服务旧 policy；
- old/behavior logprob 绑定到错误 policy 或错误 trajectory；
- server 生成 token 被 trainer 从文本重新 tokenize；
- completion/action mask 发生 shift，选中了 prompt 或 padding；
- reward verifier、dataset split、事件顺序或 artifact 内容不一致；
- prompt 内容在相同 ID 下被替换。

项目工程面包括：Pydantic schema、不可变事件、content-addressed blob、append-only log、reason-coded validators、确定性 fault injection、paired gradient replay、CLI、GitHub Actions、Prometheus 格式 metrics、resume plan、Streamlit demo、Docker 配置和带校验和的实验制品。

这是一个合理且有辨识度的题目。后训练系统中的很多严重错误不会表现为 `loss=NaN`；训练曲线甚至可能“看起来正常”，但 rollout 来自旧 policy、mask 或 old logprob 已经接错。这个选题与两个 GRPO 研究项目连接自然，不像为了简历临时拼出的 CRUD 工程。

---

## 3. 本次实际核验到的证据

### 3.1 仓库与 CI 状态

- 当前 `main` 快照为 `d8c650e`；源码约 7,216 行 Python，测试约 2,845 行，发现 176 个 `test_*` 函数。
- 当前 GitHub Actions [run #110](https://github.com/hxm2023/GRPO-Guard/actions/runs/32761961370) 在 Python 3.11 和 3.12 两个 job 上均为 success；依赖安装、测试与 core coverage gate、F1–F10 contract checks、Day-3/v0.2 matrix、稳定制品 SHA 和 event-chain verify 均成功。
- CI 的可直接证明范围是 **core coverage ≥80%**，因为 workflow 的硬门槛是 80%。仓库文档写“87%”，但当前公开 CI 页面没有可下载的 coverage report；简历上优先写“核心模块覆盖率门槛 80%”或补交 `coverage.xml` 后再写精确 87%。
- 本次 fresh snapshot 的 core 依赖安装成功，以下命令本地复核通过：
  - `grpo-guard verify`：stable checksums + event seals/order/refs 通过；
  - F1–F4、F5–F8、F9–F10 frozen contract checks 通过；
  - Day-3 matrix：canonical 4/4、normal 32/32、boundary 4/4，PASS；
  - v0.2 matrix：12/12 matched、normal 4/4，PASS。
- 完整 `--extra test` 的本地重跑在下载 181.5 MB CPU PyTorch wheel 时遇到外部 TLS/网络中断，未进入 pytest；这不是本地测试失败。当前 commit 的 GitHub CI 双版本成功是测试状态的主要证据。

### 3.2 稳定制品与校验和

- `artifacts/v0.1.0/SHA256SUMS` 当前包含 327 个条目，本次 `sha256sum -c` **327/327 全通过**。
- `grpo-guard verify` 对指定 loop event store 的事件 seal、顺序和引用完整性通过。
- 但 `artifacts/v0.2.0-dev/SHA256SUMS` 当前 **2/3 不匹配**：

| 文件 | SHA256SUMS 记录值 | 当前文件实际值 | 状态 |
|---|---|---|---|
| `contract_check.json` | `b97d88ec...` | `f44673b1...` | FAIL |
| `fault_matrix.json` | `d2b09b21...` | `976d9694...` | FAIL |
| `fault_matrix_online.json` | 与实际一致 | 与记录一致 | PASS |

当前 CI 只验证 `artifacts/v0.1.0`，因此 GitHub CI 绿色与 dev 包校验失败并不矛盾。dev 包可以不作为 release evidence，但必须移出“已发布证据”叙事，或修复并在 CI 中显式验证。

### 3.3 真实 rollout、注入故障与 paired replay

已确认的正确表述是：

- 先从 Qwen3-4B / vLLM server 取得 **256 条真实生成 trajectory**；
- 对每条真实 trajectory 分别构造 F1–F8 八类确定性变体；
- 因而得到 2048 次**故障注入判定**，全部匹配冻结 oracle；
- 同一批 256 条正常 trajectory 为 256/256 ALLOW；
- 当前 `main` 另有 512 条真实 rollout × 8 类注入 = 4096 次判定，但该证据位于尚未重新发布的 main 快照中。

这能证明 validator 对**预定义、定向构造**的 fault family 回归测试有效；不能改写成“检测了 2048/4096 个真实线上事故”，也不能仅凭 100% 命中率推断对未知自然故障的召回率。

24 对 gradient replay 的证据也真实存在，分组统计为：

| 故障 | n | mean cosine | min | max | 正确解释 |
|---|---:|---:|---:|---:|---|
| F2 misbound logprobs | 8 | 0.927 | 0.786 | 0.998 | 数值接近时梯度可能几乎不变，说明仅看 loss/ratio 不够 |
| F3 retokenization | 8 | 0.532 | 0.250 | 0.634 | token 改写明显改变更新方向 |
| F4 mask shift | 8 | 0.192 | −0.029 | 0.431 | mask 错位可显著放大甚至翻转更新 |

但 replay 使用的是 `v1 weights + deterministic drift(seed=7, sigma=0.005)`，退化 reward group 还会按文档翻转一个值以产生非零 advantage。它是**真实 4B 模型上的机制探针 + 文档化合成扰动**，不是在线训练中自然发生的 24 个事故。

### 3.4 训练闭环和能力提升边界

仓库有两组容易混淆的 RL 训练证据：

1. `rl_training/rl_training.json`：从事件和 run log 恢复的 19-step 旧训练；首个训练内 success 28.1%，峰值 78.1%，末尾 9.4%，参数距离约 10.4。曲线明确发生后期 collapse。
2. `rl_training_final/`：P0 修订后的完整 20-step 重跑，经历 3 次 interruption/resume；制品中的 step 4–20 文件与 `full_curve.json` 共同记录整条曲线，最终参数距离约 9.53。

这两组证据可以证明：

- bounded off-policy GRPO loss 非零；
- optimizer 确实移动参数；
- update/sync/rollout/recovery 能跑多步；
- 失败和恢复过程被保留，而不是只截最好一步。

它们不能证明：

- GSM8K 或数学推理能力提升；
- 训练稳定性优于 baseline；
- Guard 会提升最终 reward。

原因是训练仅循环 8 个手写 GSM8K-style prompt；冻结的 8 个 disjoint held-out prompt 上，base 与 trained 都是 **2/8（25%→25%，delta=0）**。该 probe 也不是官方 GSM8K accuracy：它使用“抽取最后一个数字并 exact match”的确定性 verifier，部分看起来包含正确数字的回答仍可能因后续文本中的数字而判错。因此它只能支持“这个小型冻结 evaluator 下未观察到提升”，不能支持更强的泛化结论。此外，20-step run 后期 importance ratio 的 max 多次达到数十至数百，clip fraction 较高，这更像一个用于验证系统闭环和恢复的压力测试，而不是可用的算法训练 recipe。

### 3.5 上游贡献与外部认可

- [huggingface/trl PR #6876](https://github.com/huggingface/trl/pull/6876) 当前为 **Open**，改动是给 vLLM extra 增加 FastAPI 版本下界；PR 页面显示尚无 reviewer。
- 可以写“提交 upstream compatibility fix PR（open）”，不能写“合入 TRL”“被社区采用”或把一行依赖修复放成主 bullet。
- 仓库目前没有社区 adoption 证据。对校招项目这不致命，但 README 不应暗示已经过真实团队长期使用。

---

## 4. 为什么这个项目有真实求职价值

### 4.1 它补的是你现在最缺的能力，不是重复证明 research

你已经有两个 GRPO 研究项目。GRPO-Guard 新增的是一组不同的信号：

- 能从失败训练中提炼系统 invariant；
- 理解 rollout 与 trainer 的异步/版本关系；
- 知道 behavior logprob、token ids、loss mask、reward provenance 为什么必须绑定；
- 能设计 schema、状态机、content hash 和错误码，而不仅是 notebook；
- 会把失败模式做成 frozen fixture、fault injection 和 CI；
- 会保留 collapse、OOM、engine death、恢复和 held-out 不提升等负结果；
- 能说明系统校验与模型能力提升是两类不同 claim。

这正是很多“调用 TRL 跑一次 GRPO”的学生项目缺失的部分。

### 4.2 事故驱动的故事非常适合技术面

项目不是凭空想到的 observability 工具，而是来自旧 `grpo-credit-assignment` 项目的静态 rollout 事故：训练循环在更新，但 rollout service 仍可能服务旧策略；于是 iteration success 不能被当成 trained-checkpoint evaluation。把这次失败拆成 online GRPO-Guard 与 offline Credit Auditor，是一个可信的工程演进故事。

最有力量的表达不是“我做了十个 validator”，而是：

> 我最初在 Agent-RL credit assignment 项目里看到训练曲线与真实 checkpoint 行为不一致。排查后发现，仅记录 reward/loss 无法证明 rollout 来自当前 policy，也无法证明 trainer 消费的是 server 实际采样的 token。于是我把 policy version、checkpoint、sync、token、behavior logprob、mask、reward 和 optimizer input 做成可追溯契约，再用真实 vLLM rollout 上的确定性 fault injection 和 gradient replay 验证哪些错误会被挡住、哪些错误仅看数值发现不了。

这个故事同时包含失败、根因、抽象、系统实现、实验验证和边界，远强于罗列技术栈。

### 4.3 负结果反而提高可信度

仓库公开保留了多个不漂亮但重要的结果：

- on-policy/bf16 小梯度无法移动权重；
- 19-step 训练峰值后 collapse；
- 小型冻结 held-out probe 为 25%→25%，未观察到训练内曲线对应的泛化提升；
- Guard on/off 三个 seed 没有明显质量差异；
- Docker 未在原环境实际构建；
- canary 只是 behavior sketch，不是 byte-level attestation；
- 不防恶意 producer。

这些内容说明作者至少理解 claim ceiling。面试时主动讲清楚，比被追问后承认更有说服力。

---

## 5. 当前必须修的 P0 问题

下面的问题不是“可以以后再美化 README”，而是会直接影响核心项目可信度。

### P0-1：`guarded_optimizer_step` 不是事务原子，README 的强声明不成立

README 当前写道：validation、nonce、artifact、loss、backward、step、commit 在函数内“atomic”，并称任何失败都使参数可证明不变。

实际执行顺序是：

```text
verify all preconditions
    ↓
persist nonce + consume handles
    ↓
loss
    ↓
backward
    ↓
optimizer.step
    ↓
commit_fn(checkpoint + event)
```

代码没有 model/optimizer snapshot、rollback、write-ahead log 或进程隔离。因此：

- loss/backward 失败时，nonce 与 handle 已被消费，无法直接重试；
- `optimizer.step()` 部分失败时，可能已有部分参数/optimizer state 改变；
- checkpoint 或 event append 失败时，内存参数已经更新，但没有 committed checkpoint/event；
- 这最多证明**列出的 validation precondition 会在 backward 前 fail**，不能证明完整事务原子。

本次最小可执行 control-flow probe 直接复现：让 fake optimizer 成功更新，再让 `commit_fn` 抛错，结果为：

```text
commit_error = commit failed
model_value_after_commit_failure = 1
nonce_consumed_after_commit_failure = True
handle_retry = HandleConsumedError
```

整改建议分两级：

**立即修文案（当天完成）：**

- 将 “atomic / any failure leaves parameters unchanged / unbypassable” 改为：
  - “all contract preconditions are checked before backward”；
  - “supported guarded loop uses a single capability-gated update entry”；
  - “post-step failures follow crash-recovery protocol; no full in-memory rollback is currently claimed”。

**真正做成 crash-consistent（P0 工程）：**

1. 引入 `PREPARED → APPLIED → CHECKPOINTED → COMMITTED/ABORTED` update state machine；
2. 在 `PREPARED` 写入 WAL，记录输入 hashes、parent version、nonce set 和 intended output version；
3. 将 optimizer worker 与 control-plane commit 隔离：若 step/checkpoint 失败，丢弃 worker 并从 last committed checkpoint 重启，而不是继续使用不确定内存状态；
4. checkpoint 先写临时目录，校验 shards 后 `fsync + atomic rename`；
5. committed event 使用 idempotency key/unique constraint，恢复时 reconciliation；
6. 对 loss、backward、optimizer、checkpoint shard、rename、event append 每个 fault point 做故障注入测试；
7. 报告“crash consistency / exactly-once promotion”，不要轻易承诺数据库意义的参数事务原子。

### P0-2：persistent nonce registry 不是跨进程原子

`NonceRegistry` 初始化时把文本文件读入内存，`consume` 时只检查自己的 set，再 append 一行。没有 file lock、SQLite unique constraint、原子 compare-and-set 或 fsync。

本次用两个同时初始化、指向同一文件的 registry 复现：

```text
registry_1.consume("same-nonce")
registry_2.consume("same-nonce")
file = ["same-nonce", "same-nonce"]
```

因此“survive processes”只表示下一个进程重新加载文件后通常能看到旧值，不表示并发 exactly-once。

整改：改用 SQLite/LMDB 等带唯一键和事务的 registry，至少实现：

```sql
BEGIN IMMEDIATE;
INSERT INTO consumed_nonce(nonce, update_id, created_at)
VALUES (?, ?, ?);  -- nonce PRIMARY KEY / UNIQUE
COMMIT;
```

验收必须包含 `multiprocessing` 32/64 workers 同时争抢同一 nonce，且恰好一个成功；再加入 kill -9、磁盘满、重复恢复测试。

### P0-3：custom guarded path 也没有完整绑定实际 update semantics

当前 `UpdateInputEvent` 与 `guarded_optimizer_step` 对 sequence token、loss mask 和 behavior logprob 做了 artifact hash 复核，这是正确的；但真正影响 GRPO loss 的输入不止这三个数组：

- `rewards` 以普通 NumPy 数组参数传入 `materialize()`，进入 `MaterializedBatch`；它没有对应的 reward artifact/value hash，`guarded_optimizer_step` 也不检查数组是否等于所引用 `RewardEvent` 的值。相同 reward event 可以被配上不同的实际 reward tensor。
- `materialized_layout_sha256` 只写入 reward event ID，没有写入实际 reward values；甚至没有把 reward event SHA 纳入 layout 内容。
- `group_size` 在调用 `guarded_optimizer_step()` 时由 caller 另行传入，没有冻结在 validated handle/update event 中。
- batches 的顺序与 prompt/group identity 未在 `MaterializedBatch` 中绑定；重排合法 handle 后可能把不同 prompt 放入同一 GRPO group。
- actual `model`/optimizer identity 由 caller 传入；函数没有证明这个 model 就是 pre-update contract 中声明的 parent policy。
- `loss_fn` 允许 caller 注入任意实现；在绝对“不可绕过”威胁模型下，它甚至可以忽略 validated batches。
- 字段名 `single_use_nonce_sha256` 实际直接保存 caller 传入的 `nonce` 字符串，当前主 loop 传的是 `nonce-<event-id>`，并未计算 SHA-256。

这些不是恶意攻击才会触发的问题，普通 wiring bug 就可能把正确验证的 envelope 接到错误 reward/group/model 上。

整改验收标准：

1. 将 reward values 做成 content-addressed artifact，或把 canonical reward tensor hash 与 shape/dtype 写入 `UpdateInputEvent`；
2. `materialized_layout_sha256` 覆盖 sequence、mask、logprob、reward、prompt/group membership、顺序和 group size；
3. handle 内写入 expected parent policy/checkpoint identity，executor 在 backward 前核对 actual model manifest；
4. supported production API 不接受外部 `loss_fn`，测试 hook 与生产入口分离；
5. nonce 字段要么存真正的 SHA-256，要么改名为 opaque nonce，并由事务 registry 生成；
6. 增加 reward substitution、group reorder、wrong group size、wrong model object 的负向测试，且都必须在 backward 前失败。

### P0-4：官方 `GuardedGRPOTrainer` 还没有守住实际 optimizer input

这个模块是当前项目与岗位最相关、也最容易被面试官追代码的地方。现状比 README/模块 docstring 弱：

1. `_guard_pre_update(inputs)` 没有读取或 hash `inputs`，只检查已记录 completion 总长度大于 0；本次给它传入故意错误的 token/mask/logprob，方法仍无异常返回。
2. GenerationEvent 的 mask/logprob artifact 是 `sha256="0"*64` 的 placeholder；policy version 固定 0，checkpoint/tokenizer/template hash 为空，不能进入严格证据链。
3. official trainer path 没有 materialize `ValidatedBatchHandle`，也没有让真正的 optimizer step 依赖 handle/nonce。
4. Hugging Face Trainer 的 optimizer step 不在该 mixin 的 `training_step` 内；当前 override 只在调用 `super().training_step` 前做轻量检查。
5. `_guard_commit` 只计算模型 digest 放进属性；没有如 docstring 所述生成 PolicyManifest/UpdateCommitted event，而且 checkpoint save 不一定每个 optimizer step 都发生。
6. `_guard_rollouts` 持续 append，没有按 step 清空；一条 smoke 看不出多步 stale/内存累积问题。
7. 当前真实 evidence 只有 1 step、4 rollouts、一个 commit hash、0 violation；这证明 seam 能被调用，不证明 F1–F10 在 official path 被挡住。

整改验收标准：

- 从 official TRL 返回中保存真实 sequence、mask、authoritative behavior logprob artifacts；
- 在 loss/backward 前对**实际被消费**的 `input_ids`、mask、old logprobs、reward 做逐字段 hash/shape/producer 比对；
- 明确唯一 authoritative logprob source 和 precedence，禁止 service/scorer 两套同时无规则存在；
- 用 `GuardedOptimizer`/Accelerate optimizer hook 或受控 executor，让 actual `optimizer.step()` 必须持有该 step 的 validated capability；
- 每步清空/rotate rollout records，并绑定 global step/update id；
- 在 official path 实际注入 F1–F10，至少各 3 个 magnitude/variant；
- 做 20-step official trainer run，而不是只做 1-step smoke；
- 每个 step 都有完整 `UpdateInputEvent → update_applied → checkpoint_promoted → UpdateCommitted` 链。

### P0-5：release/tag 与当前 main 严重漂移，版本化 artifact 目录并不不可变

`v0.3.0` release 指向 `31bb20d`，当前 `main` 比它多 **59 个 commit**。这些并非只改文档，而包含：

- guarded optimizer 与 nonce P0 修改；
- sync state machine 修改；
- official trainer wrapper；
- 20-step P0-fixed 训练、stale-runtime、512-rollout、guard on/off、held-out 等新 evidence；
- CI 与依赖变化。

但 README 仍写“Status v0.3.0 released”，`pyproject.toml` 仍是 0.3.0。更严重的是，`artifacts/v0.1.0` 在 v0.1.0、v0.2.0、v0.3.0 之后被多次追加新文件并刷新 `SHA256SUMS`；当前 main 相比 v0.3.0 又向该目录增加约 4 万行制品。

所以当前 SHA 校验能证明“当前 commit 中这些 bytes 与当前清单一致”，不能证明 `artifacts/v0.1.0` 从 release 起保持不变。对一个主打 evidence chain 的项目，这是必须优先修的发布治理问题。

整改：

1. 把当前 main 明确标成 `v0.4.0-dev`；
2. 冻结新的 `artifacts/v0.4.0/`，禁止后续修改；新实验只能进入新 run pack/新版本；
3. 每个 release pack 使用独立 `RELEASE_MANIFEST.json`，包含 code commit、dirty flag、全部 artifact hashes、model/dataset revision、commands、hardware/env、start/end、exit status；
4. CI 遍历并校验所有标为 released 的 pack；dev pack 要么校验，要么移出 release 路径；
5. 发布 annotated/signed tag 与 GitHub release asset，不只在仓库中更新目录；
6. release 后用 CI 断言历史 pack byte-for-byte 不变；
7. 修正顶层 `run_manifest.json`：当前初始 commit 为空、`stages={}`，platform 显示 Windows/Python 3.14.3，与主要 Linux/GPU 实验来源不一致，不能承担完整 provenance 入口。

P0-1 至 P0-5 完成前，简历和 README 都不应使用“不可绕过、事务原子、生产级、每个数字均对应已发布不可变 pack”这些强声明。

---

## 6. P1/P2 重要差距

### P1-1：sync 仍是 caller-observed，不是 runtime-attested

`sync_complete` 的 398 calls 和 param digest 由调用方传入并写入 event；它比“sync 前就自报成功”进步很多，但 server 本身没有返回已加载 checkpoint 的独立 digest。canary 是 greedy token behavior sketch，可能出现不同权重给出相同输出的 false negative。

建议：

- runtime RPC 返回 load generation、参数名/shape digest、抽样 tensor hash；
- trainer/control plane 的 source digest 与 runtime digest 独立生成后比对；
- canary 同时使用 token、selected logits/top-k、hidden checksum，仍把它称为 probabilistic sketch；
- 覆盖 no-op、半数参数更新、乱序 chunk、重复请求、timeout-after-commit、worker restart 等 fault；
- 明确 retry winner、uncertain commit 和 conflict 的状态迁移。

### P1-2：完美 fault 命中主要是定向 regression，不是未知故障检测能力

当前 injected fault 由框架自身构造，oracle 与相应规则一一对应。它很好地证明规则实现没有回归，但外部效度有限。

下一步应增加：

- fault magnitude sweep：1-token/多-token mask shift、近似/远离 old logprob、lag 0/1/2/3、partial sync 比例；
- cross-version：多组 TRL/vLLM 版本和 tokenizer/chat template；
- unseen composition：两类/三类 fault 同时出现，reason code 是否稳定；
- mutation/fuzz：随机改 event DAG、shape、dtype、producer ref；
- baseline：仅看 loss/reward/NaN、普通 asserts、数据校验器分别能发现多少；
- 指标：per-family recall、normal false-reject、unknown/quarantine rate、time-to-detect、GPU-hours saved，并给置信区间。

### P1-3：系统收益还没有被量化成招聘方最关心的结果

目前最亮的数字是 validator 约 1 ms/envelope 和 fault decision 数量，但还缺“它为什么值得接入训练系统”：

- 总 step wall-time 增量百分比；
- rollout tokens/s、train tokens/s 变化；
- p50/p95/p99 validator/IO latency；
- 每万条 trajectory 的存储放大、event bytes、CPU 核占用；
- 注入真实 stale rollout 后，无 Guard 浪费多少 step/GPU-hour，Guard 多快停止并恢复；
- crash 后 RTO、RPO、重复 update 数；
- 开启 Guard 对 held-out quality 的 non-inferiority，而不是只比较训练内 reward。

建议把核心工程结果从“检测 4096 次”升级为：

> 在每 K step 注入一次 stale/retokenization/mask fault 的 5-seed 训练中，Guard 将错误 update acceptance 从 X/Y 降到 0/Y，将故障发现时间从 N steps 降到当前 step，恢复后无重复 commit；端到端吞吐损失低于预注册阈值 Z%。

这类结果比继续扩大到 1024 rollout 更有岗位价值。

### P1-4：任务、模型和系统规模仍窄

当前主要是 Qwen3-4B、手写 math/countdown、单机 2-GPU。还没有：

- 官方 GSM8K/MATH 或真实 tool-use/agent benchmark；
- Llama/Qwen 多模型、多 tokenizer；
- 长期任务、数百/数千 update；
- DDP/FSDP/ZeRO、多节点、worker elasticity；
- 多 reward source、learned RM/PRM；
- NCCL/HCCL rank 故障、distributed checkpoint；
- Ascend/NPU backend。

不需要一口气做全，但至少再打通“official dataset + official Trainer + 20/50 steps + 一类真实 agent trajectory”。

### P1-5：tool-use 仍是 deterministic toy adapter

当前 `tool_env.py` 能证明 action-only mask 与 observation 因果顺序的 schema/validator 设计，但不能写“支持真实 Agent RL 平台”。

建议选择一个可控而非过大的真实闭环：

- calculator/search/file 工具中的 2–3 类；
- trajectory 至少含 `assistant action → tool result → next action/final`；
- 注入 stale/duplicate/orphan observation、tool result 串线、action mask shift；
- 报告正常任务成功率、invalid call、fault acceptance、恢复行为；
- 把你现有 Agent credit assignment 的 step credit 作为 envelope 字段接入，但不要在没有算法证据时声称 credit 改善。

### P1-6：Guard on/off 3-seed 只能支持“初步未见劣化”

当前 3 seeds × 10 steps，文档报告 guard-on `0.698±0.049`、off `0.709±0.050`。这是训练内 success 的短 run 均值；任务仅 8 个反复使用的 prompt，且 held-out 无提升。

正确结论是“在这个小型 smoke 中未观察到明显训练内 reward 劣化”，不是统计等价。若要做 non-inferiority：

- 预注册主指标与 margin；
- 复用相同 rollout/seed 做 paired comparison；
- 至少 5 seeds，报告 paired CI/effect size；
- 主指标用独立 held-out task success，系统开销单独报告；
- 正常训练与 fault-injected 训练分成两张表。

### P2：开源产品化与外部反馈

- 增加 `CHANGELOG.md`、`CONTRIBUTING.md`、版本兼容表自动测试和 issue templates；
- 将自审稿放入 `docs/audits/`，保留“问题→commit→验证”的映射；
- 提供 10 分钟 CPU demo 与 30 分钟 GPU demo，不要求读完整设计手册；
- 做一个真实上游 adapter/plugin，而不只是一行依赖 PR；
- 找 1–2 名同学在 fresh machine 按 README 复现并提交 independent reproduction log；
- Docker 必须实际 build/run 后再写已支持；当前保持“配置已提供、原环境未验证”是正确的。

---

## 7. 实验完整性审计（A–F）

| 项 | 结论 | 说明 |
|---|---|---|
| A. Ground truth / oracle provenance | **PASS with scope** | fault oracle 是预冻结、确定性的注入规则；适合 regression，不是真实事故总体分布的 ground truth |
| B. 指标/归一化 | **未发现明显欺骗性归一化** | fault decision 和 paired gradient 原始值可查；但 100% 命中必须标明是 targeted injections |
| C. 结果存在性 | **stable pack PASS；dev pack FAIL** | v0.1.0 当前 327/327 checksum pass，CLI verify pass；v0.2.0-dev 有 2 个 checksum mismatch |
| D. 代码是否真接入 | **partial** | custom guarded loop 真正调用 guarded step；official GuardedGRPOTrainer 只做浅 seam instrumentation，未比对 actual optimizer inputs |
| E. 结果范围 | **单模型/小任务/单机** | Qwen3-4B、TRL/vLLM、2 GPU、短训练；无多节点/NPU/真实 agent benchmark |
| F. 证据分类 | **real rollout + synthetic fault injection + mechanism replay** | 不应合并描述成真实线上故障、生产部署或能力提升 |

本次审计没有发现把随机数直接伪装成 GPU 结果、把不存在文件写进报告等明显 fabrication；主要问题是**claim 强度高于实现语义、版本冻结不严和外部效度不足**。

---

## 8. 针对不同岗位，项目应该排在哪里

| 目标岗位 | 当前推荐顺序 | 面试主线 |
|---|---|---|
| 大模型后训练算法研究 | 两篇 GRPO 投稿在前，Guard 第 3 | “研究机制 + 用 Guard 保证实验链路可信” |
| 大模型后训练算法工程 | Guard 可排第 1/2，但用安全表述 | “真实 rollout/update 系统、故障定位、恢复、性能与证据” |
| RLHF/训练平台/可靠性 Infra | 完成 P0 后排第 1 | “capability-gated update + crash consistency + runtime attestation” |
| Agent-RL 算法 | credit assignment 项目在前，Guard 作系统底座 | “action/observation/credit lineage”，需补真实 tool-use |
| 评测/安全/数据质量 | Guard 很匹配 | reward provenance、split leakage、prompt mutation、audit |
| 华为昇腾/AI 框架/计算产品线 | 当前作为 CUDA 侧原型，不能冒充昇腾项目 | 补 backend abstraction、torch_npu/MindSpeed-RL/HCCL 真实证据后再前置 |

### 对你的最终建议

简历项目区不要把三者写成三个彼此独立的 GRPO 项目。应写成一条能力链：

```text
Reward Hacking Research
        ↓ 发现 reward/metric 不能等于真实行为
Agent Credit Assignment Research
        ↓ 遇到 static rollout / trajectory provenance 失败
GRPO-Guard Engineering
        ↓ 把 policy-token-logprob-mask-reward-update 做成可验证系统
```

这样面试官看到的是“围绕 Agent-RL/Post-training 持续深入”，而不是“重复做了三个相似课题”。

---

## 9. 简历怎么写

### 9.1 当前快照可安全使用的版本

**项目名：GRPO-Guard：在线 GRPO 轨迹一致性与故障注入框架｜PyTorch、TRL、vLLM、Qwen3-4B**

- 从 Agent-RL 静态 rollout 失败中抽象 policy/checkpoint、server token、behavior logprob、loss mask 与 reward lineage 契约，实现 content-addressed artifact/event store、reason-coded validator、冻结故障矩阵和 evidence-chain CLI/CI。
- 在 2×RTX 6000D 的 Qwen3-4B/vLLM 闭环中采集 256 条真实 rollout，并对 F1–F8 八类预定义接线故障逐条注入；2048/2048 次判定匹配 frozen oracle，正常轨迹 256/256 ALLOW；24 对 4B 梯度 probe 量化 retokenization 与 mask shift 对更新方向的影响。
- 打通 bounded off-policy GRPO 多步 update→checkpoint→sync→rollout 与 interruption/resume，公开当前快照内可通过 SHA-256 校验的证据包和 Python 3.11/3.12 CI；同时保留训练 collapse 与小型冻结 held-out probe 25%→25% 的负结果，不将训练内 reward 宣传为能力提升。

这版有意不写：

- production-ready；
- atomic optimizer transaction；
- unbypassable；
- 2048 个真实线上故障；
- GSM8K 28%→78% 能力提升；
- 精确 87% coverage；
- TRL PR 已合入。

### 9.2 完成 P0、发布不可变 v0.4.0 后的增强版

只有在本文 P0 验收全部通过后，才建议升级为：

- 设计 crash-consistent guarded update executor，以事务 nonce registry、WAL、原子 checkpoint promotion 和 idempotent commit event 保证 interruption/resume 下无重复 update；对 loss/backward/step/checkpoint/event 五类故障点完成 failpoint matrix。
- 将契约接入 official TRL GRPOTrainer 的真实 optimizer inputs，逐字段校验 token/mask/old-logprob/reward artifacts，并用 runtime-side parameter digest + behavior canary 验证 vLLM load；完成 20-step official-path 闭环和 F1–F10 注入。
- 发布 immutable v0.4.0 evidence pack、双版本 CI 与完整 provenance manifest，在正常训练中满足预注册 quality/throughput non-inferiority，并在 stale rollout 注入下实现 0 错误 update acceptance 和可量化 GPU-hour 节省。

注意：“0 错误 acceptance”“non-inferiority”“GPU-hour 节省”必须先有真实结果，不能把规划提前写成已完成。

### 9.3 一行版

> **GRPO-Guard**：面向 TRL+vLLM 在线 GRPO 的轨迹一致性/故障注入框架，绑定 policy-token-logprob-mask-reward-update lineage；在 Qwen3-4B 真实 rollout 上完成冻结故障矩阵、梯度影响探针、多步更新恢复和可校验证据发布。

### 9.4 技能栈不要堆满

简历只保留你能被连续追问 15 分钟的关键词：

```text
Python / PyTorch / TRL / vLLM / GRPO
content-addressed artifacts / event lineage / fault injection
checkpoint-resume / CI / profiling
```

没有真实运行证据前，不要加入 DeepSpeed、FSDP、Ray、Kubernetes、Ascend、HCCL。

---

## 10. 技术面怎么讲这个故事

### 10.1 90 秒主叙事

> 我之前做 Agent 场景 GRPO credit assignment 时遇到过一个很隐蔽的失败：trainer 在更新，但 rollout service 可能一直返回旧 policy 的 trajectory。训练日志里有 loss 和 reward，却不能证明数据是哪个 checkpoint 生成的，也不能证明 trainer 最后优化的是 server 真正采样的 token。  
>  
> 我因此做了 GRPO-Guard，把 checkpoint/sync、policy version、token ids、behavior logprob、completion mask、reward verifier 和 optimizer input 全部绑定到不可变 event/artifact，并在 reward 前和 update 前分阶段校验。然后我没有只写单测，而是在 Qwen3-4B + TRL/vLLM 的真实 rollout 上逐条注入 static policy、logprob misbinding、retokenization、mask shift 等故障，并用 paired gradient replay 量化这些错误会怎样改变更新。  
>  
> 结果说明 mask/token 错误会显著改变梯度，而 value-close 的错误 logprob 可能几乎不改变 loss，所以只看训练曲线不够。系统还跑通了多步 update/sync/rollout 和恢复。项目也暴露了边界：训练内 reward 没有在小型冻结 held-out probe 上复现，canary 只是 sketch，而且我后来发现最初的“原子更新”表述过强，因此正在把它改成 WAL + crash-consistent checkpoint promotion。这些边界我都保留在 artifacts 和 postmortem 中。

### 10.2 面试官很可能追问的问题

#### Q1：为什么 checkpoint hash 还不够？

因为 hash 只能标识一个 checkpoint 文件，不能证明 vLLM runtime 已经加载它，也不能证明某条 trajectory 是在它加载后生成的。还需要 sync lifecycle、runtime load epoch、generation event 与 checkpoint/policy version 的因果绑定。

#### Q2：为什么 old logprob 必须有 producer identity？

GRPO/PPO ratio 的分母应对应生成该 action/token 的 behavior policy。数值 shape 对上不代表语义对上；错误 policy 的 logprob 可能数值很接近，loss 看不出异常，但 estimator 已经失去正确含义。

#### Q3：为什么禁止 retokenization？

文本不是 token sequence 的可逆唯一表示。chat template、special token、normalization 或 tokenizer revision 改变时，重编码会改变 action boundary 和 mask。训练应消费 server 采样的 token ids，文本只供 reward/evaluation 查看。

#### Q4：2048/2048 到底是什么？

是 256 条真实 rollout 上，每条分别派生八类确定性 fault 的 2048 次判定，不是 2048 个自然发生的线上事故。它证明 frozen regression matrix，不证明未知故障召回率。

#### Q5：为什么 F2 gradient cosine 仍然接近 1？

因为构造的错误 logprob 数值接近 control，梯度变化小。这正说明用数值异常检测会漏掉语义 misbinding；lineage contract 的价值在于检查 producer/policy identity，而不是等到 loss 爆炸。

#### Q6：你真的保证 optimizer transaction atomic 吗？

当前不应回答“是”。正确回答：validation preconditions 能在 backward 前 fail；但 step 后 checkpoint/event 失败目前没有参数 rollback。我已经用 failpoint probe 复现，因此下一版会用 WAL、worker isolation、atomic checkpoint promotion 和 reconciliation 做 crash consistency，不滥用“事务原子”。

#### Q7：为什么 canary 不能证明权重同步？

不同权重可能在有限 greedy prompts 上输出相同 token，存在 false negative。canary 适合作为低成本行为 sketch，需要与 runtime-side parameter digest/load epoch 联合使用。

#### Q8：这个项目提升模型效果了吗？

没有可信的 held-out 提升，8 个 held-out prompt 是 25%→25%。项目证明的是训练链路正确性和故障阻断，不是新算法效果。正常路径的目标应是 non-inferiority；价值来自避免错误 update 和减少故障浪费。

#### Q9：为什么不用现有 logging/W&B？

普通 logging 记录“某个值”，但通常不绑定 producer、policy version、artifact bytes 和消费关系。GRPO-Guard 关注的是 causal lineage 与 fail-closed contract；W&B 可以作为可视化层，但不能替代 producer identity。

#### Q10：和 verl/OpenRLHF 的关系是什么？

它不是替代训练框架，而是可嵌入的 correctness/observability layer。当前先在 TRL/vLLM 验证；下一步应做 backend-neutral adapter，而不是再造调度器。

#### Q11：项目中最失败的一次是什么？

可以讲三层：旧项目 static rollout 导致评估不可信；初版 on-policy bf16 更新几乎不移动；初版把 guarded step 叫 atomic，但 failpoint 审查发现 commit failure 会留下已更新参数。重点是你如何把失败升级成 invariant、测试和发布门槛。

#### Q12：如果线上发生不确定 commit，怎么办？

不要继续在未知内存状态上训练。读取 WAL/event log，检查 checkpoint 是否完成 promotion、commit event 是否存在；若无法证明 committed，则销毁 worker，从 last committed checkpoint 恢复，并通过 unique update id/nonce 防止重复消费。

---

## 11. 建议的新架构：从“validator demo”升级为训练可靠性系统

```text
                  Runtime Attestation
             ┌────────────────────────┐
Checkpoint → │ vLLM/rollout runtime   │
 manifest    │ load epoch + param hash│
             └───────────┬────────────┘
                         ↓
     token / mask / behavior-logprob artifacts
                         ↓
              Contract Validator
                         ↓
              Validated Capability
                         ↓
┌──────────────── Transactional Update Executor ────────────────┐
│ SQLite nonce + WAL                                            │
│ PREPARED → APPLIED → CHECKPOINTED → COMMITTED                 │
│                    ↘ failure → worker discard/reconcile        │
└────────────────────────┬───────────────────────────────────────┘
                         ↓
          immutable checkpoint + event pack
                         ↓
        metrics / alerts / replay / resume audit
```

关键设计原则：

- **不要承诺做不到的内存事务回滚**，把目标定义为 crash consistency 与 exactly-once checkpoint promotion；
- **actual consumed tensors 才是 contract 边界**，不能只检查 rollout 旁路记录；
- **trainer 与 runtime 分别出具观测**，避免单方自报 sync 成功；
- **release artifact pack 不可变**，每次追加实验必须新 run/new version；
- **failure evidence 与 success evidence 同等保留**。

---

## 12. 分阶段优化路线图

### Phase 0：1–3 天，先把简历风险降下来

| 任务 | 验收标准 |
|---|---|
| 修 README 强声明 | 全库不再把 supported path 写成绝对 unbypassable/atomic；明确 precondition vs post-step failure |
| 修 release 版本 | `main` 改为 0.4.0-dev；冻结新 pack；历史 pack 不再修改 |
| 修 dev checksum | dev pack 全通过或明确排除并从 release table 移除 |
| 发布 claim matrix | 每个简历数字链接到 commit、artifact、生成命令、evidence tier |
| 补 manifest | 每个 GPU run 含 git/model/dataset/config/env/hardware/seed/command/exit |
| 当前安全版简历 | 只使用本文 9.1 的 claim ceiling |

这阶段多数是 CPU/文档工作，不需要 GPU。

### Phase 1：1–2 周，闭合真正的 P0 工程路径

| 任务 | 必须测试 |
|---|---|
| SQLite transactional nonce | 64-process same-nonce race，恰好一个成功 |
| WAL/update state machine | loss、backward、step、checkpoint、event append 五处 failpoint |
| checkpoint promotion | partial shard、磁盘满、kill -9、重复 resume |
| official Trainer actual-input check | 篡改 token/mask/logprob/reward，均在 backward/step 前失败 |
| Guarded optimizer hook | 无 capability 的 actual `optimizer.step()` 在 supported runner 中失败 |
| runtime-side attestation | no-op/partial/old checkpoint sync 均能区分 |
| multi-step official run | 20 steps，event DAG 连续，无 stale rollout/nonce duplicate |

GPU 需求可控制在 Qwen3-1.7B/4B 的 2-GPU 短 run；先证明语义闭环，不必追大模型。

### Phase 2：2–4 周，做出真正“招聘杀伤力”的结果

#### 实验 E1：故障注入下的训练生存性

- 设计：每 K step 注入 stale policy、partial sync、retokenization、mask shift；Guard on/off paired seeds；
- 主指标：bad update accepted、detection latency、rollback/recovery time、wasted GPU-hour；
- 次指标：held-out quality、throughput overhead；
- 价值：把“validator 正确”升级为“训练损失被避免”。

#### 实验 E2：正常路径 non-inferiority

- 使用 official GSM8K 或小型 MATH split；
- 5 seeds、paired rollouts、预注册 margin；
- 报告 quality delta CI、tokens/s、step time、CPU/storage overhead；
- Guard 的目标是“不明显伤害质量/吞吐”，不是凭空提升算法。

#### 实验 E3：真实 Agent trajectory

- 选择 calculator + search/mock DB 两类工具；
- action/observation/final reward 全链路；
- 注入 orphan/duplicate/stale observation 和 action mask shift；
- 与你的 credit assignment 模块接通 step credit artifact；
- 至少给一个真实多步 tool-use benchmark，而非仅 toy unit test。

#### 实验 E4：跨版本/跨模型兼容

- Qwen 1.7B/4B + 另一 tokenizer family；
- 两组 TRL/vLLM profile；
- LoRA 与 full/partial finetune 各一组；
- 输出兼容矩阵，不追求所有组合都绿，失败组合也记录原因。

### Phase 3：4–8 周，按目标公司做系统扩展

#### 如果目标是通用后训练算法工程

- 对接 verl/OpenRLHF 中一个真实 runner；
- 加 DDP/FSDP/ZeRO 下 rank-aware policy/version lineage；
- distributed checkpoint 与 worker restart；
- 长 run 的 storage compaction、sampling audit 与 p99 latency。

#### 如果目标是华为计算产品线/昇腾

- 抽象 `RuntimeAdapter`、`CheckpointAdapter`、`CollectiveSyncAttestor`；
- 增加 `torch_npu`/MindSpeed-RL 或实际可用的昇腾后训练 runner；
- 记录 HCCL rank/world size、NPU graph/precision profile、distributed checkpoint；
- 在真实 NPU 上至少完成 compatibility smoke、one-step update、sync attestation、fault matrix 和 resume；
- 做 CUDA 与 Ascend 相同 contract 的 cross-backend report。

在实际 NPU 证据产生前，简历只能写“架构预留 backend-neutral adapter”，不能写“支持昇腾”。

---

## 13. v0.4.0 Release Gate：达到后再把它升为第一核心项目

建议把以下项目全部做成机器可判定 gate：

- [ ] `main` version 与 tag/release 一致，release 后 artifact pack 不可修改；
- [ ] released pack 全部 checksum/event/provenance verify，dev pack 不混入；
- [ ] actual official Trainer token/mask/logprob/reward input 与 producer artifacts hash 一致；
- [ ] official Trainer 路径 F1–F10 全部按预注册 reason code 拒绝/隔离；
- [ ] nonce 64-process race 恰好一次成功；
- [ ] 五类 post-validation failpoint 均能恢复到唯一 last committed checkpoint；
- [ ] 无重复 update、无悬空 committed event、无 partial checkpoint promotion；
- [ ] runtime 独立 attestation 能检测 no-op、partial 和 stale sync；
- [ ] 20-step official-path run 完整，无靠手工拼接的主结果；
- [ ] normal path 5-seed quality/throughput non-inferiority 达到预注册阈值；
- [ ] fault-injected on/off 显示 bad update 与 GPU-hour waste 的可量化差异；
- [ ] coverage report 作为 CI artifact 对外可下载，简历精确数字可复核；
- [ ] Docker fresh build/run 或删除“已支持”的暗示；
- [ ] README、release notes、resume bullets 通过逐项 claim-to-artifact audit。

完成前：**核心项目可以写，但用当前安全版。**  
完成后：**可以作为后训练算法工程/训练可靠性岗位第一主项目。**

---

## 14. GitHub 首页应如何重构

当前 README 信息很多，但容易让招聘者先看到大数字，再在代码里发现边界。建议首页按下列顺序：

1. 30 秒问题定义：为什么 reward/loss 不能证明 rollout/update 正确；
2. 一张 architecture 图；
3. `Evidence tier`：
   - T0 unit/synthetic；
   - T1 real rollout + injected fault；
   - T2 custom real update loop；
   - T3 official Trainer guarded update；
   - T4 distributed/production shadow；
4. 三个最强且不误导的结果：normal acceptance、fault regression、gradient mechanism；
5. 一条冻结 evaluator 负结果：小型 held-out probe 25%→25%；
6. Quickstart；
7. Release/provenance 表；
8. Known limitations；
9. Roadmap。

把“atomic/unbypassable”替换成精确状态：

| 能力 | 当前状态 |
|---|---|
| precondition fail-before-backward | implemented + tested |
| persistent nonce across sequential restart | implemented |
| concurrent exactly-once nonce | not yet |
| post-step rollback | not claimed |
| crash-consistent checkpoint promotion | planned/in progress |
| official Trainer actual-input gating | partial |
| malicious producer resistance | out of scope |

这样的 README 反而更像成熟工程师写的，而不是削弱项目。

---

## 15. 最终建议

### 可以保留并主打的内容

- 事故驱动的问题发现；
- policy/token/logprob/mask/reward/update lineage；
- real vLLM rollout + deterministic fault injection 的严格区分；
- content-addressed artifacts、事件链、reason code、frozen fixtures；
- 24 对 gradient mechanism probe；
- 多步 update/sync/rollout 与 interruption/resume；
- CI、SHA、CLI 和负结果公开；
- “系统正确性不等于模型能力提升”的认识。

### 必须删除或降级的内容

- “任何失败参数都不变”；
- “不可绕过的唯一 optimizer entry”；
- “完整事务原子”；
- “2048/4096 个真实线上故障”；
- “GSM8K 28%→78% 提升”；
- “production-ready”；
- “v0.3 release 已包含当前 P0/P1 证据”；
- “TRL 上游贡献已被接受”。

### 最后的岗位判断

**这个项目值得做，也值得放在核心项目区。** 它对你最重要的价值不是再证明一次会 GRPO，而是把你从“论文型候选人”推向“理解真实后训练数据与系统边界的算法工程候选人”。

但它当前最危险的地方也恰好是项目主题本身：一个主打可验证证据链的仓库，不能在 optimizer 原子性、actual update-input binding、official Trainer 接线、并发 nonce 和 release artifact 不可变性上使用过强表述。把这些 P0 正面修掉，项目的可信度会比继续增加 fault 数量或 rollout 数量提升得更多。

**推荐执行顺序：先用 1–3 天收紧 claim、修版本与证据包治理；再用 1–2 周完成事务 nonce、crash-consistent executor、actual update-input binding 与 official Trainer path；最后做 fault-injected training survival + normal-path non-inferiority。到那时，它完全可以成为你申请大模型后训练算法工程岗时最强的工程核心项目。**

---

## 附录 A：本次审计的关键可复核入口

- 当前仓库：[GitHub](https://github.com/hxm2023/GRPO-Guard)
- 当前审计 commit：[`d8c650e`](https://github.com/hxm2023/GRPO-Guard/commit/d8c650e4edc8c1a9c8a856cd41eb4078ac5740aa)
- v0.3.0 Release：[`31bb20d`](https://github.com/hxm2023/GRPO-Guard/releases/tag/v0.3.0)
- 当前 CI：[run #110](https://github.com/hxm2023/GRPO-Guard/actions/runs/32761961370)
- 上游 PR：[huggingface/trl #6876](https://github.com/huggingface/trl/pull/6876)
- 重点实现：`src/grpo_guard/adapters/guarded_update.py`
- official wrapper：`src/grpo_guard/adapters/guarded_grpo_trainer.py`
- sync control：`src/grpo_guard/adapters/trl_control.py`
- 主训练 loop：`examples/countdown/rl_training_loop.py`
- 稳定制品：`artifacts/v0.1.0/`
- 小型冻结 held-out probe：`artifacts/v0.1.0/heldout/heldout_eval.json`
- P0-fixed 20-step：`artifacts/v0.1.0/rl_training_final/`
- 配套实验完整性审计：`GRPO-Guard_d8c650_EXPERIMENT_AUDIT.md`
- 机器可读审计结论：`GRPO-Guard_d8c650_EXPERIMENT_AUDIT.json`

## 附录 B：读者自测问题

只读本报告后，读者应能准确回答：

1. GRPO-Guard 是新算法、训练框架，还是 correctness/observability layer？
2. 2048 次判定中有多少条真实 rollout，多少是派生故障判定？
3. 当前能否证明 held-out 模型能力提升？
4. 为什么当前 `guarded_optimizer_step` 不能称为完整事务原子？
5. official GuardedGRPOTrainer 尚缺哪一条最关键证据？
6. stable 与 dev artifact pack 的校验状态分别是什么？
7. 为什么 v0.3.0 release 不能代表当前 main 的 P0/P1 实现？
8. 当前最安全的三条简历 bullet 是什么？
9. 哪些整改完成后项目才适合作为第一核心项目？
10. 若投华为昇腾方向，还缺哪些真实运行证据？

若读者把它总结成“GRPO 提升 28%→78% 的生产级平台”，说明项目 README 或简历仍然写错了。
