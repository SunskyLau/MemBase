# 三个 MEME 基线：正式实验准备度审计

日期：2026-09-08。范围：Full Context、Dense RAG、MD-flat。审阅为独立上下文、同模型家族的暂定判断；不等于跨模型或运行验收。

## 执行边界说明

本轮只检查并保存报告，没有修改任何基线、调用或评分实现，没有安装环境或发起付费请求。确定性检查另见 [offline_checks.md](offline_checks.md)。21项离线测试20项通过，1项历史源码字节断言失败；9组干运行通过，但这不证明真实端到端运行已通过。

实施时须保留基线语义：MD-flat 合理地不更新文件或明确没有相关记忆，可以是合法行为；不能强制每个会话都写入。需要区分这些合法结果与工具解析错误、输出截断、调用失败和轮数耗尽。对正式指标的修复应区分真正答错与技术失败计零，不修改官方成功判据或题目分母。下方为审阅原文，建议项不等于本轮已经实施。

审阅标记：`review_independence=same-family`，`acceptance_status=provisional`。本轮只读；未修改文件、未安装依赖、未启动实验、未调用付费接口。

## 总体结论：FAIL——尚不具备正式 GPT 付费运行条件

核心数据来源和主评分分母基本正确，但正式入口存在配置分叉，三个目标运行目录均无真实结果，MD-flat 会把工具失败静默当作正常输出，恢复绑定和成本/失败指标也不满足冻结方案。此结论针对准备度与公平性，不表示存在欺诈意图；文档反而明确把成绩标为待准备。

## A–F

| 检查 | 状态 | 结论 |
|---|---|---|
| A. 真值与输入泄漏 | PASS | 数据集散列、上游提交及解包内容均受校验；三种方法只向被测模型传入观察点之前的会话正文和问题文本。 |
| B. 分母与平凡通过过滤 | PASS | 正式汇总以全部 694 个变化后问题为分母；Cas/Abs/Del 仅将前后均正确记作真实通过，无自归一化。 |
| C. 结果与指标存在性 | FAIL | 三个 `runs/` 下只有历史命令日志，没有 `config.json`、`status.json`、回答、评分或 `summary.json`；不能声称已有成绩。 |
| D. 死代码/未记录指标 | FAIL | 准确率主指标可汇总，但跳数、正确变化前准确率、完整成本、失败、延迟和 Dense 嵌入指标没有形成可靠结构化产物。 |
| E. 计划范围与可运行范围 | FAIL | `core` 范围实现正确，但三个指定 Conda 环境当前不存在；直接 `experiments/*/run.sh` 还使用错误模型与不可恢复的旧 `core_01` 目录。 |
| F. 评测类型 | PASS | `real_gt`：数据集提供真值；ER 为确定性子串匹配，其余任务为模型裁判（LLM-as-a-judge），不是人工评测或模型生成代理真值。 |

## 优先发现

1. **[P0，已确认] 两套入口执行的不是同一个正式实验。** 冻结方案要求回答/内部模型为 `gpt-4o-mini`，裁判为 `gpt-4o-2024-11-20`（[experiment_results.md:20](/home/jovyan/agent-memory/MemBase/docs/experiment_results.md:20)、[experiment_results.md:42](/home/jovyan/agent-memory/MemBase/docs/experiment_results.md:42)）。三个示例入口配置正确，但直接入口全部使用 `gpt-4.1-mini`：

   - [meme_in_context/run.sh:7](/home/jovyan/agent-memory/MemBase/experiments/meme_in_context/run.sh:7)
   - [meme_dense/run.sh:8](/home/jovyan/agent-memory/MemBase/experiments/meme_dense/run.sh:8)
   - [meme_md_flat/run.sh:7](/home/jovyan/agent-memory/MemBase/experiments/meme_md_flat/run.sh:7)

   直接入口还导出空密钥；除非把密钥写进脚本，否则会覆盖已有环境值。其 `RUN_ID=core_01` 对应现有非空、无清单目录，而恢复逻辑会明确拒绝这种目录（[experiment.py:62](/home/jovyan/agent-memory/MemBase/membase/utils/experiment.py:62)）。正式运行必须指定唯一规范入口并换全新运行编号。

2. **[P0，已确认] MD-flat 会静默接受未执行工具或工具轮数耗尽。** 无工具调用时直接返回模型文本；五轮耗尽返回空文本（[md_file.py:208](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/md_file.py:208)、[md_file.py:259](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/md_file.py:259)、[md_file.py:290](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/md_file.py:290)）。摄入阶段不验证是否实际写入记忆，检索空输出则静默变成 `(no relevant facts)`（[md_file.py:298](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/md_file.py:298)、[md_file.py:317](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/md_file.py:317)）。这会把协议失败当成方法错误，系统性压低 MD-flat。

3. **[P0，已确认] 当前没有任何正式结果。** 汇总只会在所有答案和评分验证通过后写入（[meme.py:84](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:84)、[meme.py:126](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:126)、[meme.py:141](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:141)）。目标目录扫描未发现这些产物；登记表也明确三项均“待准备”（[experiment_results.md:132](/home/jovyan/agent-memory/MemBase/docs/experiment_results.md:132)）。

4. **[P1，已确认] 参数清单与实际调用不一致。** 通用入口把 `temperature=0.7` 写入运行配置并接受 `seed`、`parallel_jobs`（[run_benchmark.py:33](/home/jovyan/agent-memory/MemBase/scripts/run_benchmark.py:33)、[run_benchmark.py:41](/home/jovyan/agent-memory/MemBase/scripts/run_benchmark.py:41)、[run_benchmark.py:90](/home/jovyan/agent-memory/MemBase/scripts/run_benchmark.py:90)），但 MEME 原生命令未传这三项（[meme.py:56](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:56)）；实际回答和 MD 内部调用硬编码温度 0。示例传入的 `SEED=0` 对这三条原生路径无效且不进入恢复绑定。

5. **[P1，已确认] 恢复机制局部可靠，但版本绑定不完整。** 优点是运行配置必须完全一致，评分收据绑定回答与评分文件散列（[experiment.py:62](/home/jovyan/agent-memory/MemBase/membase/utils/experiment.py:62)、[meme.py:97](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:97)、[meme.py:126](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:126)）。缺口是协议只记录上游提交、数据散列和传输适配器散列，没有绑定本地运行器、验证器和汇总代码（[meme.py:159](/home/jovyan/agent-memory/MemBase/membase/runners/meme.py:159)）；同一运行编号可能跨本地评分代码版本续跑。回答验证也不要求预算、延迟、结束原因等字段存在（[meme.py:37](/home/jovyan/agent-memory/MemBase/membase/evaluation/meme.py:37)）。

6. **[P1，已确认] 裁判可把不完整 JSON 静默计为错误答案。** `{"correct": ...}` 缺失时默认 `False`，理由可为空；Agg 缺少 `results` 时也默认全错（[judge.py:238](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/judge.py:238)、[judge.py:269](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/judge.py:269)）。本地验证仅拒绝字面值 `"missing"`，因此空理由仍被接受（[meme.py:55](/home/jovyan/agent-memory/MemBase/membase/evaluation/meme.py:55)）。这是保守但不公平的评分失败处理。

7. **[P1，已确认] 变化前准确率现有总数不可直接使用。** 裁判跳过全部变化前 ER 问题，却把它们重组为 `u_pass=False` 并计入 `before_total`（[judge.py:317](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/judge.py:317)、[judge.py:354](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/judge.py:354)、[judge.py:403](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/judge.py:403)）。原始 `filler32k` 有 100 个此类问题，因此必须离线重新做确定性 ER 匹配，不能汇报现有 `totals.before_pass/before_total`。

## 截断与参数来源

- Full Context 确实拼接到对应观察点的全部会话，没有代码级输入裁剪（[in_context_baseline.py:43](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/in_context_baseline.py:43)、[in_context_baseline.py:78](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/in_context_baseline.py:78)）。本地原始数据声明每情节约 33,306–37,044 词元；但没有服务端窗口预检或实际输入长度断言。
- Dense 使用官方代码默认的 4,096 词元分块；超长单轮再按 8,000/400 滑窗处理，并取 top-k=5（[_chunking.py:18](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/_chunking.py:18)、[dense_memory.py:27](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/dense_memory.py:27)、[dense_memory.py:44](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/dense_memory.py:44)）。top-k 和嵌入模型与冻结方案一致；4,096/8,000/400 是固定上游代码默认，并非数据集 README 给出的论文推荐。
- 三种最终回答都限制为 500 输出词元；MD 工具轮每次限制 2,000、最多五轮（[in_context_baseline.py:55](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/in_context_baseline.py:55)、[base.py:101](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/base.py:101)、[md_file.py:172](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/md_file.py:172)）。所有路径均未保存或校验 `finish_reason`；非空的截断回答/工具参数可能被接受。
- Full Context 与 Dense/MD-flat 的回答提示并不完全相同（[in_context_baseline.py:29](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/in_context_baseline.py:29)、[base.py:23](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/agents/base.py:23)）。这是上游实现差异，但对 Absence/Deletion 的拒答倾向可能有影响，应在正式协议中披露。

## 必需指标覆盖

| 指标 | 覆盖状态 |
|---|---|
| 六类、总体、领域准确率 | **已保存（设计上）**：`summary.json` 含 `by_task`、`by_domain`、总体；当前尚无实际文件（[meme.py:103](/home/jovyan/agent-memory/MemBase/membase/evaluation/meme.py:103)）。 |
| 平凡通过明细 | **已保存/可离线汇总**：逐情节评分含 `pass_type` 与 `trivial_analysis`。 |
| 1/2 跳 | **可离线计算**：`hop` 随问题行保留；汇总器未生成 `by_hop`。 |
| 变化前准确率 | **可离线重算**：原回答和真值保留；现有 `totals.before_*` 因 ER 跳过而错误。 |
| 完成情节/问题数 | **已保存（设计上）**：`episodes`、`total`；当前无正式产物。 |
| Full Context 调用/词元 | **部分保存但标签错误**：预算追踪器默认作用域为 ingest，而该入口从未切换到 answer（[budget_tracker.py:60](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/budget_tracker.py:60)、[in_context_baseline.py:84](/home/jovyan/agent-memory/MemBase/external/MEME-public/code/eval/in_context_baseline.py:84)）。 |
| Dense 回答调用/词元 | **已部分保存**；嵌入调用、嵌入词元和嵌入成本缺失。 |
| MD-flat 调用/词元 | **已部分保存**：ingest/retrieve/answer 作用域可用；失败请求不计，检索工具轨迹被丢弃。 |
| 工具调用 | **部分可离线计算**：MD 摄入轨迹保存；检索 `read_memory` 轨迹未保存。 |
| 查询 P50/P95 | Dense/MD 的 `answer_time_sec` 可离线计算“检索＋回答”总延迟；Full Context **缺失埋点**，且均无法拆分检索与回答延迟。 |
| 累计阶段耗时 | Dense/MD 可从逐会话/逐问题时间近似相加；Full Context 和结构化裁判耗时 **缺失**。 |
| 裁判调用/词元 | **部分保存**：只计成功调用；重试失败成本不计，并发计数器无显式锁。 |
| 技术失败数、阶段、请求编号 | **缺失**：当前原生路径遇硬失败会中止；`allow_failures` 分支没有启用。MD 的空工具回退甚至不被标成失败。 |
| 实际货币成本 | **缺失**：无价格快照或成本汇总；Dense 嵌入和失败尝试使现有词元也不足以可靠反推。 |
| 输入长度、输出截断 | Dense/MD 的检索文本可从逐题输出离线分词；Full Context 可从原始数据重建。三者均无结构化长度/截断标志。 |

## 正式运行前最低门槛

1. 只保留一个规范入口，使用全新运行编号，统一为冻结模型；密钥只从环境读取。
2. 准备并验证三个硬编码 Conda 环境。
3. MD-flat 对每次摄入/检索强制验证成功工具调用；空响应、参数解析失败、轮数耗尽均记技术失败。
4. 对裁判 JSON 做严格模式校验；失败重试计成本，最终失败计零并保留原因。
5. 修正温度/种子记录，或删除无效参数；把本地运行器和评分器散列加入恢复绑定。
6. 补齐 Dense 嵌入用量、Full Context 逐题延迟、跳数、技术失败及截断字段，再进行独立短场景验收。

未检查范围：其他基线、论文外部文献与公开成绩、真实服务商路由/上下文窗口/用量返回、运行时依赖行为及任何动态模型输出。由于未启动实验，真实截断率、失败率、延迟、费用和裁判格式稳定性均为未知。
