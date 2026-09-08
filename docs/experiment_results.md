# 核心实验结果

记录已完成的实验，供后续写作引用。每个方法版本单独保留一行，不用新结果覆盖旧结果；未完成实验不填写最终成绩。记录更新日期：2026-09-07。

## MemoryAgentBench：6k 冲突消解

评估范围为 6k 单跳（single-hop）和多跳（multi-hop）两个子集，各 100 题，共 200 题。指标为官方子串精确匹配（substring exact match）；表中百分比按固定题数计算，技术失败计零、不缩小分母。

| 方法与运行版本 | 构建／回答模型 | 单跳（single-hop），100 题 | 多跳（multi-hop），100 题 | 总体，200 题 | 备注 |
|---|---|---:|---:|---:|---|
| [MemBase A-MEM](/home/jovyan/agent-memory/MemBase/experiments/amem_memoryagentbench/runs/mab_6k_amem_4o_mini_01/summary.json) | gpt-4o-mini | 77/100（77%） | 9/100（9%） | 86/200（43%） | 199 题有回答及评分；多跳第 61 题连续超时，计零 |
| [OurMem：原读取版本](/home/jovyan/agent-memory/MemBase/experiments/ourmem_memoryagentbench/runs/mab_6k_ourmem_4o_mini_01/summary.json) | gpt-4o-mini | 95/100（95%） | 18/100（18%） | 113/200（56.5%） | 200 题有回答及评分；两个样本均有记忆维护告警 |
| [OurMem：关系链读取版本](/home/jovyan/agent-memory/MemBase/experiments/ourmem_memoryagentbench/runs/mab_6k_ourmem_4o_mini_readchain_01/summary.json) | gpt-4o-mini | 92/100（92%） | 25/100（25%） | 117/200（58.5%） | 复用上一行的构建；200 题有回答及评分，原维护告警保留 |

共同配置：嵌入模型（embedding model）为 `text-embedding-3-small`，回答温度为 `0.7`，随机种子为 `0`，官方回答上限为 10 个词元（tokens），最终证据上限为 8,000 个词元（tokens）。A-MEM 的检索数量（top-k）为 10；两版 OurMem 使用各自冻结的分阶段读取策略，不能据此声称检索数量或总计算量相同。

写作边界：以上是单次运行结果，不是多随机种子的均值。OurMem 的这批题目已用于错误诊断，应标为开发对照，不宣称稳定、显著或未见数据上的提升。这里的“关系链读取版本”不是后续讨论的简化混合检索（hybrid retrieval）版本。

方法名称链接到本地原始汇总；同目录的 `config.json`、`read_revision.json`（如有）、逐题答案与评分用于追溯实际配置和代码版本。原始产物位于 Git 忽略的 `experiments/`，论文归档时需另行保存，不能只保留本表。
