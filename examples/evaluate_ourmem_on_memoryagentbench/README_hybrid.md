# 6k 混合检索实验

从任意目录执行：

```bash
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k_hybrid.sh --dry-run
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k_hybrid.sh
```

第一条为干运行（dry run），不调用模型；第二条从 `envs/.env` 读取凭据并开始付费实验。不要同时启动相同运行名。已有部分产物时，同一配置可继续执行第二条命令；运行期间不要修改方法代码。

默认使用 `membase-ourmem` 环境、`gpt-4o-mini` 和 `text-embedding-3-small`，样本并发为 2。范围是 6k 单跳（single-hop）与多跳（multi-hop）两个完整子集，各 456 条输入、100 个问题。回答温度 0.7、输出上限 10 个词元（token），保持官方提示和评分不变。

查询仅做一次稠密与 BM25 混合检索（hybrid retrieval），按记忆条目融合前 20 项；支持展开不再受 20 条限制。最终证据上限为 8,000 个词元（token），包含必要来源并区分历史值与当前值。没有查询规划或额外读取判断模型。

本次写入、协调和归纳也发生变化，因此从原始输入重新构建，不复用旧实验的数据库。已完成的 `hybrid_02` 保存在 `experiments/ourmem_memoryagentbench/runs/mab_6k_ourmem_4o_mini_hybrid_02/`，单跳（single-hop）89%、多跳（multi-hop）22%，技术失败题为 0，仍有记忆维护警告。`hybrid_01` 在首批健康检查中发现误归并，已停止并保留全部记录；不用于报告准确率。

实验结束后又修正了“矛盾分项判断静默改成独立新增”的问题，上面的分数不是该修正后的效果。当前入口在脚本内通过 `experiment_model="gpt"` 或 `experiment_model="qwen"` 选择模型，分别使用新运行名 `mab_6k_ourmem_hybrid_04_gpt` 和 `mab_6k_ourmem_hybrid_04_qwen`；本轮未启动实验。模型与接口配置见[统一说明](../README_model_profiles.md)，历史结果见[结果与结论](../../experiments/ourmem_memoryagentbench/analysis/hybrid_comparison_01/结果与结论.md)。

快速离线检查：

```bash
cd /home/jovyan/agent-memory/MemBase
/home/jovyan/my-conda-envs/membase-ourmem/bin/python scripts/validate_ourmem_core.py
```

仅运行十个合成核心场景，不调用接口。单跳（single-hop）和多跳（multi-hop）的实际效果由上述完整实验判断；与旧结果比较时，应标明这是整体改动，不能将全部差异归因于检索简化。
