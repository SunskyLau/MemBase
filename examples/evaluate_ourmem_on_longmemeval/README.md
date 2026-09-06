# evaluate_ourmem_on_longmemeval

本示例使用 MemBase 的构建、检索、评测三个公共运行器。先复制 `run.example.sh` 为本目录的 `run.sh`，在其顶部填写配置和密钥，然后执行 `./run.sh --dry-run` 检查范围，再执行 `./run.sh`。脚本可从任意工作目录调用。

也可分别执行 `./run_construction.sh`、`./run_search.sh`、`./run_evaluation.sh`，三个阶段读取同一份 `run.sh` 配置。

输出位于 `experiments/ourmem_longmemeval/runs/<RUN_ID>/`。相同新格式运行可续跑；更改配置须换名。旧运行保持只读，不自动迁移，不提供跨运行记忆缓存。默认请求总数没有上限，正式运行会产生费用。

`MODE=core` 和 `full` 均评测 LongMemEval-S 清洗版本的全部样本；`smoke` 只取第一个样本及其问题，保留完整历史。使用官方判分提示，报告总体和分类结果。更多说明见 [统一实验流程](../README_experiments.md)。
