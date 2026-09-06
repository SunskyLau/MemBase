# evaluate_ourmem_on_memoryagentbench

本示例使用 MemBase 的构建、检索、评测三个公共运行器。先复制 `run.example.sh` 为本目录的 `run.sh`，在其顶部填写配置和密钥，然后执行 `./run.sh --dry-run` 检查范围，再执行 `./run.sh`。脚本可从任意工作目录调用。

也可分别执行 `./run_construction.sh`、`./run_search.sh`、`./run_evaluation.sh`，三个阶段读取同一份 `run.sh` 配置。

输出位于 `experiments/ourmem_memoryagentbench/runs/<RUN_ID>/`。相同新格式运行可续跑；更改配置须换名。旧运行保持只读，不自动迁移，不提供跨运行记忆缓存。默认请求总数没有上限，正式运行会产生费用。

`MODE=core` 为 6k、32k 的单跳（single-hop）和多跳（multi-hop），四组共 400 题；`smoke` 取 6k 单跳（single-hop）前 4 题，保留完整历史；`full` 再加入 64k、262k。更多说明见 [统一实验流程](../README_experiments.md)。
