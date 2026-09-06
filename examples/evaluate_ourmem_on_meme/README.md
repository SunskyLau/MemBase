# evaluate_ourmem_on_meme

本示例使用 MemBase 的构建、检索、评测三个公共运行器。先复制 `run.example.sh` 为本目录的 `run.sh`，在其顶部填写配置和密钥，然后执行 `./run.sh --dry-run` 检查范围，再执行 `./run.sh`。脚本可从任意工作目录调用。

也可分别执行 `./run_construction.sh`、`./run_search.sh`、`./run_evaluation.sh`，三个阶段读取同一份 `run.sh` 配置。MEME 会在构建过程的观察点调用共用检索和回答组件，最后统一评分；不是先建完全部历史再补生成早期答案。

输出位于 `experiments/ourmem_meme/runs/<RUN_ID>/`。相同新格式运行可续跑；更改配置须换名。旧运行保持只读，不自动迁移，不提供跨运行记忆缓存。默认请求总数没有上限，正式运行会产生费用。

`MODE=smoke` 取无填充版本两个领域各一个样本；`core` 先完成这两个样本，再运行 32k 全部样本；`full` 覆盖无填充、32k、128k 三个版本。各模式均保留变化前后提问、六类官方评分及平凡通过过滤（trivial-pass filtering）。更多说明见 [统一实验流程](../README_experiments.md)。
