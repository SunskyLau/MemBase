# evaluate_long_context_on_memoryagentbench

本示例使用 固定版本的官方基线实现。先复制 `run.example.sh` 为本目录的 `run.sh`，在其顶部填写配置和密钥，然后执行 `./run.sh --dry-run` 检查范围，再执行 `./run.sh`。脚本可从任意工作目录调用。

只统一启动、进度和结果校验，不替换官方方法的分块、检索或回答逻辑。

输出位于 `experiments/memoryagentbench_long_context/runs/<RUN_ID>/`。相同新格式运行可续跑；更改配置须换名。旧运行保持只读，不自动迁移，不提供跨运行记忆缓存。默认请求总数没有上限，正式运行会产生费用。

`MODE=core` 的 MAB 范围为 6k、32k 的单跳和多跳，共 400 题；`smoke` 仅限制问题数而保留对应完整历史。MEME 沿用无填充、32k、128k 的既定模式范围。更多说明见 [统一实验流程](../README_experiments.md)。
