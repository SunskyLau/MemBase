# A-MEM 与 OurMem 的 MAB 核心对照

本目录复用 MemBase 内置 A-MEM，回补官方记忆标识和邻居更新修复；不使用 OurMem 的抽取、派生或读取规划。实现差异见 [来源说明](../../membase/baselines/amem/UPSTREAM.md)。

两边都处理 6k、32k 单跳（single-hop）与多跳（multi-hop），四组共 400 题。构建和回答模型均为 `gpt-4.1-mini`，嵌入为 `text-embedding-3-small`；回答温度为 0.7，上限为官方 10 个词元。A-MEM 保留内部温度 0.7、输出上限 1,000、`TOP_K=10`、演化刷新阈值 100。最终证据上限均为 8,000 个词元，但实际用量与总计算成本不相等。

无密钥示例是 `run.example.sh`，本地 `run.sh` 已被忽略。所有普通参数位于脚本顶部。不要把更换了模型配置的结果称为原论文配置的逐项复现。

## 离线检查

在 `/home/jovyan/agent-memory/MemBase` 执行：

```bash
/home/jovyan/my-conda-envs/membase-amem/bin/python -m unittest tests.benchmarks.test_amem_mab -b -q
```

这会使用真实本地 Chroma 和模拟接口验证保存加载、向量刷新及公共三阶段，不调用付费模型。

在 `/home/jovyan/agent-memory` 执行：

```bash
./MemBase/examples/evaluate_amem_on_memoryagentbench/run.sh --dry-run
./MemBase/examples/evaluate_ourmem_on_memoryagentbench/run.sh --dry-run
```

应分别显示 `amem` 和 `ourmem`，均为 `core`、4 组、400 题、各自并发 4；两边同时运行时合计最多处理 8 个独立样本。干运行不创建记忆或启动实验。帮助可用 `python MemBase/scripts/run_benchmark.py --help` 查看。

## 手动并行启动

完成验证后，在两个终端分别启动，不要在运行中修改公共代码：

```bash
# 终端一，在 agent-memory 目录
./MemBase/examples/evaluate_ourmem_on_memoryagentbench/run.sh
```

```bash
# 终端二，在 agent-memory 目录
./MemBase/examples/evaluate_amem_on_memoryagentbench/run.sh
```

默认运行名称分别为 `mab_core_ourmem_02`、`mab_core_amem_02`，输出位于各自的 `experiments/ourmem_memoryagentbench/runs/` 和 `experiments/amem_memoryagentbench/runs/`。OurMem 的新名称用于完整请求预算修复后的重建，A-MEM 的新名称用于 Chroma 并发初始化修复后的重建；旧目录均保留原样。核心实验默认不限制请求总数，会产生费用；并发数可在首次启动前调整。

也可用本目录的 `run_construction.sh`、`run_search.sh`、`run_evaluation.sh` 单独执行阶段，它们读取同一份本地配置。

## 进度、恢复与成绩

A-MEM 每 8 条输入及阶段结束保存完整检查点，笔记与索引各自状态分别保留。通常最多重做最近 7 条；若写第 8 条检查点时中断，最多重做未提交的 8 条。加载直接恢复已有向量，不重做分析或嵌入；已发生的费用不因中断清零。不提供跨运行缓存，不迁移旧产物。

原始材料无日期时保持未知。检索结果标记为 `not_assessed`，表示普通检索，没有伪造语义可用性判断。原生分析或演化回退会记录警告；单题检索、回答或评分的可恢复模型失败按零分计入原定分母，不伪造回答，也不在同一运行反复尝试已计零题。凭据、预算、存储与程序错误仍明确停止。

请同时检查 `status.json`、`summary.json`、请求成本和技术失败数。`complete_with_warnings` 表示流程已结束但有技术失败或记忆回退；只有完整的 400 题终态才可作全量汇总。对比应逐组报告准确率、失败数以及构建/检索/回答成本。

## 小额真实验证

专用脚本 `scripts/validate_amem_mab.py` 仅在显式传入 `--real` 时调用接口。所有尝试共用 `experiments/mab_comparison_validation/request_budget.sqlite`，总上限为 20 次语言模型请求、20 次嵌入请求（包含重试），不能通过更换尝试名称扩额。它使用独立短场景，不是 MAB 正式结果；验证记录保存在同目录的报告中。
