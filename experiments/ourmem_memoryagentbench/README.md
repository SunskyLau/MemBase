# OurMem — memoryagentbench

在本目录填写 `run.sh` 顶部配置后运行：

```bash
./run.sh --dry-run
./run.sh
```

新克隆的项目先把无密钥的 `run.example.sh` 复制为 `run.sh`。运行脚本可从任意工作目录调用。环境使用 `membase-ourmem`；数据与官方代码准备入口为项目根目录的 `python scripts/prepare_benchmarks.py --benchmark memoryagentbench`，加 `--check-only` 只核对。

只评估冲突消解（Conflict Resolution）。本项目主要评估范围为 `MODE="core"`：6k、32k 的单跳（single-hop）与多跳（multi-hop），共四个完整子集、400 题。`smoke` 仅用于检查流程，使用 6k 单跳（single-hop）的完整输入及前 4 题；`full` 额外包含 64k、262k，共八组 800 题。这些模式是本项目的运行约定，不能将 `core` 称为 MAB 官方主实验范围。输入保留公开事实序号；采用官方提示与子串精确匹配（substring exact match）评分，不调用模型评判。

方法参数默认沿用 V5（最多 5 层派生）。`MEMORY_CONFIG` 可指定不含凭据的参数 JSON；不要给 OurMem 设置基线通用的单一检索数量。多个独立样本可通过 `WORKERS` 并发，同一样本始终顺序写入。

每次运行放在 `runs/<RUN_ID>/`。相同数据、代码、模型及方法配置可以续跑；配置改变须使用新名称。结果包括配置及来源映射、每样本 SQLite、阶段快照、逐题回答与检索证据、官方评分、调用日志和成本。技术失败及缺项会返回非零退出码，不把部分结果汇报为全量完成。

受限验证可同时设置 `MAX_LLM_REQUESTS=100`、`MAX_EMBEDDING_REQUESTS=20`，需要多个实验共用上限时设置相同的 `BUDGET_LEDGER` 绝对路径。实际请求、失败和重试均计数；任一种请求达到上限后停止外发，不自动追加额度。

这些入口已做离线验证，不能据此声称正式实验已经完成。历史实验结果不代表 V5 的效果。
