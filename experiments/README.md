# 实验运行

每个“数据集＋基线”对应一个目录。打开其中的 `run.sh`，在顶部填写 `CONDA_ENV` 和 `OPENAI_API_KEY`，按需修改接口、模型、`MODE` 和并发数，然后在该目录直接执行：

```bash
./run.sh
```

`MODE` 可选 `smoke`、`core`、`full`；将 `DRY_RUN` 改为 `1` 可只预览命令，也可执行 `./run.sh --dry-run`。预览只使用当前 Python 的标准库，不创建结果或调用模型。正式执行使用指定的已有 Conda 环境，无需在终端提前激活或导出变量。

日志和结果保存在本实验的 `runs/<RUN_ID>/`。相同配置续跑时保持 `RUN_ID`，更换模式、模型或其他实验参数时修改该名称；留空时按 UTC 时间创建目录。Python 运行器核对配置一致性，只复用完整回答；MEME 同时校验回答和评分的哈希，仅重评新增或已变化的回答。旧 Shell 版本产生的运行目录保留，首次使用新版时需选择新的 `RUN_ID`。

本地 `run.sh` 已被忽略，可以直接填写密钥。`run.example.sh` 是无密钥模板；重新克隆项目时，先在实验目录执行 `cp run.example.sh run.sh`，再填写配置。

Python 入口位于 `scripts/run_benchmark.py`；数据逻辑位于 `membase/datasets/`，运行编排位于 `membase/runners/`，评分检查与汇总位于 `membase/evaluation/`，文件与进程工具位于 `membase/utils/`。官方基线和评分程序固定在 `external/`；数据仍保存在 `data/`。

数据准备与检查入口（从仓库根目录执行）：

```bash
python scripts/prepare_benchmarks.py --benchmark all
python scripts/prepare_benchmarks.py --benchmark all --check-only
```

首次解析 MAB Parquet 需要已有环境中的 `pyarrow`，准备成功后生成的问题清单可由标准库读取。其他真实运行依赖由用户安装；脚本不会创建环境或安装包。

| 模式 | MemoryAgentBench | MEME |
| --- | --- | --- |
| smoke | 6k 单跳前 4 问 | 无填充版本两个领域各 1 个样本 |
| core | 6k、32k 单跳/多跳，共 400 问 | 小规模检查后，32k 全量 100 个样本 |
| full | 四种长度单跳/多跳，共 800 问 | 无填充、32k、128k，共 240 个样本 |

运行器以本地数据的样本和问题清单核对产物。缺失、重复、损坏或评分未完成时返回非零退出码，`status.json` 不会标为完成；成功时写入 `summary.json`。MAB 汇总官方子串精确匹配（substring exact match）分数；MEME 对级联、缺失和删除只计官方 `real` 通过，其余任务沿用官方 `u_pass`。
