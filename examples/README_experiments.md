# 统一实验流程

OurMem 四个数据集复用 MemBase 原有的构建、检索和评测运行器。原始数据与固定版本官方仓库位置不变。五层派生、SQLite、时间与删除语义保持 V5 约定；自定义提示会随实现修复更新，官方回答与评分提示保持原样。

## 入口

| 示例目录 | 执行方式 |
| --- | --- |
| `evaluate_ourmem_on_locomo` | 公共三阶段 |
| `evaluate_ourmem_on_longmemeval` | 公共三阶段 |
| `evaluate_ourmem_on_memoryagentbench` | 公共三阶段 |
| `evaluate_ourmem_on_meme` | 公共三阶段，构建时包含在线观察点回答 |
| `evaluate_amem_on_memoryagentbench` | 公共三阶段，A-MEM 原生笔记与检索 |
| `evaluate_long_context_on_memoryagentbench`、`evaluate_bm25_on_memoryagentbench` | 官方原生基线执行 |
| `evaluate_in_context_on_meme`、`evaluate_bm25_on_meme`、`evaluate_dense_on_meme`、`evaluate_md_flat_on_meme` | 官方原生基线执行 |

在选定示例目录复制 `run.example.sh` 为本地 `run.sh`，编辑顶部配置。密钥只放本地脚本或环境变量中。先运行 `./run.sh --dry-run`，确认范围后再手动运行 `./run.sh`；正式运行会产生费用。

OurMem 也可分别执行 `./run_construction.sh`、`./run_search.sh`、`./run_evaluation.sh`。它们共用同一份 `run.sh`，调用根目录对应的 `memory_*.py --protocol official`。从其他目录启动时使用脚本路径即可，不需要事先激活环境。

三个根入口原有的默认参数与 MemBase 原生示例保持兼容。新的官方协议既支持 `--benchmark`，也支持原风格的 `--dataset-type`、`--memory-type OurMem` 或 `A-MEM`、`--dataset-path`、`--config-path` 和 `--num-workers`。A-MEM 本轮仅接入 MAB。已有新格式运行可用 `--protocol official --run-dir <运行目录>` 单独执行一个阶段，并读取冻结配置；接口密钥仍由环境提供。

## 三阶段的职责

普通数据集先构建并保存记忆，再检索并保存完整证据，最后生成答案和评分。检索结果含公共 `QuestionAnswerPair`、`MemoryEntry` 和快照审计；改变读写流程不改变官方输入与评分定义。

MEME 在构建过程中按观察点执行：摄入到指定会话 → 刷新 → 保存快照 → 调用共用检索与问答组件 → 保存回答或技术失败终态 → 继续摄入。后续检索阶段读取已记录结果，评测阶段只评分。变化前失败不能在变化后补答，问题和答案不会写回记忆。

MAB 的 `core` 仍为 6k、32k 的单跳（single-hop）与多跳（multi-hop），四组共 400 题；`smoke` 使用完整 6k 单跳（single-hop）历史和前 4 题；`full` 再加入 64k、262k。四组保持独立构建，不自动合并同长度历史。其他数据集沿用原有模式范围。

OurMem 与 A-MEM 的核心对照入口已统一为各自并发 4、相同回答模型/温度/输出上限和 8,000 个词元的证据上限。A-MEM 保留自己的分析、演化及查询逻辑，不使用 OurMem 的推理组件。具体配置、检查点频率和两个终端启动方式见 [A-MEM 对照说明](evaluate_amem_on_memoryagentbench/README.md)。

## 输出、进度与恢复

输出仍在 `experiments/<数据集和方法>/runs/<RUN_ID>/`，包括冻结配置、数据与源码指纹、请求账本、SQLite、会话检查点、各阶段完成记录、快照、检索结果、回答与评分。`PROGRESS_INTERVAL` 控制只读进度刷新，终端输出保存在 `console.log`。

必须同时满足输入、方法、源码与完整性记录匹配才能复用阶段，数据库文件存在本身不代表构建完成。同一新格式运行的中断可恢复，已提交事实不从零重做。当前不提供跨运行缓存；旧 V5 运行保持原样，不迁移或覆盖，请为新流程选择新的 `RUN_ID`。

`Ctrl+C` 停止本次进程组并保留已提交产物；未完成请求可能已经计费。OurMem 各阶段共用同一持久化预算，默认不设总次数上限，设置上限后不会因切换阶段而重置。原生基线继续使用官方调用和成本记录。

官方原生基线只统一配置、启动、日志与产物校验，不换成 MemBase 同名但行为不同的方法。没有独立保存接口的基线不伪装成三个独立可恢复阶段。

## 局部失败与成绩含义

OurMem 的模型调用最多额外重试两次。新主张派生、依赖验证及派生协调仍失败时，不保存无效关系，利用现有处理进度登记未完成范围；已确认受影响且没有其他支持的旧结论继续暂停，不阻断无关事实。相同任务不会因刷新或续跑再次生成，本版本不增加后台补救轮次。

某题的检索、回答或评分发生可恢复模型故障时，将原因、请求标识和输入绑定记录在相应观察点的 `failures/` 中，继续其他题，并按 0 分计入原定分母。没有回答时保留空值，不伪造“我不知道”；没有官方评分时保留缺失值，以 `score_origin=technical_failure` 明确区分技术失败计零和模型答错。MEME 的变化前技术失败不能使依赖前后回答的任务获得真实通过。

`summary.json` 报告预期题数、实际回答和评分数、分阶段技术失败数及记忆维护缺口。MEME 的预期题数包括前后观察点，主成绩的 `total` 仍只统计官方规定的变化后题目。所有题均有终态但存在故障或维护缺口时，状态是 `complete_with_warnings`，进程正常结束；这是含失败计零的端到端成绩，不是完整无故障的官方评判结果。同一运行不重试已计零题，避免通过不断重跑刷分。

底层事实处理、索引或必要删除失败仍阻断相关样本。凭据无效、预算耗尽、产物损坏、源码不一致和内部程序错误不会被计零掩盖；保留已有产物，返回非零码，并在 `partial_summary.json` 列出覆盖情况，不输出完整主成绩。判断本次是否完成，以 `status.json` 为准，不能仅看历史汇总文件是否存在。

进度中的“接口失败”和“校验失败”分开计数，校验失败仍保留真实请求用量，不重复计算费用。本次修复后请使用新的运行名称，例如 `mab_smoke_02`；原 `mab_smoke_01` 保持只读。

## 离线检查

在 `membase-ourmem` 环境运行 `python -m unittest discover -s tests -q`。测试使用模拟模型，不发起付费请求；同时可检查各示例的 `--dry-run` 以及根三入口的 `--help`。通过这些检查只代表重构与原协议一致，不代表正式实验准确率。
