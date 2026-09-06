# OurMem V5：代码导航与运行

这里是在原有 OurMem 上迁移的实现；设计依据是 [pipeline_v5.md](../../docs/pipeline_v5.md)。旧实验产物保持不变，新运行从原始消息重建，不读取旧记忆格式。

## 从底层到上层阅读

| 文件 | 负责什么 |
| --- | --- |
| `models.py` | 来源、记忆版本、依赖、控制操作、时间及回答结果的数据约定。 |
| `persistence.py`、`store.py` | SQLite 事务、追加记录、处理进度、读取快照和向量缓存。每个独立样本一个文件。 |
| `maintenance.py` | 判断当前或历史记忆能否使用，展开完整证据，处理多路径、关闭、纠错、删除及依赖传播。 |
| `retriever.py` | NumPy 精确向量排序、BM25 稀疏检索、时间候选融合及原文分页。先执行可见性过滤，再选候选。 |
| `llm.py`、`tokenization.py` | 共用模型调用、输出校验、至多两次额外重试、请求预算、日志和词元统计。 |
| `extractor.py`、`reconciler.py` | 抽取并定位原文；结合局部候选决定新增、补充支持、修订或控制操作。 |
| `inducer.py`、`writer.py` | 生成并验证局部多层依赖，顺序提交来源批次，维护待修复目标。 |
| `calculation.py`、`evidence_format.py`、`reader.py` | 分解查询、补检、集合扫描、完整证明去重与有原文依据的数值计算；不写回语义记忆。 |
| `system.py` | 组装组件，对外提供摄入、刷新、读取和删除接口。 |

事实与高层结论共用 `MemoryVersion`。一条 `DependencyLink` 内的前提共同成立；指向同一版本的不同支持关系可以独立成立。依赖绑定具体版本，不随修订自动换成新版本。当前默认最多派生五层。

读取时以 `MaintenanceEngine.evaluate(...)` 返回的 `Resolution` 为准，不能用不可变记录中写入时的 `status` 判断现在是否有效；时间、支持路径和控制操作共同决定当前状态。

正常使用顺序是 `ingest(...) → flush(...) → answer(...)`。`flush` 返回绑定命名空间的快照标识；`prepare_evidence(...)` 只准备证据，让评测运行器继续使用官方回答模板。技术失败不会发布一个假装处理完成的快照；语义处理预算耗尽则明确保留未完成状态。

参数集中在 [`membase/configs/ourmem.py`](../configs/ourmem.py)。提示词为英文，证据和记忆内容沿用来源语言。数据库和日志保留原文，属于实验数据，不应公开提交。

## 四个实验入口

每个实验目录都有无密钥的 `run.example.sh`。复制成同目录下已忽略提交的 `run.sh`，填写顶部配置后执行 `./run.sh`；先运行 `./run.sh --dry-run` 可以核对实验范围而不调用模型、不创建记忆数据库。

| 目录（相对仓库根目录） | 范围与评分 |
| --- | --- |
| `experiments/ourmem_locomo/` | 第 1～4 类问题；官方分类评分，额外模型评判单独报告。 |
| `experiments/ourmem_longmemeval/` | LongMemEval-S 清洗版本；官方回答判分及分类汇总。 |
| `experiments/ourmem_memoryagentbench/` | 已固定的八个冲突消解子集；官方子串匹配。 |
| `experiments/ourmem_meme/` | 无填充、32k、128k 版本；变化前后分别提问，官方六类任务判分与平凡通过过滤。 |

`MODE` 选择 `smoke`、`core` 或 `full`。冒烟模式只减少样本及问题，不悄悄截短这些问题对应的历史。输出保存在各目录 `runs/<RUN_ID>/`，包括配置与源码指纹、每样本 SQLite、回答、证据、调用日志、成本及评分。

同一个 `RUN_ID` 只复用完整且配置、数据和源码一致的阶段。修改方法、提示词或参数后使用新的 `RUN_ID`，不能混合新旧结果。MEME 的变化前回答只能来自变化前快照。

入口通过 `scripts/run_benchmark.py` 调用 `membase/runners/ourmem.py`；输入白名单和来源映射位于 `membase/datasets/ourmem_benchmarks.py`，官方评分接入位于 `membase/evaluation/ourmem.py`。现有其他基线继续走原运行器。

## 验证与预算

在 `membase-ourmem` 环境、仓库根目录运行 `python -m unittest discover -s tests -q`，执行离线回归；不会请求模型。必需依赖固定在 `envs/ourmem_requirements.txt`，官方代码和数据检查由 `scripts/prepare_benchmarks.py` 负责。

`scripts/validate_ourmem.py` 用独立构造的小场景验证初始回答、状态更新和删除。验证产物位于 `experiments/ourmem_validation/runs/v5_acceptance/`。其所有尝试共用 `request_budget.sqlite3`：累计最多 100 次语言模型请求和 20 次嵌入请求，包含重试及评判，不能通过换尝试名称重置。

本轮实际通过与未完成项见 [验证记录](../../experiments/ourmem_validation/README.md)。`--real-data-extraction` 仅检查首批真实消息的抽取接入，不是完整历史上的基准问答。

正式入口默认不限制请求总数，不能把冒烟测试误当成廉价的几次调用。若要限制，设置脚本顶部两个请求上限；跨运行限制需同时指定相同的 `BUDGET_LEDGER`。任一上限达到即停止，不静默丢掉失败样本。

离线回归验证程序规则；小场景验证实际模型是否遵守协议；四个数据集的准确率、成本及论文结论仍需要正式实验。完成某一种验证不代表另外两种也已经完成。
