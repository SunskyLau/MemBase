# OurMem：实现与运行

OurMem 保留来源（Source）、记忆版本（MemoryVersion）和依赖关系（DependencyLink）三类核心记录。事实与派生结论共用版本机制；来源真实、引用明确、删除隔离和禁止循环自证是程序边界，记忆内容和结构由实际输入与模型归纳形成。

## 一次输入怎样处理

原文先写入 SQLite。每批默认最多 8 条消息；抽取逐项保留合法事实，无法核对的内容保留原文与未决记录。协调模型先比较陈述关系：等价、独立补充、变化、纠错、冲突等；程序再映射到写入动作。涉及归并或替代的候选额外做一次简短的两句对照，不提供前次判断作暗示。兼容补充保持独立，关系未决时不关闭旧事实。标识由程序绑定，未知目标经过一次关系补检，仍不明确则待决。

每条已确认变化立即重算已知依赖，必要修复使用当时的来源前缀。可选归纳在批末执行，默认最多 8 次生成；没有新结论或预算用完是正常结束，不是维护故障。批内中间变化不被批尾状态覆盖，评测观察点之间不混合输入。

可选归纳与必要修复使用不同提示及输出结构。归纳重点是比较、条件应用和有范围的概括；无提案时保存简短原因（no_op_reason），不强制产生主张。公开更新策略按实际输入提供，不在普通对话中默认启用新来源覆盖规则。两句复核的请求计入 reconcile_check 阶段的实际成本。

每条充分支持路径独立验证、提交。或关系（OR）的一条路径失败，不阻挡其他路径；与关系（AND）的必要前提不能截断。验证可对新结论做一次有依据的范围或条件修订。主张、关系数量是分批容量，不因超额拒绝整份结果；单条前提不设固定数量上限，完整路径默认最多五层。

每条依赖本身就是一组共同前提。验证模型只判断整组是否充分，不再输出前提索引或将其重新拆成多个路径；程序保留全部已核对前提。多条独立支持在派生提案中分别表达，仍保留多路径能力。

只有实际内容、支持或可用状态变化才继续传播。普通召回不触发旧决定复核；复核依据变化或新证据时，不允许旧来源绕过公开的新来源优先规则。来源修正、撤回与删除仍有不同语义。

## 读取与故障

接口保持：

```python
ingest(messages, namespace, input_policy)
flush(namespace, source_cutoff) -> snapshot_id
answer(query, namespace, snapshot_id, query_time) -> AnswerResult
```

快照（snapshot）固定已提交记忆及相关未决记录。读取显式使用公开输入规则；规划或辅助判断失败时，保留已取得的合法证据并记录降级状态。删除过滤仍覆盖原文、历史、证据展开与降级读取。正式回答和评分继续使用各数据集的官方协议。

网络故障与内容纠错分开计数。OurMem 的恢复窗口默认 900 秒，退避 5、15、30、60、120 秒，之后每次 120 秒；单请求不超过剩余窗口和 120 秒。窗口耗尽后保存检查点并暂停。结构纠错最多额外两次，所有真实请求、失败和重试进入账本；SDK 不再自行重试。A-MEM 和官方原生基线的默认恢复行为不变。

监控分别展示输入未决、实际待修复目标和可选探索结束次数。它们不是同一种失败。只有获得合法终态的问题才参与对应结果检查；技术失败依照原协议明确计零，分母不变，未完成实验不发布完整准确率。

## 模块导航

| 模块 | 作用 |
|---|---|
| `models.py`、`store.py`、`persistence.py` | 记录、事务、快照、待决进度与向量缓存 |
| `extractor.py`、`reconciler.py` | 原文核对、语义协调与内部标识绑定 |
| `writer.py`、`inducer.py` | 分批归纳、逐路径验证与局部提交 |
| `maintenance.py`、`retriever.py` | 统一可用性、反向依赖索引、混合检索与缓存 |
| `reader.py`、`calculation.py` | 需求检索、完整证据、只读推理与确定性计算 |
| `system.py`、`membase/layers/ourmem.py` | 公共接口与 MemBase 薄适配 |
| `membase/inference_utils/model_client.py`、`reference_codec.py` | 调用恢复、请求账本和请求内短引用 |

参数位于 `membase/configs/ourmem.py`。初始请求不再统一预留 6000 词元的旧输出；重试按实际失败输出和反馈组织。上下文按选中的完整路径展开，进度更新不重建语义视图，稀疏索引和向量按数据变化复用。

## 验证与启动

从项目根目录 `/home/jovyan/agent-memory/MemBase` 执行离线检查：

```bash
/home/jovyan/my-conda-envs/membase-ourmem/bin/python -m unittest discover -s tests -q -b
/home/jovyan/my-conda-envs/membase-ourmem/bin/python scripts/validate_ourmem_reliability.py
```

第二条只显示独立合成验证计划，不调用接口。显式添加 `--real` 才调用模型，接口凭据通过环境变量提供。所有尝试共用 `experiments/ourmem_validation/reliability/budget.sqlite`，总上限为 30 次语言模型请求、10 次嵌入请求，包含失败和重试；不能换尝试名重置上限。

6k 单跳与多跳共 200 题的入口可从任何目录预演：

```bash
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k.sh --dry-run
```

不加 `--dry-run` 才启动付费实验。其余四数据集入口继续使用公共构建、检索、评测三阶段。旧数据库和实验结果原样保留；新存储格式为 `ourmem-v5-3`，不迁移旧记忆，正式实验需要新运行名称。方法配置和纯运行配置分别记录；本轮不提供跨运行记忆缓存。

历史设计文档保持不变；这里描述当前代码。离线规则通过不等于真实模型永不出错，合成场景通过也不等于正式基准准确率已经验证。具体小额验证结果以实际产物为准。

本次小额验证未全部通过：14 次语言模型请求、10 次嵌入请求后达到额度边界；发现一次语义误归并，尚未验证真实多层派生和最终问答。详见 [验证记录](../../experiments/ourmem_validation/reliability/README.md)。不要将入口已就绪理解为正式实验效果已验证。

随后针对误归并新增了关系比较与归纳提示分离。独立组件验证入口为 `scripts/validate_ourmem_semantics.py`：不加参数只显示计划，显式 `--real` 使用另行授权的最多 6 次语言模型请求，不调用嵌入。它不修改之前的 30/10 账本，也不代表完整端到端或正式基准测试；结果保存在 `experiments/ourmem_validation/semantic_admission/`。

该组件验证已使用 6 次请求：规则与距离并存、明确纠错及派生生成通过；发现验证器误拆共同前提后，改为程序保留整组，并通过原响应离线回放。最新验证提示未再在线测试，详见 [组件验证与审计](../../experiments/ourmem_validation/semantic_admission/README.md)。
