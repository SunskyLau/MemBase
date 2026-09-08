# A-MEM：官方内部参数与统一 MAB 对照

本目录复用 MemBase 内置 A-MEM。方法参数优先参考官方论文实验仓库 `WujiangXu/AgenticMemory@0c8039f28fdcc08189a23c07a3437d9d2482f9c2`，不能将系统库函数默认值误称为论文实验配置。邻居标识修复仍参考 `A-mem-sys@f303dfc71e07bdc787f4bc135d4cea328ae30e99`。模型统一与输出完整性适配另行披露，不声称原论文的逐项复现。详细差异见 [来源说明](../../membase/baselines/amem/UPSTREAM.md)。

## 参数分层

| 项目 | 官方论文实验代码 | 当前设置 |
|---|---|---|
| 分析/演化温度 | 0.7 | 0.7 |
| 分析/演化输出上限 | 论文实现为1000；系统库可不指定 | 不发送，保留已验证的完整输出适配；明确不是论文原值 |
| 演化邻居数 | 5 | 5 |
| 演化刷新阈值 | 100 | 100 |
| 最终检索数量 | 评测入口默认10，README建议按模型调整 | 10，不根据正式测试成绩扫描选择；未发现MAB/Qwen3专属推荐 |
| 嵌入模型 | all-MiniLM-L6-v2 | text-embedding-3-small，统一对照配置 |
| 构建和回答模型 | 可配置 | GPT-4o-mini / Qwen3-30B-A3B-Instruct-2507，各组方法统一 |
| 最终回答温度/输出上限 | 非 A-MEM 内部参数 | MAB 回答协议：0.7 / 10词元 |
| 最终证据预算 | 没有额外8000上限 | 不额外限制，完整保留方法检索结果 |
| 无真实日期 | 原实现可用机器时间 | 保留未知，避免将虚构日期送入模型 |
| 恢复保存间隔 | 接入层职责 | 每8条输入及阶段结束保存 |

参数文件：[official_config.json](official_config.json)。其中 null 表示不主动发送输出上限，服务商仍有默认和硬上限；不代表无限输出。若服务商默认仍截断，必须用明确记录的足够上限验证，再为两种模型采用事先确定的可比策略。

演化输入保留官方方法选择的完整笔记内容、上下文、关键词和标签，不通过截短输入规避失败。A-MEM 的公共调用配置也不再继承人为16000输入上限；仍受服务商实际容量约束。OurMem 自身的输入与证据配置不随之修改。

原文、提示、邻居数量和演化策略不因调整输出上限而缩减。发生输出截断时明确停止受影响的构建，不将其当作正常“不演化”回退。其他原生可恢复回退仍记录警告；必须在论文中报告而非隐去。

## 干运行与启动

从任意工作目录调用完整路径，或在 agent-memory 目录运行：

```bash
./MemBase/examples/evaluate_amem_on_memoryagentbench/run_6k_qwen.sh --dry-run
./MemBase/examples/evaluate_amem_on_memoryagentbench/run_6k_4o_mini.sh --dry-run
```

两者都是6k单跳、多跳各100题、并发2，分别保存到：
- `experiments/amem_memoryagentbench/runs/mab_6k_amem_official_03_qwen/`
- `experiments/amem_memoryagentbench/runs/mab_6k_amem_official_03_gpt/`

确认后去掉 `--dry-run` 才会调用付费接口。脚本中的 `construction/search/evaluation` 参数可单独执行同一配置的阶段。密钥和服务地址仅从 `envs/.env` 读取，嵌入仍走单独的 OpenAI 兼容服务，不发往百炼。

`run.example.sh` 保留较大 core 范围（6k/32k，共400题）；不要把它与6k入口混淆。

## 离线验证

```bash
cd /home/jovyan/agent-memory/MemBase
/home/jovyan/my-conda-envs/membase-amem/bin/python -m unittest tests.benchmarks.test_amem_mab -b -q
```

覆盖原生笔记、邻居更新、向量刷新、集合隔离、保存加载和三阶段恢复，以及默认不发送 max_tokens、显式覆盖和截断不能默默回退；另用超过8000词元的模拟检索结果验证完整保留到回答阶段。

旧运行不改写，不将中途改参数后的结果拼到旧运行中。每8条输入保存完整检查点；相同运行配置下可恢复，不因加载重新生成向量。修改内部温度和输出策略后应重新构建。
