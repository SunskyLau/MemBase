# 读取侧关系链对照

本次沿用本地 `membase-ourmem` 环境及 `envs/.env` 的接口，无本地模型权重、无 GPU 分配、无依赖安装。

原运行 `mab_6k_ourmem_4o_mini_01` 保持不变，新运行是 `mab_6k_ourmem_4o_mini_readchain_01`。脚本只复制已完成构建及请求历史，随后执行检索、回答和官方评分；6k 单跳和多跳各 100 题，模型仍为 `gpt-4o-mini`，样本并发仍为 2。费用历史被继承，新增请求成本需要相对 `comparison_source.json` 中的原账本统计计算。

从任意工作目录检查（不调用接口、不创建新数据库）：

```bash
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k_readchain.sh --dry-run
```

实际运行会调用付费接口：

```bash
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k_readchain.sh
```

离线检查（不调用接口）：

```bash
cd /home/jovyan/agent-memory/MemBase
/home/jovyan/my-conda-envs/membase-ourmem/bin/python -m unittest tests.ourmem.test_relation_reader tests.benchmarks.test_read_revision -b -q
bash -n examples/evaluate_ourmem_on_memoryagentbench/run_6k_readchain.sh
```

新读取保留查询内的关系步骤及字面值绑定；后续查询采用已绑定实体，最后一跳的证据包含前面各跳。当前关系查询紧凑呈现可用事实及其来源，不展示无关邻居与旧值正文；未验证的自由解释不再进入最终回答上下文。仍保留历史读取、集合扫描、五层派生能力及既有预算；本轮没有改变写入或重新生成高层记忆。

正式运行前冻结代码。原 100 道多跳题已被用于诊断，本轮是开发对照，不是独立未见测试。不得在预测过程中使用人工逐题关系链、标准答案或修改官方评分。一次运行结束后如实报告进退，不自动进行无限参数搜索。
