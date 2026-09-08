# 三个 MEME 基线入口：离线检查原始记录

日期：2026-09-08。本文件记录确定性检查，不作正式实验完整性裁决。未安装环境、未调用模型、未修改实现或已有运行。

- external/MEME-public HEAD：0271ad85389a963cbc4892a36391f868ba4d18d1；git diff --name-only 为空。
- nofiller：100情节，494个before问题、694个after问题；SHA-256 1687d028cc1638986df9f58f0c3f072f614cf13d72a791541b4439fddf701636。
- filler32k：100情节，494个before问题、694个after问题；SHA-256 a88d28374a002b3e5b1683fb7201d06a1ce739d2ebf94c971c37bb65cf6ebdd3。
- filler128k：40情节，197个before问题、277个after问题；SHA-256 eb861f866dde067e6c7ea6db9bdf17b83a56cdc578dcca57d9d8809dc22747fe。
- examples/evaluate_{in_context,dense,md_flat}_on_meme/run.example.sh 与对应 experiments/meme_*/run.sh 均存在、bash -n退出0。
- 三个示例从/tmp执行smoke/core/full共9次干运行，全部退出0。smoke为nofiller2情节；core为该冒烟加filler32k100；full为nofiller100＋filler32k100＋filler128k40。
- Conda登记中未发现membase-meme-incontext、membase-meme-dense、membase-meme-mdflat。

两套入口的字面模型设置：experiments/meme_*/run.sh的回答和评判均为gpt-4.1-mini，MD-flat内部模型也为gpt-4.1-mini；examples/evaluate_*_on_meme/run.example.sh的回答/内部模型为gpt-4o-mini，评判为gpt-4o-2024-11-20。本文仅记录文件现值。

测试命令：

```text
/home/jovyan/my-conda-envs/membase-ourmem/bin/python -m unittest tests.benchmarks.test_benchmark_workflow tests.benchmarks.test_protocol_parity -b -q
```

原始结果：21项，失败1项。失败为test_native_baseline_execution_is_unchanged，在tests/benchmarks/test_protocol_parity.py:70比较membase/runners/memoryagentbench.py与历史版本源码哈希不等。其余20项通过。该条记录本身不判定是否存在MEME功能回归；需要单独核对断言适用范围。
