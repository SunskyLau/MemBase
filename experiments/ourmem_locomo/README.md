# OurMem on LoCoMo

该实验采用固定协议评估 OurMem：

- 使用 LoCoMo 全部 10 条对话；
- 只评估第 1–4 类问题，共 1,540 道；
- 每条双人对话维护一个共享记忆；
- 将数据集发布的图像描述 `blip_caption` 和图像检索词 `query` 追加到原消息；
- 记忆构建、回答和评判均使用 `gpt-4.1-mini`；
- 嵌入使用 `text-embedding-3-small`；
- 主结果固定检索 10 条记忆；
- 报告模型评判准确率、分类准确率、F1、BLEU-1 和证据召回率。

`query` 可能比图像描述包含更具体的实体或地点，因此属于特权元数据。当前结果必须
明确标记为“图像描述＋检索词协议”，不能与只使用图像描述的运行直接比较。

原始数据中有 3 个证据编号无法对应到现有消息：`D`、`D10:19` 和 `D4:36`。
这些问题仍参与准确率评测，但其证据召回率可能被低估，最终报告必须单独注明。

完整运行：

```bash
conda run --no-capture-output -n membase-ourmem \
  python experiments/ourmem_locomo/run.py \
  --stage all \
  --run-name full_v1 \
  --workers 4 \
  --evaluation-concurrency 4
```

环境和接口预检：

```bash
conda run --no-capture-output -n membase-ourmem \
  python experiments/ourmem_locomo/run.py \
  --stage preflight \
  --run-name preflight
```

各阶段可以独立重跑：

```bash
python experiments/ourmem_locomo/run.py --stage construction --run-name full_v1
python experiments/ourmem_locomo/run.py --stage search --run-name full_v1
python experiments/ourmem_locomo/run.py --stage evaluation --run-name full_v1
python experiments/ourmem_locomo/run.py --stage analysis --run-name full_v1
```

默认从 `examples/ourmem/api_config.json` 读取本地接口配置。运行产物写入
`experiments/ourmem_locomo/runs/<run-name>/`，其中包含日志、分片结果、汇总结果、
运行清单和最终报告。密钥不会写入运行清单或日志。
