# 选择实验模型

当前四个研究数据集的公共入口、OurMem、A-MEM 的 MAB 入口及六个官方基线模板共用模型配置。专用旧示例和绑定历史产物的续跑脚本保留原行为。

脚本顶部设置：

```bash
experiment_model="gpt"  # 只选 gpt 或 qwen：构建和回答一起切换
JUDGE_PROFILE="gpt"
EMBEDDING_PROFILE="gpt"
EMBEDDING_MODEL="text-embedding-3-small"
```

`gpt` 固定对应 `gpt-4o-mini`，`qwen` 固定对应 `qwen3-30b-a3b-instruct-2507`。两套实验统一使用 `gpt-4o-2024-11-20` 评判。MAB 使用官方确定性评分，不调用评判模型。嵌入服务独立选择，不随构建模型切换到百炼。

地址与密钥统一读取 `MemBase/envs/.env`：GPT 服务使用 `OPENAI_BASE_URL`、`OPENAI_API_KEY`；百炼使用 `DASHSCOPE_BASE_URL`、`DASHSCOPE_API_KEY`。已导出的同名环境变量优先于该文件。密钥不保存到运行配置或命令日志。

例如，从任意目录运行 OurMem 的 6k 实验：

```bash
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k_hybrid.sh --dry-run
/home/jovyan/agent-memory/MemBase/examples/evaluate_ourmem_on_memoryagentbench/run_6k_hybrid.sh
```

先在脚本内修改 `experiment_model`，无需在启动命令中传模型参数。运行名称自动追加 `_qwen` 或 `_gpt`，当前脚本分别写入 `mab_6k_ourmem_hybrid_04_qwen` 和 `mab_6k_ourmem_hybrid_04_gpt`。同一配置可以续跑；修改模型、服务地址或方法配置后应换一个运行名称，不能混用旧记忆与结果。

三个阶段读取同一份已保存配置。未提供 `--model-profile` 的旧命令保留原默认行为。官方原生基线仅适配模型服务调用，保留原方法、提示与评分。

本次检查包括五项离线路由检查、十项 OurMem 核心检查、17 个脚本的语法检查，以及 11 个模板各两套配置的干运行（dry run）。未调用付费接口，也未启动正式实验；完整千问实验的兼容性和效果仍需实际运行验证。
