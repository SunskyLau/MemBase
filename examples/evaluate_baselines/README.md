# Unified baseline smoke tests

This directory contains small end-to-end checks for the memory baselines used in
this project. Run all commands from the baseline-specific Conda environment.

## Long-Context

The smoke scripts construct and retrieve memory for exactly one trajectory.
They do not call a QA model or an LLM judge.

```bash
conda activate membase-longcontext
cd /home/jovyan/agent-memory/MemBase

bash examples/evaluate_baselines/run_long_context_locomo_smoke.sh
bash examples/evaluate_baselines/run_long_context_longmemeval_smoke.sh
```

After configuring an OpenAI-compatible endpoint, run evaluation with:

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="https://.../v1"

bash examples/evaluate_baselines/run_long_context_evaluation.sh locomo
bash examples/evaluate_baselines/run_long_context_evaluation.sh longmemeval
```

`QA_MODEL` and `JUDGE_MODEL` default to `gpt-4.1-mini` and may be overridden as
environment variables. LongMemEval evaluation automatically appends the
question timestamp.

