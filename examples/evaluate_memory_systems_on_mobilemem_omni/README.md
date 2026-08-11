# Evaluate Textual Memory Systems on MobileMem-Omni

This example runs Long Context, NaiveRAG, Mem0, LangMem, and EverMemOS on
[MobileMem-Omni](https://github.com/zjunlp/MobileMem) through MemBase's shared
construction, retrieval, question-answering, and evaluation pipeline.

The five methods use the same prepared dataset, QA prompt, judge, metrics, and
stage scripts. Their memory construction and retrieval behaviour is selected by
the method name and its corresponding configuration file.

## Supported methods

| Method argument | Memory system | Requirements | Extra services |
|---|---|---|---|
| `long_context` | Long-Context | `envs/long_context_requirements.txt` | None for construction/retrieval |
| `rag` | NaiveRAG | `envs/rag_requirements.txt` | Embedding endpoint |
| `mem0` | Mem0 | `envs/mem0_requirements.txt` | LLM and embedding endpoints |
| `langmem` | LangMem | `envs/langmem_requirements.txt` | LLM and embedding endpoints |
| `evermemos` | EverMemOS | `envs/evermemos_requirements.txt` | LLM, embedding, and reranker endpoints |

For other memory methods, see the [MobileMem-Omni evaluation directory](https://github.com/zjunlp/MobileMem/tree/main/omni/eval).

Use a separate Python environment for each method because their dependencies
may conflict:

```bash
conda create -n membase-mobilemem-omni-rag python=3.12 -y
conda activate membase-mobilemem-omni-rag
pip install -r envs/rag_requirements.txt
pip install nltk
```

## 1. Convert MobileMem-Omni to LoCoMo-shaped JSON

The Stage5/Stage6/Stage10 conversion remains in the MobileMem repository because
it is dataset-specific. For a caption-only evaluation of these textual methods,
keep captions in the dialogue text and remove raw image fields:

```bash
cd /path/to/MobileMem
python omni/eval/eval/Jsonl2Locomo.py \
  --stage5 /path/to/stage5.jsonl \
  --stage6-dir /path/to/stage6 \
  --stage10 /path/to/stage10_image_summaries.jsonl \
  --output-dir /path/to/converted_locomo \
  --no-image
```

Do not pass `--no-caption-in-text` when category-6 visual reasoning questions
are included. These five methods are textual and consume image information only
through captions embedded in the conversation text.

## 2. Merge and validate the converted files

The converter writes one `locomo_u*.json` file per user. Merge them into the
single input file expected by this example:

```bash
cd /path/to/MemBase
python examples/evaluate_memory_systems_on_mobilemem_omni/prepare_data.py \
  /path/to/converted_locomo
```

The prepared file is written to:

```text
examples/evaluate_memory_systems_on_mobilemem_omni/data/mobilemem_omni_locomo.json
```

Preparation fails on duplicate users, missing session timestamps, or question
categories outside MobileMem-Omni's supported range 1-7.

## 3. Configure model endpoints

Edit the selected method config under `configs/` and replace all API-key and
endpoint placeholders. Also edit `configs/api_config.json`; it supplies the QA
model and LLM judge endpoints.

NaiveRAG, Mem0, LangMem, and EverMemOS use an OpenAI-compatible embedding endpoint.
The provided configs expect all-MiniLM-L6-v2 on port 8008:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/all-MiniLM-L6-v2 \
  --port 8008 \
  --served-model-name all-MiniLM-L6-v2 \
  --gpu-memory-utilization 0.5
```

EverMemOS additionally expects Qwen3-Reranker-4B on port 8001:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve pretrained_models/Qwen3-Reranker-4B \
  --port 8001 \
  --served-model-name Qwen3-Reranker-4B \
  --gpu-memory-utilization 0.4 \
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
```

## 4. Run one method

Each method has a dedicated entry script. Run one stage at a time:

```bash
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_long_context.sh construction
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_long_context.sh search
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_long_context.sh evaluation
```

The other method entry points are:

```bash
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_naive_rag.sh construction
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_mem0.sh construction
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_langmem.sh construction
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_evermemos.sh construction
```

Passing `all` runs the three stages sequentially in the foreground:

```bash
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_naive_rag.sh all
```

## 5. Use the shared stage scripts directly

The dedicated scripts are thin wrappers around these shared commands:

```bash
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_construction.sh rag
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_search.sh rag
bash examples/evaluate_memory_systems_on_mobilemem_omni/run_evaluation.sh rag
```

Supported environment overrides include:

```bash
DATASET_PATH=/path/to/mobilemem_omni_locomo.json
NUM_WORKERS=8
SAMPLE_SIZE=2
TOP_K=15
START_IDX=0
END_IDX=2
QA_MODEL=gpt-5.4-mini
JUDGE_MODEL=Qwen3-14B
QA_BATCH_SIZE=4
JUDGE_BATCH_SIZE=4
RERUN=0
```

Run evaluation once with each QA backbone. GPT-5.4-mini is the default; use
`QA_MODEL=Qwen3-VL-8B-Instruct` for the second run. Preserve or rename the first
evaluation and summary files before the second run because both runs otherwise
write to the same output paths.

Keep `TOP_K`, `START_IDX`, and `END_IDX` aligned between search and evaluation.
Long Context defaults to `top_k=1`; the other methods default to `top_k=15`.
The scripts automatically set `MEMORY_LANGUAGE=zh` for EverMemOS unless it is
already defined.

## Outputs

Each method writes to its own directory under `outputs/`:

```text
outputs/<method>/
├── MobileMemOmni_stage_1.json
├── <top_k>_<start_idx>_<end_idx>.json
├── <top_k>_<start_idx>_<end_idx>_evaluation.json
└── <top_k>_<start_idx>_<end_idx>_evaluation_summary.json
```

The evaluation uses the same MobileMem-Omni QA prompt and computes MemBase's
default F1, BLEU-1, and LLM-as-a-Judge metrics. `summarize_results.py` saves both
overall scores and per-question-type scores for all seven categories.

## Dataset logs

The [MobileMem-Omni Logs](https://drive.google.com/file/d/1ZPaUzsu-gyDXZYqiy2b47v6grnfBg7HW/view?usp=drive_link) contain the experimental logs.
