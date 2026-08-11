# Evaluate Memory Systems on MobileMem

This example evaluates Long-Context, Mem0, NaiveRAG, A-MEM, MemOS, EverMemOS, LangMem, and HippoRAG 2 on [MobileMem](https://github.com/zjunlp/MobileMem) with the unified three-stage MemBase workflow: memory construction, memory retrieval, and question answering with automatic evaluation.

---

## Step 1: Prepare the Environment

Different memory systems have different dependencies. Use a separate Python environment for each memory system. The commands below use Mem0 as an example. Choose a corresponding environment name and requirements file when evaluating another method.

### Option A: Install with pip

```bash
conda create -n membase_mem0 python=3.12 -y
conda activate membase_mem0
pip install -r envs/mem0_requirements.txt
pip install "vllm<=0.11.1"
pip install nltk
```

Use the requirements file for the selected method:

| Method | Requirements File |
|---|---|
| Long-Context | `envs/long_context_requirements.txt` |
| Mem0 | `envs/mem0_requirements.txt` |
| NaiveRAG | `envs/rag_requirements.txt` |
| A-MEM | `envs/amem_requirements.txt` |
| MemOS | `envs/memos_requirements.txt` |
| EverMemOS | `envs/evermemos_requirements.txt` |
| LangMem | `envs/langmem_requirements.txt` |
| HippoRAG 2 | `envs/hipporag_requirements.txt` |

### Option B: Install with uv

```bash
conda create -n membase_mem0 python=3.12 -y
conda activate membase_mem0
pip install uv
uv pip install -r envs/mem0_requirements.txt
uv pip install "vllm<=0.11.1"
uv pip install nltk
```

---

## Step 2: Prepare Models, Services, and Data

Every method except Long-Context uses Qwen3-Embedding-4B through an OpenAI-compatible endpoint. See the related [model download example](../download_models/) before starting the local vLLM server:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/Qwen3-Embedding-4B \
    --port 8008 \
    --served-model-name Qwen3-Embedding-4B \
    --gpu-memory-utilization 0.5 \
    --hf_overrides '{"is_matryoshka": true}'
```

Long-Context does not use an embedding model, so the embedding server is not required for that method.

EverMemOS also uses Qwen3-Reranker-4B. Start a second vLLM server with the sequence-classification overrides required by the reranker endpoint:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve pretrained_models/Qwen3-Reranker-4B \
    --port 8001 \
    --served-model-name Qwen3-Reranker-4B \
    --gpu-memory-utilization 0.4 \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
```

MemOS stores its tree memory in Neo4j. Start a local instance whose credentials match `configs/memos_config.json`:

```bash
docker run --name membase-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -d neo4j:latest
```

Download the [MobileMem dataset](https://huggingface.co/datasets/zjunlp/MobileMem) and place it at `examples/evaluate_memory_systems_on_mobilemem/data/MobileMem/mobilemem_data.json`.

The file must be a JSON array in which each item contains one persona's `person`, `sessions`, `graphs`, and `question_type_toolbook` fields. If the data is stored elsewhere, update `dataset_path` near the top of `run_construction.sh`.

---

## Step 3: Configure the Run

Set `METHOD` near the top of all three run scripts to the same value:

```bash
METHOD="amem"
```

Supported profiles are:

| `METHOD` | Memory System | Config |
|---|---|---|
| `long_context` | Long-Context | `configs/long_context_config.json` |
| `mem0` | Mem0 | `configs/mem0_config.json` |
| `mem0_graph` | Mem0's graph variant | `configs/mem0_graph_config.json` |
| `rag` | NaiveRAG | `configs/rag_config.json` |
| `amem` | A-MEM | `configs/amem_config.json` |
| `memos` | MemOS | `configs/memos_config.json` |
| `evermemos` | EverMemOS | `configs/evermemos_config.json` |
| `langmem` | LangMem | `configs/langmem_config.json` |
| `hipporag` | HippoRAG 2 | `configs/hipporag_config.json` |

Mem0's graph variant uses the same MemBase memory type as Mem0. The `graph_store_provider` field in `mem0_graph_config.json` enables its Kuzu graph store.

Before running, replace the API key and base URL placeholders in the selected memory-system config. Depending on the method, these credentials may be used during memory construction, retrieval, or both.

Also replace the placeholders in `configs/api_config.json`. MemBase uses these endpoints for question answering and LLM-as-a-Judge evaluation. HippoRAG 2 reads its LLM key from `OPENAI_API_KEY`. If the variable is unset, the construction and retrieval scripts automatically use the first key from `configs/api_config.json`.

Each config has an independent `save_dir` under `outputs/`. You can change model names, endpoint URLs, embedding dimensions, and output directories there. The embedding model name and dimension must match the model served in Step 2.

The construction and retrieval scripts set `MOS_EMBEDDER_TIMEOUT=120` for MemOS because its default five-second timeout is usually too short for a vLLM-served embedding model. For EverMemOS, both scripts set `MEMORY_LANGUAGE=zh` because MobileMem contains Chinese data.

---

## Step 4: Run Memory Construction

```bash
bash examples/evaluate_memory_systems_on_mobilemem/run_construction.sh
```

The script runs in the background and writes:

- Constructed memory state under the selected config's `save_dir`.
- A standardized dataset at `<save_dir>/MobileMem_stage_1.json`.
- Token-cost statistics at `<save_dir>/token_cost_<METHOD>.json`.
- A log at `examples/evaluate_memory_systems_on_mobilemem/logs/<METHOD>/construction.log`.

When `sample_size` is left empty, the script sets it to the number of personas in the raw MobileMem file. This processes the complete dataset while still generating the standardized Stage-1 file. Set `sample_size` to a positive integer for a smaller reproducible run. Construction uses `--rerun`, so an existing memory state for the selected personas is rebuilt.

---

## Step 5: Run Memory Retrieval

```bash
bash examples/evaluate_memory_systems_on_mobilemem/run_search.sh
```

Retrieval reads `<save_dir>/MobileMem_stage_1.json`, ensuring that it uses exactly the personas selected during construction. Results are written to:

```text
<save_dir>/<top_k>_<start_idx>_<end_idx>.json
```

Long-Context defaults to `top_k=1` because its complete retained history is represented as one memory entry. All other methods default to `top_k=30`. Set `top_k` near the top of both `run_search.sh` and `run_evaluation.sh` to override these defaults. An empty `end_idx` automatically resolves to the number of personas in the standardized dataset.

The background process writes its log to `examples/evaluate_memory_systems_on_mobilemem/logs/<METHOD>/search.log`.

---

## Step 6: Run Evaluation

```bash
bash examples/evaluate_memory_systems_on_mobilemem/run_evaluation.sh
```

The evaluation stage uses the retrieved memories to answer MobileMem questions and computes MemBase's default metrics, including LLM-as-a-Judge. It writes results to:

```text
<save_dir>/<top_k>_<start_idx>_<end_idx>_evaluation.json
```

Set `qa_model`, `judge_model`, `qa_batch_size`, and `judge_batch_size` near the top of `run_evaluation.sh` according to the models available from the endpoints in `configs/api_config.json`. Keep `METHOD`, `top_k`, `start_idx`, and `end_idx` aligned with the retrieval run. Evaluation runs in the background and logs to `examples/evaluate_memory_systems_on_mobilemem/logs/<METHOD>/evaluation.log`.

> [!NOTE]
> We also provide execution logs from our baseline runs on [Google Drive](https://drive.google.com/drive/folders/1beKoGFSIJDLjfUWDmlKtz7a3uk2IxGJf?usp=sharing) for reference.
