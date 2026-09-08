"""以固定官方入口运行 MAB 子集；子集之间隔离并发和续跑。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

from ..datasets import memoryagentbench as data
from ..evaluation import memoryagentbench as evaluation
from ..utils.benchmark_files import preserve_incomplete, write_json, sha256_file
from ..utils.experiment import child_environment, finish_run, require_runtime, run_process, start_run
from .benchmark import BenchmarkRunConfig


def agent_config(config: BenchmarkRunConfig, output_dir: Path) -> dict:
    result = {
        "agent_name": (f"Long_context_agent_{config.answer_model.replace('/', '-')}"
                       if config.baseline == "long_context" else "Simple_rag_bm25"),
        "model": config.answer_model, "temperature": config.temperature,
        "input_length_limit": 1000000,
        "buffer_length": 10000 if config.baseline == "long_context" else 200,
        "output_dir": str(output_dir),
    }
    if config.baseline == "bm25":
        result["retrieve_num"] = config.top_k
    return result


def result_path(config: BenchmarkRunConfig, source: str, stage_dir: Path) -> Path:
    length = {"6k": 6000, "32k": 32768, "64k": 65536, "262k": 300000}[source.rsplit("_", 1)[1]]
    name = f"{source}_None_in{length}_size10_shots0_max_samples1"
    if config.baseline == "bm25":
        name += f"_k{config.top_k}_chunk4096"
    return stage_dir / "outputs/Conflict_Resolution" / f"{name}_results.json"


def run(config: BenchmarkRunConfig) -> dict | None:
    prepared = data.check(config.data_root, config.upstream_dir)
    questions = data.load_manifest(config.data_root)["questions"]
    selected = data.subsets(config.mode)
    if not config.dry_run:
        imports = ["openai", "yaml", "datasets", "torch", "transformers", "tiktoken",
                   "langchain_core", "numpy", "nltk", "rouge_score", "editdistance", "tqdm", "dotenv"]
        if config.baseline == "bm25":
            imports += ["langchain_community", "rank_bm25"]
        require_runtime(imports, config.api_key_env)
        start_run(config.run_dir, config.saved_config(),
                  {"upstream_commit": data.UPSTREAM_COMMIT, "data": prepared,
                   "transport_adapter": sha256_file(Path(__file__).with_name("native_transport.py"))})

    def run_subset(source: str) -> tuple[str, dict | None]:
        stage_dir = config.run_dir / source
        expected = questions[source][:4] if config.mode == "smoke" else questions[source]
        output = result_path(config, source, stage_dir)
        cfg_path = stage_dir / "agent.json"
        ds_config = config.upstream_dir / "configs/data_conf/Conflict_Resolution" / f"{source.capitalize()}.yaml"
        if not ds_config.is_file():
            raise FileNotFoundError(ds_config)
        if not config.dry_run:
            write_json(cfg_path, agent_config(config, stage_dir / "outputs"))
            if output.exists():
                try:
                    saved = evaluation.validate_result(output, expected, source, config.answer_model, allow_partial=True)
                    if len(saved["data"]) == len(expected):
                        print(f"复用完整结果：{source} ({len(expected)} 问)")
                        return source, saved
                except (ValueError, KeyError, TypeError):
                    preserve_incomplete(output)
        command = [sys.executable, str(config.upstream_dir / "main.py"),
                   "--agent_config", str(cfg_path), "--dataset_config", str(ds_config)]
        if config.model_profile:
            command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/run_native_benchmark.py"),
                       "--benchmark", "memoryagentbench", "--upstream", str(config.upstream_dir), "--", *command[2:]]
        if config.mode == "smoke":
            command += ["--max_test_queries_ablation", "4"]
        env = child_environment(config.base_url, config.api_key_env)
        env["MAB_CONFLICT_PARQUET"] = str(data.raw_path(config.data_root))
        env["HF_HOME"] = str(stage_dir / "cache/huggingface")
        env["TIKTOKEN_CACHE_DIR"] = str(stage_dir / "cache/tiktoken")
        run_process(command, stage_dir / "work", stage_dir / "run.log", env=env, dry_run=config.dry_run)
        if config.dry_run:
            return source, None
        return source, evaluation.validate_result(output, expected, source, config.answer_model)

    with ThreadPoolExecutor(max_workers=config.parallel_jobs) as pool:
        results = dict(pool.map(run_subset, selected))
    if config.dry_run:
        print(f"仅预览：{len(selected)} 个子集；未启动官方进程")
        return None
    summary = evaluation.summarize(results)
    finish_run(config.run_dir, summary)
    return summary
