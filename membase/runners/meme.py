"""保留 MEME 官方问答与评分协议，并在阶段边界检查完整性。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from ..datasets import meme as data
from ..evaluation import meme as evaluation
from ..utils.benchmark_files import preserve_incomplete, read_json, write_json, sha256_file
from ..utils.experiment import child_environment, finish_run, require_runtime, run_process, start_run
from .benchmark import BenchmarkRunConfig


def input_directory(stage_dir: Path, label: str, files: list[tuple[str, Path]], dry_run: bool) -> Path:
    """每次只交给官方入口尚未完成的文件，不让存在但损坏的产物被跳过。"""
    digest = hashlib.sha256(json.dumps([(name, str(path)) for name, path in files]).encode()).hexdigest()[:12]
    target = stage_dir / "inputs" / f"{label}_{digest}"
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        for name, source in files:
            link = target / name
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            if link.exists():
                raise ValueError(f"输入目录已有不同文件：{link}")
            link.symlink_to(source.resolve())
    return target


def run_stage(config: BenchmarkRunConfig, variant: str, smoke: bool) -> dict | None:
    episodes = data.select_episodes(config.data_root, variant, smoke)
    stage_dir = config.run_dir / ("smoke" if smoke else "full") / variant
    output_dir = stage_dir / "outputs"
    judge_dir = stage_dir / "judge"
    baseline = "md_file" if config.baseline == "md_flat" else config.baseline
    pending = []
    for ep in episodes:
        path = output_dir / evaluation.output_name(ep, baseline, config.answer_model)
        if path.exists() and not config.dry_run:
            try:
                evaluation.validate_answer(path, ep, baseline, config.answer_model,
                                           config.internal_model, config.top_k)
                continue
            except (ValueError, KeyError, TypeError):
                preserve_incomplete(path)
        pending.append((f"episode_{data.episode_key(ep)}.json", data.episode_path(config.data_root, variant, ep)))

    env = child_environment(config.base_url, config.api_key_env)
    judge_env = child_environment(config.judge_base_url or config.base_url, config.judge_api_key_env or config.api_key_env)
    for values in (env, judge_env):
        values["PYTHONPATH"] = str(config.upstream_dir / "code")
        values["TIKTOKEN_CACHE_DIR"] = str(stage_dir / "cache/tiktoken")
    if pending:
        inputs = input_directory(stage_dir, "answers", pending, config.dry_run)
        module = "eval.in_context_baseline" if baseline == "in_context" else "eval.run_agent"
        command = [sys.executable, "-m", module, "-d", str(inputs), "-o", str(output_dir),
                   "--model", config.answer_model, "-w", str(config.workers), "--skip-existing"]
        if baseline != "in_context":
            command += ["--agent-type", baseline, "--internal-model", config.internal_model]
        if baseline in {"bm25", "dense"}:
            command += ["--top-k", str(config.top_k)]
        if baseline == "dense" and config.model_profile:
            import os
            env.update(MEMBASE_EMBEDDING_API_KEY=os.environ.get(config.embedding_api_key_env or config.api_key_env, ""),
                       MEMBASE_EMBEDDING_BASE_URL=config.embedding_base_url or config.base_url,
                       MEMBASE_EMBEDDING_MODEL=config.embedding_model)
            command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/run_native_benchmark.py"),
                       "--benchmark", "meme", "--upstream", str(config.upstream_dir), "--module", module,
                       "--", *command[3:]]
        run_process(command, stage_dir / "work", stage_dir / "logs/answers.log",
                    env=env, dry_run=config.dry_run)

    if config.dry_run:
        command = [sys.executable, "-m", "eval.judge", "-d", str(stage_dir / "inputs/pending_judges"),
                   "-o", str(judge_dir), "--judge-model", config.judge_model,
                   "-w", str(config.judge_workers), "--check-workers", str(config.check_workers)]
        run_process(command, stage_dir / "work", stage_dir / "logs/judge.log", env=judge_env, dry_run=True)
        print(f"预期：{variant}{' smoke' if smoke else ''}，{len(episodes)} 个样本及全部前后问题")
        return None

    # 即便官方入口返回 0，也必须逐一找到全部预期回答。
    answers, errors = {}, []
    for ep in episodes:
        key = data.episode_key(ep)
        path = output_dir / evaluation.output_name(ep, baseline, config.answer_model)
        try:
            answers[key] = evaluation.validate_answer(path, ep, baseline, config.answer_model,
                                                      config.internal_model, config.top_k)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        raise ValueError(f"MEME 回答阶段不完整 ({len(errors)}/{len(episodes)})：\n" + "\n".join(errors))

    receipts_path = stage_dir / "judge_receipts.json"
    try:
        receipts = read_json(receipts_path) if receipts_path.exists() else {}
    except ValueError:
        preserve_incomplete(receipts_path)
        receipts = {}
    pending = []
    for ep in episodes:
        key = data.episode_key(ep)
        answer_path = output_dir / evaluation.output_name(ep, baseline, config.answer_model)
        judge_path = judge_dir / evaluation.judge_name(ep, baseline, config.answer_model, config.judge_model)
        if evaluation.judge_is_reusable(answer_path, judge_path, config.judge_model,
                                        receipts.get(key), answers[key]):
            continue
        preserve_incomplete(judge_path)
        receipts.pop(key, None)
        pending.append((answer_path.name, answer_path))

    failure = None
    if pending:
        inputs = input_directory(stage_dir, "judges", pending, False)
        command = [sys.executable, "-m", "eval.judge", "-d", str(inputs), "-o", str(judge_dir),
                   "--judge-model", config.judge_model, "-w", str(config.judge_workers),
                   "--check-workers", str(config.check_workers)]
        try:
            run_process(command, stage_dir / "work", stage_dir / "logs/judge.log", env=judge_env)
        except RuntimeError as exc:
            failure = exc

    # 部分评分失败时仍保存已验证的完成记录，下一次不重复支付这些评分。
    results, errors = [], []
    for ep in episodes:
        key = data.episode_key(ep)
        answer_path = output_dir / evaluation.output_name(ep, baseline, config.answer_model)
        judge_path = judge_dir / evaluation.judge_name(ep, baseline, config.answer_model, config.judge_model)
        try:
            result = evaluation.validate_judge(judge_path, answers[key], config.judge_model)
            receipts[key] = evaluation.receipt(answer_path, judge_path, config.judge_model)
            results.append((ep["domain"], result))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
    write_json(receipts_path, receipts)
    if errors or failure:
        raise ValueError(f"MEME 评分阶段不完整：{failure or ''}\n" + "\n".join(errors))
    summary = evaluation.summarize(results)
    write_json(stage_dir / "summary.json", summary)
    return summary


def run(config: BenchmarkRunConfig) -> dict | None:
    prepared = data.check(config.data_root, config.upstream_dir)
    if not config.dry_run:
        imports = ["openai", "numpy", "tiktoken"]
        if config.baseline == "bm25":
            imports += ["bm25s"]
        if config.baseline == "md_flat":
            imports += ["dotenv"]
        require_runtime(imports, config.api_key_env)
        from ..configs.model_profiles import credential
        credential(config.judge_api_key_env or config.api_key_env)
        if config.baseline == "dense":
            credential(config.embedding_api_key_env or config.api_key_env)
        start_run(config.run_dir, config.saved_config(),
                  {"upstream_commit": data.UPSTREAM_COMMIT, "data": prepared,
                   "transport_adapter": sha256_file(Path(__file__).with_name("native_transport.py"))})
    summaries = {}
    for variant, smoke in data.stages(config.mode):
        key = f"{variant}_smoke" if smoke else variant
        summaries[key] = run_stage(config, variant, smoke)
    if config.dry_run:
        print("仅预览；未启动官方进程")
        return None
    finish_run(config.run_dir, {"stages": summaries})
    return summaries
