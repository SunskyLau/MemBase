"""实验进程、日志和续跑配置；不导入任何模型后端。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

from .benchmark_files import read_json, write_json


def require_runtime(modules: list[str], api_key_env: str = "OPENAI_API_KEY") -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError("当前 Conda 环境缺少依赖：" + ", ".join(missing))
    from ..configs.model_profiles import credential
    credential(api_key_env)


def child_environment(base_url: str, api_key_env: str = "OPENAI_API_KEY") -> dict[str, str]:
    env = os.environ.copy()
    env.update(OPENAI_BASE_URL=base_url, OPENAI_API_BASE=base_url, PYTHONUNBUFFERED="1")
    env["OPENAI_API_KEY"] = os.environ.get(api_key_env, "")
    return env


def run_process(command: list[str], cwd: Path, log_path: Path, *,
                env: dict[str, str], dry_run: bool = False) -> None:
    def redact(text: str) -> str:
        for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "MEMBASE_EMBEDDING_API_KEY"):
            key = env.get(name, "")
            if key:
                text = text.replace(key, "[REDACTED]")
        return text

    display = f"cd {shlex.quote(str(cwd))}\n{shlex.join(command)}\n"
    print(redact(display), end="", flush=True)
    if dry_run:
        return
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(redact(display))
        log.flush()
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(redact(line))
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise RuntimeError(f"官方进程退出码 {code}；日志：{log_path}")


def start_run(run_dir: Path, config: dict, protocol: dict) -> None:
    """只在相同协议和配置下续跑；不把旧目录中的未知结果自动并入新实验。"""
    path = run_dir / "config.json"
    expected = {"schema_version": 1, "config": config, "protocol": protocol}
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"RUN_ID 对应的配置不同，请使用新的 RUN_ID：{run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"运行目录已有旧产物但没有配置记录，请使用新的 RUN_ID：{run_dir}")
    else:
        write_json(path, expected)
    write_json(run_dir / "status.json", {"status": "running"})


def finish_run(run_dir: Path, summary: dict) -> None:
    warnings = bool(summary.get("technical_failure_questions") or summary.get("memory_warning_samples"))
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "status.json", {"status": "complete_with_warnings" if warnings else "complete"})
    label = "流程结束（含技术失败计零或记忆维护缺口，请查看汇总）" if warnings else "全量完成"
    print(f"{label}：{run_dir / 'summary.json'}")


def fail_run(run_dir: Path, error: Exception) -> None:
    # 不保存包含接口响应原文的异常，以免把凭据写进配置/状态文件。
    write_json(run_dir / "status.json", {"status": "incomplete", "error_type": type(error).__name__})
