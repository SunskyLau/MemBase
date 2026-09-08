"""实验模型与服务地址的统一入口；只保存环境变量名，不保存密钥。"""
from __future__ import annotations

import os
from pathlib import Path
import shlex

DEFAULT_GPT_MODEL = "gpt-4o-mini"
DEFAULT_QWEN_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-11-20"
ENV_FILE = Path(__file__).resolve().parents[2] / "envs/.env"


def load_environment(path: Path = ENV_FILE) -> None:
    """读取字面量配置，不执行 Shell；显式导出的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        parts = shlex.split(value, comments=True)
        os.environ.setdefault(key.strip(), " ".join(parts))


def endpoint(profile: str) -> tuple[str, str]:
    """返回服务地址和密钥变量名；不按模型名称猜测服务商。"""
    if profile == "gpt":
        return os.environ.get("OPENAI_BASE_URL", "https://llm-api.net/v1"), "OPENAI_API_KEY"
    if profile == "qwen":
        url = os.environ.get("DASHSCOPE_BASE_URL")
        if not url:
            raise ValueError("请在 envs/.env 中配置 DASHSCOPE_BASE_URL")
        return url, "DASHSCOPE_API_KEY"
    raise ValueError(f"未知模型配置：{profile}")


def add_profile_arguments(parser) -> None:
    parser.add_argument("--model-profile", choices=["gpt", "qwen"], help="整次实验的构建和回答模型配置")
    parser.add_argument("--gpt-model", default=DEFAULT_GPT_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--judge-profile", choices=["gpt", "qwen", "same"], default="gpt")
    parser.add_argument("--embedding-profile", choices=["gpt", "qwen"], default="gpt")


def apply_profile(args) -> dict:
    if args.model_profile is None:
        args.internal_model = args.internal_model or "gpt-4.1-mini"
        args.answer_model = args.answer_model or "gpt-4.1-mini"
        args.judge_model = args.judge_model or "gpt-4.1-mini"
        args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return {}
    names = {"gpt": args.gpt_model, "qwen": args.qwen_model}
    judge_profile = args.model_profile if args.judge_profile == "same" else args.judge_profile
    base_url, key_env = endpoint(args.model_profile)
    judge_url, judge_key_env = endpoint(judge_profile)
    embedding_url, embedding_key_env = endpoint(args.embedding_profile)
    args.internal_model = args.internal_model or names[args.model_profile]
    args.answer_model = args.answer_model or names[args.model_profile]
    args.judge_model = args.judge_model or (DEFAULT_JUDGE_MODEL if args.judge_profile == "gpt" else names[judge_profile])
    args.base_url = args.base_url or base_url
    return dict(model_profile=args.model_profile, api_key_env=key_env,
                judge_base_url=judge_url, judge_api_key_env=judge_key_env,
                embedding_base_url=embedding_url, embedding_api_key_env=embedding_key_env)


def credential(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"缺少 {name}；请在 envs/.env 或环境变量中配置")
    return value


def redact(text: str) -> str:
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "MEMBASE_EMBEDDING_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text
