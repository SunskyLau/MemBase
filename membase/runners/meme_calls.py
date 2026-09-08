"""官方 MEME 调用的记录边界；不裁剪输入，不改变检索和记忆算法。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from threading import RLock
import time

from ..configs.model_profiles import redact
from ..inference_utils.model_client import (
    RequestBudget, StructuredOutputError, OutputLimitError,
    ModelClient, TransportUnavailable,
)
from ..utils.benchmark_files import read_json, write_json
from ..utils.tokenization import count_tokens

_SCOPE = ContextVar("meme_call_scope", default={})
_ACTIVE = ContextVar("meme_episode", default=None)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@contextmanager
def scope(**values):
    token = _SCOPE.set({**_SCOPE.get(), **values})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def active():
    audit = _ACTIVE.get()
    if audit is None:
        raise RuntimeError("MEME call outside an episode audit scope")
    return audit


class EpisodeAudit:
    def __init__(self, directory: Path, binding: dict, settings: dict):
        self.directory, self.settings = directory, settings
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "binding.json"
        if path.exists() and read_json(path) != binding:
            raise ValueError("MEME episode input/configuration changed; use a new run directory")
        write_json(path, binding)
        self.binding = binding
        self.budget = RequestBudget(ledger_path=directory / "requests.sqlite")
        self.lock = RLock()

    def log(self, record):
        with self.lock, (self.directory / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(redact(json.dumps(record, ensure_ascii=False, default=str)) + "\n")

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)
            self.budget.close()

    def call(self, invoke, kind, original):
        """唯一重试层：SDK 重试关闭；所有尝试先登记，再发出请求。"""
        options = deepcopy(original)
        if kind == "llm" and self.settings.get("seed") is not None:
            options["seed"] = self.settings["seed"]
        context = _SCOPE.get()
        stage = context.get("stage", "judge" if self.binding.get("stage") == "judge" else "ingest")
        validator = context.get("validator")
        ids = []
        deadline = None
        transport_failures = invalid_outputs = 0
        recovery_seconds = self.settings.get("transport_recovery_seconds", 600)
        while True:
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise TransportUnavailable(
                    "MEME interface recovery window exhausted; resume from checkpoint", request_ids=ids)
            attempt = len(ids)
            request_id = self.budget.reserve(kind, stage, options.get("model", "unknown"))
            ids.append(request_id)
            if "request_ids" in context:
                context["request_ids"].append(request_id)
            started = time.monotonic()
            usage, error = None, None
            record = {"id": request_id, "kind": kind, "stage": stage, "attempt": attempt,
                      "model": options.get("model"), "phase": context.get("phase"),
                      "question_id": context.get("question_id"), "session_index": context.get("session_index"),
                      "input_sha256": digest(options), "temperature": options.get("temperature"),
                      "max_tokens": options.get("max_tokens"), "seed": options.get("seed")}
            payload = options.get("messages", options.get("input", []))
            record["estimated_input_tokens"] = count_tokens(json.dumps(payload, ensure_ascii=False, default=str))
            if kind == "llm":
                record["messages"] = payload
            try:
                # 恢复期限不因下一次请求或错误类型变化而重置。
                request_options = dict(options)
                if remaining is not None:
                    from httpx2 import Timeout
                    request_options["timeout"] = Timeout(remaining, connect=min(300, remaining))
                    record["recovery_remaining_seconds"] = remaining
                response = invoke(**request_options)
                value = getattr(response, "usage", None)
                usage = value.model_dump() if hasattr(value, "model_dump") else vars(value) if value is not None else None
                if kind == "llm":
                    if not response.choices:
                        raise StructuredOutputError("MEME model returned no completion choices")
                    choice = response.choices[0]
                    record["finish_reason"] = choice.finish_reason
                    message = choice.message
                    record["response"] = message.model_dump() if hasattr(message, "model_dump") else vars(message)
                    if choice.finish_reason == "length":
                        raise OutputLimitError("MEME model output was truncated")
                    if choice.finish_reason == "content_filter":
                        raise StructuredOutputError("MEME provider filtered the output")
                    if getattr(message, "refusal", None):
                        raise StructuredOutputError("MEME model refused the requested output")
                    if not getattr(message, "tool_calls", None) and not (message.content or "").strip():
                        raise StructuredOutputError("MEME model returned empty output")
                    if validator:
                        validator(message.content)
                else:
                    import numpy as np
                    vectors = np.asarray([item.embedding for item in response.data])
                    expected = 1 if isinstance(options["input"], str) else len(options["input"])
                    if len(vectors) != expected or vectors.ndim != 2 or not np.isfinite(vectors).all():
                        raise StructuredOutputError("Invalid embedding response")
                    if [item.index for item in response.data] != list(range(expected)):
                        raise StructuredOutputError("Embedding response order does not match input")
                self.budget.finish(request_id, usage=usage, elapsed=time.monotonic()-started)
                record.update(status="complete", usage=usage, elapsed_seconds=time.monotonic()-started)
                self.log(record)
                return response
            except Exception as exc:
                error = exc
                self.budget.finish(request_id, usage=usage, error=redact(str(exc)), elapsed=time.monotonic()-started)
                if isinstance(exc, (StructuredOutputError, OutputLimitError)):
                    self.budget.validation_failed(request_id, redact(str(exc)))
                record.update(status="failed", usage=usage, error=redact(str(exc)),
                              error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
                self.log(record)
            transient = ModelClient._retryable(error)
            invalid = isinstance(error, (StructuredOutputError, OutputLimitError))
            if not transient and not invalid:
                raise error
            if isinstance(error, OutputLimitError):
                # 保持官方输出上限；截断不是合法答案，也不自动扩大实验预算。
                error.request_ids = ids
                raise error
            if transient:
                if deadline is None:
                    deadline = time.monotonic() + recovery_seconds
                limited = getattr(error, "status_code", None) == 429
                delay = min(60 * (transport_failures + 1), 120) if limited else (
                    5, 15, 30, 60, 120)[min(transport_failures, 4)]
                transport_failures += 1
                response = getattr(error, "response", None)
                # 服务端给出秒数时遵守 Retry-After；其余使用本地退避，不猜测额度。
                try:
                    retry_after = float(response.headers.get("retry-after", "")) if response is not None else 0
                except ValueError:
                    retry_after = 0
                if math.isfinite(retry_after):
                    delay = max(delay, retry_after)
                delay += random.uniform(0, 5)  # 错开多个工作进程的重试时刻。
                wait = min(delay, max(0, deadline - time.monotonic()))
                self.log({"event":"transport_wait", "request_id":request_id, "stage":stage,
                          "model":options.get("model"), "seconds":wait,
                          "reason":redact(str(error)), "recovery_window_seconds":recovery_seconds})
                print(f"[{self.binding['episode']}/{stage}] 接口暂不可用，等待 {wait:.0f} 秒后重试；"
                      f"恢复窗口剩余 {max(0, deadline-time.monotonic()):.0f} 秒", flush=True)
                while wait > 0:
                    part = min(60, wait)
                    time.sleep(part)
                    wait -= part
                continue
            invalid_outputs += 1
            if invalid_outputs >= 3:
                error.request_ids = ids
                raise error


def audited_client_factory(original):
    def create(*args, **kwargs):
        from httpx2 import Timeout
        kwargs["max_retries"] = 0
        # 连接允许等待五分钟；其余保留客户端原有的十分钟，不缩短长请求。
        kwargs.setdefault("timeout", Timeout(600, connect=300))
        client = original(*args, **kwargs)
        audit = active()
        chat, embedding = client.chat.completions.create, client.embeddings.create
        # 评判器跨线程调用同一客户端，绑定样本而不是依赖进程级“当前样本”。
        client.chat.completions.create = lambda **values: audit.call(chat, "llm", values)
        client.embeddings.create = lambda **values: audit.call(embedding, "embedding", values)
        return client
    return create
