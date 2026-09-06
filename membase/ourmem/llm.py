"""模型调用、结构校验、重试和请求账本的共同边界。

每次真正外发之前先占用额度，失败也计数。SDK 自身不重试，避免业务层和
SDK 各重试一遍。账本使用 SQLite，使重启和并行样本不能重新获得免费额度。
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .models import utc_now

T = TypeVar("T")


class ModelCallError(RuntimeError):
    pass


class ContextLimitError(ModelCallError):
    pass


class OutputLimitError(ModelCallError):
    pass


class StructuredOutputError(ModelCallError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class TextResult(str):
    """保留官方原始短答案及终止原因，不把截断悄悄当作完整答案。"""

    def __new__(cls, text: str, finish_reason: str, request_id: int):
        result = super().__new__(cls, text)
        result.finish_reason = finish_reason
        result.request_id = request_id
        return result


class RequestBudget:
    def __init__(self, max_llm_requests: int | None = None,
                 max_embedding_requests: int | None = None,
                 ledger_path: str | Path | None = None) -> None:
        if any(value is not None and value < 0 for value in (max_llm_requests, max_embedding_requests)):
            raise ValueError("Request limits cannot be negative")
        self._lock = threading.RLock()
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.ledger_path) if self.ledger_path else ":memory:",
                                   check_same_thread=False, timeout=30)
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS limits (kind TEXT PRIMARY KEY, maximum INTEGER);
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                stage TEXT NOT NULL, model TEXT NOT NULL, started_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'reserved', usage TEXT, error TEXT,
                elapsed REAL
            );
        """)
        proposed = {"llm": max_llm_requests, "embedding": max_embedding_requests}
        with self._db:
            for kind, maximum in proposed.items():
                row = self._db.execute("SELECT maximum FROM limits WHERE kind=?", (kind,)).fetchone()
                if row is not None and maximum is not None and maximum != row[0]:
                    raise ValueError("Request budget differs from the existing ledger")
                self._db.execute("INSERT OR IGNORE INTO limits VALUES (?,?)", (kind, maximum))
        self.limits = dict(self._db.execute("SELECT kind, maximum FROM limits"))

    def reserve(self, kind: str, stage: str, model: str) -> int:
        if kind not in self.limits:
            raise ValueError(f"Unknown request kind: {kind}")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            counts = dict(self._db.execute("SELECT kind, COUNT(*) FROM requests GROUP BY kind"))
            # 受限验证达到任一种请求的上限后，整个验证停止，而不是换一种继续花费。
            for category, maximum in self.limits.items():
                if maximum is not None and counts.get(category, 0) >= maximum:
                    raise BudgetExceeded(f"{category} request budget exhausted ({maximum})")
            return self._db.execute(
                "INSERT INTO requests(kind,stage,model,started_at) VALUES (?,?,?,?)",
                (kind, stage, model, utc_now()),
            ).lastrowid

    def finish(self, request_id: int, *, usage: dict | None = None,
               error: str | None = None, elapsed: float = 0) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE requests SET status=?,usage=?,error=?,elapsed=? WHERE id=?",
                             ("failed" if error else "complete",
                              json.dumps(usage) if usage is not None else None,
                              error, elapsed, request_id))

    def summary(self) -> dict:
        with self._lock:
            counts = dict(self._db.execute("SELECT kind, COUNT(*) FROM requests GROUP BY kind"))
            last_request_id = self._db.execute("SELECT COALESCE(MAX(id),0) FROM requests").fetchone()[0]
            totals: dict[str, int] = {}
            unknown_usage = 0
            for (usage,) in self._db.execute("SELECT usage FROM requests"):
                if usage is None:
                    unknown_usage += 1
                else:
                    for name, value in json.loads(usage).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            totals[name] = totals.get(name, 0) + value
        return {"llm_requests": counts.get("llm", 0), "embedding_requests": counts.get("embedding", 0),
                "limits": self.limits, "usage": totals, "requests_without_usage": unknown_usage,
                "last_request_id": last_request_id}

    def close(self) -> None:
        self._db.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ModelClient:
    def __init__(self, config, budget: RequestBudget | None = None,
                 log_path: str | Path | None = None, backend=None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self._owns_budget = budget is None
        self.budget = budget if budget is not None else RequestBudget()
        self.log_path = Path(log_path) if log_path is not None else None
        self._backend = backend
        self._owns_backend = backend is None
        self._backend_lock = threading.Lock()
        self._sleep = sleep
        self._log_lock = threading.Lock()
        self._tokenizer = None

    @property
    def backend(self):
        with self._backend_lock:
            if self._backend is None:
                from openai import OpenAI
                self._backend = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                                       max_retries=0, timeout=self.config.request_timeout)
        return self._backend

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            from .tokenization import get_encoder
            self._tokenizer = get_encoder()
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _safe(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self.config.api_key, "[REDACTED]") if self.config.api_key else value
        if isinstance(value, dict):
            return {key: self._safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._safe(item) for item in value]
        return value

    def _log(self, record: dict) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self._safe(record), ensure_ascii=False, default=_json_default) + "\n")

    @staticmethod
    def _usage(response) -> dict | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return usage.model_dump() if hasattr(usage, "model_dump") else dict(vars(usage))

    @staticmethod
    def _retryable(error: Exception) -> bool:
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        return isinstance(error, APIStatusError) and (error.status_code == 429 or error.status_code >= 500)

    def _chat_once(self, messages: list[dict], *, stage: str, model: str,
                   temperature: float, max_tokens: int, json_mode: bool,
                   allow_truncated: bool = False) -> str:
        input_count = sum(self.count_tokens(message["content"]) for message in messages) + 32
        if input_count > self.config.max_context_tokens:
            raise ContextLimitError(f"{stage} input {input_count} exceeds {self.config.max_context_tokens}")
        backend = self.backend
        request_id = self.budget.reserve("llm", stage, model)
        started = time.monotonic()
        usage = None
        record = {"id": request_id, "kind": "llm", "stage": stage, "model": model,
                  "messages": messages, "estimated_input_tokens": input_count,
                  "max_tokens": max_tokens, "temperature": temperature,
                  "prompt_hash": sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()}
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens}
            if stage not in {"answer", "judge"}:
                kwargs["seed"] = self.config.seed
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = backend.chat.completions.create(**kwargs)
            usage = self._usage(response)
            choice = response.choices[0]
            record["response"] = choice.message.content
            record["finish_reason"] = choice.finish_reason
            if choice.finish_reason == "length" and not allow_truncated:
                raise OutputLimitError(f"{stage} output was truncated")
            if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
                raise ModelCallError(f"{stage} was refused or filtered")
            if not choice.message.content or not choice.message.content.strip():
                raise StructuredOutputError(f"{stage} returned empty content")
            result = TextResult(choice.message.content.strip(), choice.finish_reason, request_id)
        except Exception as error:
            message = self._safe(str(error))
            self.budget.finish(request_id, usage=usage, error=message, elapsed=time.monotonic() - started)
            self._log({**record, "error": message, "usage": usage})
            raise
        self.budget.finish(request_id, usage=usage, elapsed=time.monotonic() - started)
        self._log({**record, "usage": usage, "elapsed": time.monotonic() - started})
        return result

    def text(self, prompt: str, *, stage: str = "answer", model: str | None = None,
             temperature: float = 0, max_tokens: int | None = None,
             system: str | None = None, allow_truncated: bool = False) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        for attempt in range(self.config.max_llm_retries + 1):
            try:
                return self._chat_once(messages, stage=stage, model=model or self.config.answer_model,
                                       temperature=temperature,
                                       max_tokens=max_tokens or self.config.max_model_output_tokens,
                                       json_mode=False, allow_truncated=allow_truncated)
            except Exception as error:
                if not self._retryable(error) or attempt == self.config.max_llm_retries:
                    raise
                self._sleep(2 ** attempt)
        raise AssertionError("Unreachable retry state")

    def request_json(self, stage: str, prompt: str, payload: Any = None,
                     validator: Callable[[dict], T] | None = None,
                     model: str | None = None) -> T | dict:
        # payload=None 用于官方评分：不在其原版提示词后添加任何文字。
        messages = ([{"role": "user", "content": prompt}] if payload is None else [
            {"role": "system", "content": prompt if "json" in prompt.casefold() else prompt + "\nReturn a JSON object only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=_json_default)},
        ])
        last_validation = None
        for attempt in range(self.config.max_llm_retries + 1):
            request_id = None
            content = None
            try:
                content = self._chat_once(messages, stage=stage, model=model or self.config.model_name,
                                          temperature=self.config.memory_temperature,
                                          max_tokens=self.config.max_model_output_tokens, json_mode=True)
                request_id = getattr(content, "request_id", None)
                if content.startswith("```") and content.endswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                raw = json.loads(content)
                if not isinstance(raw, dict):
                    raise ValueError("Expected a JSON object")
                return validator(raw) if validator is not None else raw
            except (ValueError, StructuredOutputError) as error:
                last_validation = error
                self._log({"id": request_id, "stage": stage, "validation_error": str(error)})
                if attempt == self.config.max_llm_retries:
                    raise StructuredOutputError(self._safe(str(error))) from error
                if payload is not None:
                    correction = "Return corrected JSON only. Previous output failed validation: " + self._safe(str(error))
                    # 让模型修正实际失败的结果，避免看不到原输出而重新生成、修一处退一处。
                    previous = [{"role": "assistant", "content": str(content)}] if content is not None else []
                    messages = messages[:2] + previous + [{"role": "user", "content": correction}]
            except Exception as error:
                if not self._retryable(error) or attempt == self.config.max_llm_retries:
                    raise
                self._sleep(2 ** attempt)
        raise StructuredOutputError(str(last_validation))

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.config.embedding_batch_size):
            batch = texts[start:start + self.config.embedding_batch_size]
            for attempt in range(self.config.max_llm_retries + 1):
                backend = self.backend
                request_id = self.budget.reserve("embedding", "embedding", self.config.embedding_model_name)
                started, usage = time.monotonic(), None
                try:
                    response = backend.embeddings.create(model=self.config.embedding_model_name, input=batch)
                    usage = self._usage(response)
                    ordered = sorted(response.data, key=lambda item: item.index)
                    if [item.index for item in ordered] != list(range(len(batch))):
                        raise StructuredOutputError("Embedding response does not match input items")
                    try:
                        vectors = np.asarray([item.embedding for item in ordered], dtype=np.float32)
                    except ValueError as error:
                        raise StructuredOutputError("Embedding vectors have inconsistent dimensions") from error
                    if vectors.ndim != 2 or not np.isfinite(vectors).all() or not (np.linalg.norm(vectors, axis=1) > 0).all():
                        raise StructuredOutputError("Embedding response contains invalid vectors")
                except Exception as error:
                    self.budget.finish(request_id, usage=usage, error=self._safe(str(error)),
                                       elapsed=time.monotonic() - started)
                    self._log({"id": request_id, "kind": "embedding", "stage": "embedding",
                               "model": self.config.embedding_model_name,
                               "text_hashes": [sha256(text.encode()).hexdigest() for text in batch],
                               "usage": usage, "error": self._safe(str(error)),
                               "elapsed": time.monotonic() - started})
                    if (not self._retryable(error) and not isinstance(error, StructuredOutputError)) or attempt == self.config.max_llm_retries:
                        raise
                    self._sleep(2 ** attempt)
                else:
                    self.budget.finish(request_id, usage=usage, elapsed=time.monotonic() - started)
                    self._log({"id": request_id, "kind": "embedding", "model": self.config.embedding_model_name,
                               "text_hashes": [sha256(text.encode()).hexdigest() for text in batch], "usage": usage})
                    embeddings.extend(vectors.tolist())
                    break
        return embeddings

    def close(self) -> None:
        if self._backend is not None and self._owns_backend:
            self._backend.close()
        if self._owns_budget:
            self.budget.close()
