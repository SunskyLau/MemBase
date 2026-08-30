"""Retry and validate JSON returned by a language model."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


T = TypeVar("T")


def request_validated_json(
    interface: Callable[..., dict[str, Any]],
    messages: list[dict[str, str]],
    validator: Callable[[dict[str, Any]], T],
    context: str,
    max_attempts: int = 5,
    max_tokens: int = 4096,
) -> T:
    """Request JSON and retry when parsing or schema validation fails."""

    conversation = list(messages)
    attempt = 1
    api_failures = 0
    while attempt <= max_attempts:
        try:
            response = interface(
                [conversation],
                temperature=0.0,
                stream=False,
                max_tokens=max_tokens,
            )
        except (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ) as error:
            api_failures += 1
            delay = min(60, 2 ** min(api_failures, 6))
            print(
                f"API request failed for {context}; retrying in {delay}s "
                f"(transient failure {api_failures}): {error}"
            )
            time.sleep(delay)
            continue
        api_failures = 0
        content = response["content"]
        try:
            return validator(_parse_json_object(content))
        except (KeyError, TypeError, ValueError) as error:
            if attempt == max_attempts:
                raise
            print(
                f"Invalid structured output for {context}; "
                f"retrying ({attempt}/{max_attempts - 1}): {error}"
            )
            conversation.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed or validated. "
                            "Return the corrected JSON object only, with no Markdown or "
                            f"explanation. Validation error: {str(error)[:500]}"
                        ),
                    },
                ]
            )
            attempt += 1
    raise RuntimeError("Structured-output retry loop ended unexpectedly")


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("The response must be one JSON object")
    return parsed
