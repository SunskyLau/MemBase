"""从 MemBase 消息（message）中抽取证据原文和原子事实。"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..inference_utils.backends import get_interface_for_inference
from ..model_types.dataset import Message
from .models import AtomicFact, EvidenceQuote
from .structured_output import request_validated_json


FACT_EXTRACTION_PROMPT = """You extract memory-relevant atomic facts from conversation messages.

Return exactly one JSON object with this shape:
{
  "facts": [
    {
      "message_id": "the source message id",
      "quote": "an exact substring copied from that message",
      "content": "one independently updateable fact",
      "entities": ["entities mentioned by the fact"],
      "event_time": "the time when the fact happened, or null"
    }
  ]
}

Rules:
- Return zero or more facts.
- Ignore greetings, acknowledgements, and content with no factual, preference, event, plan,
  commitment, or update information.
- Treat the input as a chronological record, not as a summary of the current state.
- Extract every fact as it was stated in its source message, even if a later message corrects,
  contradicts, supersedes, or retracts it.
- Do not omit, merge, or rewrite an earlier fact using information from a later message.
- Return facts in the same order as their source messages.
- Keep both long-lived and temporary facts when they may matter in later interactions.
- Each fact must express one independently updateable proposition.
- If one message contains multiple independently updateable propositions, extract each
  proposition as a separate fact.
- The quote must be copied verbatim from exactly one source message.
- The fact may normalize wording, but must not add information unsupported by the quote.
- Preserve uncertainty in the fact content. Do not turn "maybe" into a certainty.
- Use null for event_time when the event time is not clear.
- Do not generate ids, statuses, evidence links, or version relations.
- Output JSON only, without Markdown fences or explanatory text.
"""


class ExtractedFact(BaseModel):
    """语言模型（language model）返回的一条事实抽取结果。"""

    message_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    content: str = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)
    event_time: str | None = None


class FactExtractionOutput(BaseModel):
    """语言模型（language model）一次调用返回的全部事实。"""

    facts: list[ExtractedFact] = Field(default_factory=list)


class FactExtractor:
    """将消息（message）转换为证据原文（evidence quote）和原子事实（atomic fact）。"""

    def __init__(
        self,
        model_name: str,
        api_keys: list[str] | str | None = None,
        base_urls: list[str] | str | None = None,
        interface: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self._interface = interface or get_interface_for_inference(
            model=model_name,
            api_keys=api_keys,
            base_urls=base_urls,
        )

    def extract(
        self,
        messages: list[Message],
        session_id: str | None = None,
        message_offset: int = 0,
    ) -> list[tuple[EvidenceQuote, AtomicFact]]:
        """从一组有序消息（message）中抽取事实。

        ``messages`` 可以包含一条消息（message），也可以包含一个完整会话
        （session）。每条结果仍然通过 ``message_id`` 指向唯一的原始消息。
        ``message_offset`` 表示第一条输入消息在会话（session）中的位置；处理完整
        会话（session）时保持为 0，处理后续消息片段时传入对应起点。
        """

        message_data = [
            {
                "message_id": message.id,
                "speaker": message.name,
                "role": message.role,
                "timestamp": message.timestamp,
                "content": message.content,
            }
            for message in messages
        ]
        request_messages = [
            {
                "role": "system",
                "content": FACT_EXTRACTION_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    message_data,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        messages_by_id = {message.id: message for message in messages}
        message_indices = {
            message.id: message_offset + index
            for index, message in enumerate(messages)
        }

        def validate_output(raw: dict[str, Any]) -> FactExtractionOutput:
            output = FactExtractionOutput.model_validate(raw)
            for fact in output.facts:
                message = messages_by_id[fact.message_id]
                if fact.quote not in message.content:
                    raise ValueError(
                        f"Quote is not present in message '{message.id}'."
                    )
            return output

        output = request_validated_json(
            self._interface,
            request_messages,
            validate_output,
            context="fact extraction",
        )
        extracted_facts = sorted(
            output.facts,
            key=lambda fact: message_indices[fact.message_id],
        )

        results = []
        for extracted_fact in extracted_facts:
            message = messages_by_id[extracted_fact.message_id]
            evidence_quote = EvidenceQuote(
                message_id=message.id,
                session_id=session_id,
                message_index=message_indices[message.id],
                speaker=message.name,
                quote=extracted_fact.quote,
                timestamp=message.timestamp,
            )
            fact = AtomicFact(
                content=extracted_fact.content,
                entities=extracted_fact.entities,
                evidence_quote_id=evidence_quote.id,
                mention_time=evidence_quote.timestamp,
                event_time=extracted_fact.event_time,
            )
            results.append((evidence_quote, fact))

        return results
