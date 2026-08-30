"""判断一条新原子事实（atomic fact）如何更新当前事实记忆。"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..inference_utils.backends import get_interface_for_inference
from .models import AtomicFact, EvidenceQuote, FactUpdate, FactUpdateAction
from .structured_output import request_validated_json


FACT_RECONCILIATION_PROMPT = """You reconcile one newly stated atomic fact with current active facts.

Return exactly one JSON object:
{
  "action": "add | duplicate | supersede | retract",
  "target_fact_id": "a candidate fact id, or null for add"
}

Definitions:
- ADD: the new fact expresses independent information that can coexist with all candidates.
- DUPLICATE: the new fact repeats the same proposition, including the same uncertainty.
- SUPERSEDE: the new fact gives a newer value or state for the same subject, attribute,
  scope, and time frame, so one current candidate must become historical.
- RETRACT: the speaker explicitly withdraws one candidate without asserting a stable
  replacement value.

Rules:
- Topic or entity overlap alone is not an update relation.
- Different events or time periods may coexist and should be ADD.
- A negative current state can SUPERSEDE a positive current state; it is not merely a
  retraction.
- Preserve uncertainty. "Maybe" does not supersede a certain fact unless the speaker
  explicitly corrects or withdraws it.
- For DUPLICATE, SUPERSEDE, or RETRACT, choose exactly one id from candidate_facts.
- For ADD, target_fact_id must be null.
- Use only the supplied facts. Output JSON only, without Markdown or explanation.

Example:
- current: "My charity 5K personal best is 27:12"
- new: "My charity 5K personal best is 25:50"
- result: supersede the 27:12 fact.
"""


class FactReconciler:
    """把新事实与候选当前事实归并为一个事实版本操作。"""

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

    def reconcile(
        self,
        new_fact: AtomicFact,
        new_evidence_quote: EvidenceQuote,
        candidate_facts: list[AtomicFact],
    ) -> FactUpdate:
        """判断新事实相对候选当前事实应执行的操作。

        候选检索由调用方负责。这里假设 ``candidate_facts`` 只包含当前有效事实，
        并用 ``new_evidence_quote`` 保留纠正、撤回等原始话语信号。该方法只完成
        一次关系判断，不直接修改事实存储。
        """

        if not candidate_facts:
            return FactUpdate(action=FactUpdateAction.ADD)

        request = {
            "new_fact": {
                **self._serialize_fact(new_fact),
                "source_quote": new_evidence_quote.quote,
            },
            "candidate_facts": [
                self._serialize_fact(fact) for fact in candidate_facts
            ],
        }
        candidate_ids = {fact.id for fact in candidate_facts}

        def validate_update(raw: dict[str, Any]) -> FactUpdate:
            normalized = dict(raw)
            normalized["action"] = normalized["action"].lower()
            update = FactUpdate.model_validate(normalized)
            if (
                update.target_fact_id is not None
                and update.target_fact_id not in candidate_ids
            ):
                raise ValueError(
                    f"Target fact '{update.target_fact_id}' is not a candidate"
                )
            return update

        return request_validated_json(
            self._interface,
            [
                {"role": "system", "content": FACT_RECONCILIATION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False, indent=2),
                },
            ],
            validate_update,
            context="fact reconciliation",
        )

    @staticmethod
    def _serialize_fact(fact: AtomicFact) -> dict[str, object]:
        return {
            "id": fact.id,
            "content": fact.content,
            "entities": fact.entities,
            "mention_time": fact.mention_time,
            "event_time": fact.event_time,
        }
