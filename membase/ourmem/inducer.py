"""从事实和低层主张中归纳并验证成立依据（justification）。"""

from __future__ import annotations

import json
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from ..inference_utils.backends import get_interface_for_inference
from .models import (
    AtomicFact,
    ClaimClause,
    ClaimKind,
    ClaimPolarity,
    ClaimValue,
    ClaimVersion,
    EvidenceQuote,
    Justification,
)
from .retriever import SemanticCandidate


JUSTIFICATION_INDUCTION_PROMPT = """You induce reusable derived memory from a bounded support set.

Return JSON:
{
  "proposals": [
    {
      "existing_claim_key": "a supplied claim key or null",
      "claim_text": "one reusable derived claim",
      "claim_kind": "state | decision",
      "polarity": "positive | negative",
      "support_version_ids": ["2 or 3 supplied ids"]
    }
  ]
}

Rules:
- Return at most 3 proposals; an empty list is valid.
- Every proposal must use the anchor id and 1 or 2 additional supplied supports.
- The full support set must jointly justify the claim.
- Every listed support must be necessary; do not add merely related evidence.
- The claim must save future reasoning, not copy or paraphrase one support.
- Prefer a concrete stable profile, constraint, or decision over a vague statement that the
  evidence "suggests" commitment, interest, or professional development.
- Do not infer a motivation or causal explanation unless the supports state it explicitly.
- Claims are query-independent and must not mention the evaluation question.
- Reuse an existing_claim_key only for the same semantic claim identity.
- Use only supplied ids and existing claim keys.
- Output JSON only.
"""


JUSTIFICATION_VERIFICATION_PROMPT = """You verify proposed derived-memory justifications.

For each proposal, decide whether the complete support set is sufficient for the claim and
which support ids are individually necessary. Return JSON:
{
  "decisions": [
    {
      "proposal_index": 0,
      "supported": true,
      "necessary_support_ids": ["ids that are necessary"]
    }
  ]
}

Use only the supplied content. Mark supported=false when the claim adds unsupported detail,
uses the wrong entity or time, remains uncertain, or turns concrete facts into a vague
"indicates/suggests" meta-claim with no additional reusable value. Output JSON only.
"""


DEFEATER_DETECTION_PROMPT = """You identify current derived claims directly defeated by one new fact.

Return JSON:
{
  "defeated_claim_keys": ["keys selected from candidate_claims"]
}

A claim is defeated only when the new fact contradicts it, invalidates a necessary condition,
or makes its stated scope no longer hold. Topic similarity is not enough. Select only supplied
claim keys. An empty list is valid. Output JSON only.
"""


class JustificationProposal(BaseModel):
    existing_claim_key: str | None = None
    claim_text: str = Field(min_length=1)
    claim_kind: ClaimKind
    polarity: ClaimPolarity = ClaimPolarity.POSITIVE
    support_version_ids: list[str] = Field(min_length=2, max_length=3)


class JustificationDecision(BaseModel):
    proposal_index: int = Field(ge=0)
    supported: bool
    necessary_support_ids: list[str] = Field(default_factory=list)


class DefeaterOutput(BaseModel):
    defeated_claim_keys: list[str] = Field(default_factory=list)


class JustificationInducer:
    """批量提出、验证成立依据，并检测新反例。"""

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

    def induce(
        self,
        anchor_id: str,
        support_candidates: list[SemanticCandidate],
        existing_claims: list[ClaimVersion],
    ) -> list[Justification]:
        """从有界候选集合中返回通过必要性验证的成立依据。"""

        if len(support_candidates) < 2:
            return []

        support_by_id = {item.id: item for item in support_candidates}
        existing_by_key = {claim.claim_key: claim for claim in existing_claims}
        request = {
            "anchor_id": anchor_id,
            "support_candidates": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "text": item.text,
                }
                for item in support_candidates
            ],
            "existing_claims": [
                {
                    "claim_key": claim.claim_key,
                    "claim_text": claim.clause.value.proposition,
                    "kind": claim.kind.value,
                }
                for claim in existing_claims
            ],
        }
        raw_output = self._call_json(JUSTIFICATION_INDUCTION_PROMPT, request)
        proposals = []
        for raw_proposal in raw_output.get("proposals", [])[:3]:
            if not isinstance(raw_proposal, dict):
                continue
            normalized = dict(raw_proposal)
            if isinstance(normalized.get("claim_kind"), str):
                normalized["claim_kind"] = normalized["claim_kind"].lower()
            polarity = normalized.get("polarity", "positive")
            if isinstance(polarity, str):
                normalized["polarity"] = polarity.lower()
            try:
                proposals.append(JustificationProposal.model_validate(normalized))
            except ValidationError:
                continue

        valid_proposals = [
            proposal
            for proposal in proposals
            if anchor_id in proposal.support_version_ids
            and set(proposal.support_version_ids).issubset(support_by_id)
            and len(proposal.support_version_ids)
            == len(set(proposal.support_version_ids))
            and (
                proposal.existing_claim_key is None
                or proposal.existing_claim_key in existing_by_key
            )
        ]
        if not valid_proposals:
            return []

        verification_request = {
            "proposals": [
                {
                    "proposal_index": index,
                    "claim_text": proposal.claim_text,
                    "polarity": proposal.polarity.value,
                    "supports": [
                        {
                            "id": support_id,
                            "text": support_by_id[support_id].text,
                        }
                        for support_id in proposal.support_version_ids
                    ],
                }
                for index, proposal in enumerate(valid_proposals)
            ]
        }
        raw_verification = self._call_json(
            JUSTIFICATION_VERIFICATION_PROMPT,
            verification_request,
        )
        decisions = []
        for raw_decision in raw_verification.get("decisions", []):
            try:
                decisions.append(
                    JustificationDecision.model_validate(raw_decision)
                )
            except ValidationError:
                continue
        decision_by_index = {
            decision.proposal_index: decision for decision in decisions
        }

        justifications: list[Justification] = []
        used_claim_keys: set[str] = set()
        for index, proposal in enumerate(valid_proposals):
            decision = decision_by_index.get(index)
            if (
                decision is None
                or not decision.supported
                or set(decision.necessary_support_ids)
                != set(proposal.support_version_ids)
            ):
                continue
            claim_key = proposal.existing_claim_key or f"claim-{uuid4()}"
            if claim_key in used_claim_keys:
                continue
            used_claim_keys.add(claim_key)
            justifications.append(
                Justification(
                    conclusion_key=claim_key,
                    conclusion_kind=proposal.claim_kind,
                    support_version_ids=proposal.support_version_ids,
                    clause=ClaimClause(
                        value=ClaimValue(
                            proposition=proposal.claim_text,
                            polarity=proposal.polarity,
                        )
                    ),
                )
            )
        return justifications

    def detect_defeated_claim_keys(
        self,
        new_fact: AtomicFact,
        new_evidence_quote: EvidenceQuote,
        candidate_claims: list[ClaimVersion],
        supporting_facts_by_claim: dict[str, list[AtomicFact]],
    ) -> list[str]:
        """判断新事实是否直接推翻候选当前主张。"""

        if not candidate_claims:
            return []
        request = {
            "new_fact": {
                "content": new_fact.content,
                "source_quote": new_evidence_quote.quote,
                "mention_time": new_fact.mention_time,
                "event_time": new_fact.event_time,
            },
            "candidate_claims": [
                {
                    "claim_key": claim.claim_key,
                    "claim_text": claim.clause.value.proposition,
                    "polarity": claim.clause.value.polarity.value,
                    "current_supports": [
                        fact.content
                        for fact in supporting_facts_by_claim[claim.claim_key]
                    ],
                }
                for claim in candidate_claims
            ],
        }
        output = DefeaterOutput.model_validate(
            self._call_json(DEFEATER_DETECTION_PROMPT, request)
        )
        candidate_keys = {claim.claim_key for claim in candidate_claims}
        return list(dict.fromkeys(
            claim_key
            for claim_key in output.defeated_claim_keys
            if claim_key in candidate_keys
        ))

    def _call_json(self, prompt: str, request: dict) -> dict:
        response = self._interface(
            [
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(request, ensure_ascii=False, indent=2),
                    },
                ]
            ],
            temperature=0.0,
            stream=False,
        )
        return json.loads(response["content"])
