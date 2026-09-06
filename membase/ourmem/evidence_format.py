"""最终证据的纯渲染：相同的完整证明只显示一次，不切分证明内部前提。"""

from __future__ import annotations

import json

from .models import EvidenceBundle


def render_evidence(candidate_ids: list[str], packets: dict[str, dict],
                    bundles: dict[str, EvidenceBundle]) -> str:
    """调用方按完整候选包取舍；本函数仅去重和序列化，不决定证据是否足够。"""
    proof_ids: dict[str, str] = {}
    candidates = []

    def register(text: str) -> str:
        if text not in proof_ids:
            proof_ids[text] = f"proof-{len(proof_ids) + 1}"
        return proof_ids[text]

    for candidate_id in dict.fromkeys(candidate_ids):
        packet, bundle = packets[candidate_id], bundles[candidate_id]
        if not bundle.complete:
            raise ValueError(f"Candidate {candidate_id} has no complete evidence package")
        rendered = dict(packet)
        if packet["kind"] == "source":
            paths = rendered.pop("supporting_paths", [])
            rendered["supporting_path_ids"] = list(dict.fromkeys(register(text) for text in paths))
        elif packet["kind"] == "memory":
            rendered.pop("evidence", None)
            rendered["proof_ids"] = [register(bundle.text)]
        else:
            # 公开查询日期等非记忆证据保留原始正文，不变成记忆或推导结论。
            rendered["text"] = bundle.text
        candidates.append(rendered)
    result = {"candidates": candidates,
              "proofs": [{"id": proof_id, "text": text} for text, proof_id in proof_ids.items()]}
    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=lambda value: value.model_dump(mode="json"))
