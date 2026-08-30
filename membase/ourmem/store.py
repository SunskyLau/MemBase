"""OurMem 的确定性事实存储（deterministic fact store）。

这个存储可以想象成两个抽屉：

- 一个抽屉保存证据原文（evidence quote）；
- 一个抽屉保存原子事实（atomic fact）。

它只负责维护对象之间的引用和事实生命周期，不负责从自然语言中抽取事实，
也不负责判断两条事实在语义上是否构成替代。上层逻辑确定关系后，再调用这里
的方法执行确定性的状态变化。
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import AtomicFact, EvidenceQuote, FactStatus


class OurMemStore:
    """在内存中保存证据原文（evidence quote）和原子事实（atomic fact）。"""

    def __init__(self) -> None:
        self._evidence_quotes: dict[str, EvidenceQuote] = {}
        self._facts: dict[str, AtomicFact] = {}

    def add_evidence_quote(self, evidence_quote: EvidenceQuote) -> None:
        """添加一条证据原文（evidence quote）。"""

        self._evidence_quotes[evidence_quote.id] = evidence_quote

    def get_evidence_quote(self, evidence_quote_id: str) -> EvidenceQuote:
        """根据标识读取证据原文（evidence quote）。"""

        return self._evidence_quotes[evidence_quote_id]

    def get_evidence_quotes(self) -> list[EvidenceQuote]:
        """按照写入顺序返回全部证据原文（evidence quote）。"""

        return list(self._evidence_quotes.values())

    def add_fact(self, fact: AtomicFact) -> None:
        """添加一条普通的新原子事实（atomic fact）。

        该方法只接受没有历史版本的新增事实。需要替代旧事实时，应调用
        :meth:`supersede_fact`，让旧状态和版本指针在同一个操作中完成更新。
        """

        self.get_evidence_quote(fact.evidence_quote_id)
        self._facts[fact.id] = fact

    def get_fact(self, fact_id: str) -> AtomicFact:
        """根据标识读取原子事实（atomic fact），包括历史版本。"""

        return self._facts[fact_id]

    def get_active_facts(self) -> list[AtomicFact]:
        """按照写入顺序返回当前仍然有效的原子事实（atomic fact）。"""

        return [
            fact
            for fact in self._facts.values()
            if fact.status is FactStatus.ACTIVE
        ]

    def save(self, path: str | Path) -> None:
        """将当前记忆保存为可读的 JSON 快照（JSON snapshot）。

        JSON 中保存完整对象，而不是只保存当前有效事实，因此历史版本、撤回状态
        和事实版本链（fact version chain）都能够在下一次运行时恢复。
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, list[dict]]:
        """返回可以直接写入 JSON 的事实存储状态。"""

        return {
            "evidence_quotes": [
                evidence_quote.model_dump(mode="json")
                for evidence_quote in self._evidence_quotes.values()
            ],
            "facts": [
                fact.model_dump(mode="json")
                for fact in self._facts.values()
            ],
        }

    @classmethod
    def load(cls, path: str | Path) -> OurMemStore:
        """从 JSON 快照（JSON snapshot）恢复事实存储（fact store）。

        加载时重新经过 Pydantic 数据模型（data model）解析，恢复证据原文
        （evidence quote）、原子事实（atomic fact）及其历史状态。
        """

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> OurMemStore:
        """从 JSON 对象恢复事实存储状态。"""

        store = cls()

        for raw_evidence_quote in data["evidence_quotes"]:
            evidence_quote = EvidenceQuote.model_validate(raw_evidence_quote)
            store._evidence_quotes[evidence_quote.id] = evidence_quote

        for raw_fact in data["facts"]:
            fact = AtomicFact.model_validate(raw_fact)
            store._facts[fact.id] = fact

        return store

    def supersede_fact(self, old_fact_id: str, new_fact: AtomicFact) -> None:
        """用新原子事实（atomic fact）替代当前有效的旧原子事实（atomic fact）。

        这个方法不判断两条事实是否真的构成语义更新。它假设上层已经完成旧事实
        匹配和关系判断，只负责执行下面三个确定性变化：

        1. 旧原子事实（atomic fact）变为 ``SUPERSEDED``；
        2. 新原子事实（atomic fact）保持 ``ACTIVE``；
        3. 新原子事实（atomic fact）通过 ``supersedes_fact_id`` 指向旧版本。
        """

        old_fact = self.get_fact(old_fact_id)
        if old_fact.status is not FactStatus.ACTIVE:
            raise ValueError(
                f"Only an active fact can be superseded, but '{old_fact_id}' "
                f"is '{old_fact.status.value}'."
            )

        self.get_evidence_quote(new_fact.evidence_quote_id)

        new_fact.supersedes_fact_id = old_fact.id
        new_fact.status = FactStatus.ACTIVE
        old_fact.status = FactStatus.SUPERSEDED
        self._facts[new_fact.id] = new_fact

    def retract_fact(
        self,
        fact_id: str,
        evidence_quote_id: str | None = None,
    ) -> None:
        """撤回一条当前有效的原子事实（atomic fact），但保留历史记录。"""

        fact = self.get_fact(fact_id)
        if fact.status is not FactStatus.ACTIVE:
            raise ValueError(
                f"Only an active fact can be retracted, but '{fact_id}' "
                f"is '{fact.status.value}'."
            )
        if evidence_quote_id is not None:
            self.get_evidence_quote(evidence_quote_id)
        fact.status = FactStatus.RETRACTED
        fact.retracted_by_evidence_quote_id = evidence_quote_id
