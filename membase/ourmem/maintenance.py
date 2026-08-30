"""OurMem 的派生主张维护器。

当前版本为每个派生主张（derived claim）维护一条当前成立依据
（justification）。一条成立依据（justification）可以包含 2–3 个共同必要的事实或低层主张，
但同一主张不同时保存多套可替代依据。

自动发现派生主张（derived claim）和支持集合属于后续依据归纳器
（justification inducer）；这里仅实现版本更新、依赖定位和确定性级联。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .models import (
    AtomicFact,
    JustificationStatus,
    ClaimKind,
    ClaimStatus,
    ClaimVersion,
    FactStatus,
    Justification,
)
from .store import OurMemStore


@dataclass
class MaintenanceReport:
    """一次局部维护真正检查和改变了什么。"""

    touched_justification_ids: list[str] = field(default_factory=list)
    changed_claim_keys: list[str] = field(default_factory=list)
    created_claim_version_ids: list[str] = field(default_factory=list)


class ClaimMemory:
    """在原子事实（atomic fact）之上维护单一依据的派生主张。"""

    def __init__(self, fact_store: OurMemStore) -> None:
        self.fact_store = fact_store
        self._justifications: dict[str, Justification] = {}
        self._current_justification_id_by_claim: dict[str, str] = {}
        self._justification_ids_by_reference: dict[str, set[str]] = defaultdict(set)
        self._claim_versions: dict[str, ClaimVersion] = {}
        self._claim_version_ids_by_key: dict[str, list[str]] = defaultdict(list)
        self._current_claim_version_id: dict[str, str] = {}

    def add_justification(
        self,
        justification: Justification,
    ) -> MaintenanceReport:
        """为一个新派生主张写入第一条当前成立依据（justification）。"""

        if justification.conclusion_key in self._current_justification_id_by_claim:
            raise ValueError("A claim can have only one current justification")
        self._validate_justification(justification)
        self._register_justification(justification)
        self._refresh_justification_status(justification)

        report = MaintenanceReport(touched_justification_ids=[justification.id])
        self._propagate_claim_changes([justification.conclusion_key], report)
        return report

    def replace_justification(
        self,
        old_justification_id: str,
        new_justification: Justification,
    ) -> MaintenanceReport:
        """用新的单一依据替代一个派生主张的旧依据。"""

        old_justification = self.get_justification(old_justification_id)
        current_id = self._current_justification_id_by_claim.get(
            old_justification.conclusion_key
        )
        if current_id != old_justification_id:
            raise ValueError("Only the current justification can be replaced")
        if (
            new_justification.conclusion_key != old_justification.conclusion_key
            or new_justification.conclusion_kind is not old_justification.conclusion_kind
        ):
            raise ValueError("A replacement must preserve claim identity and kind")

        self._validate_justification(
            new_justification,
            replacing_claim_key=old_justification.conclusion_key,
        )
        self._unregister_justification(old_justification)
        old_justification.status = JustificationStatus.SUPERSEDED
        new_justification.supersedes_justification_id = old_justification.id
        self._register_justification(new_justification)
        self._refresh_justification_status(new_justification)

        report = MaintenanceReport(
            touched_justification_ids=[old_justification.id, new_justification.id]
        )
        self._propagate_claim_changes([new_justification.conclusion_key], report)
        return report

    def supersede_fact(
        self,
        old_fact_id: str,
        new_fact: AtomicFact,
    ) -> MaintenanceReport:
        """替代一条原子事实（atomic fact），并立即维护下游主张。"""

        self.fact_store.supersede_fact(old_fact_id, new_fact)
        return self.on_version_changed(old_fact_id)

    def retract_fact(
        self,
        fact_id: str,
        evidence_quote_id: str | None = None,
    ) -> MaintenanceReport:
        """撤回一条原子事实（atomic fact），并立即维护下游主张。"""

        self.fact_store.retract_fact(fact_id, evidence_quote_id)
        return self.on_version_changed(fact_id)

    def on_version_changed(self, version_id: str) -> MaintenanceReport:
        """从一个已变化的事实或主张版本启动局部级联。"""

        report = MaintenanceReport()
        affected_claim_keys: set[str] = set()
        for justification_id in sorted(
            self._justification_ids_by_reference.get(version_id, set())
        ):
            justification = self._justifications[justification_id]
            self._refresh_justification_status(justification)
            report.touched_justification_ids.append(justification_id)
            affected_claim_keys.add(justification.conclusion_key)

        self._propagate_claim_changes(sorted(affected_claim_keys), report)
        return report

    def get_justification(self, justification_id: str) -> Justification:
        """读取一条当前或历史成立依据（justification）。"""

        return self._justifications[justification_id]

    def get_current_justification(
        self,
        claim_key: str,
    ) -> Justification | None:
        """读取一个派生主张（derived claim）的当前成立依据（justification）。"""

        justification_id = self._current_justification_id_by_claim.get(claim_key)
        if justification_id is None:
            return None
        return self._justifications[justification_id]

    def get_claim_version(self, version_id: str) -> ClaimVersion:
        """读取一个派生主张（derived claim）的历史版本。"""

        return self._claim_versions[version_id]

    def get_current_claim_version(self, claim_key: str) -> ClaimVersion | None:
        """读取派生主张（derived claim）的当前版本。"""

        version_id = self._current_claim_version_id.get(claim_key)
        if version_id is None:
            return None
        return self._claim_versions[version_id]

    def get_claim_versions(self, claim_key: str) -> list[ClaimVersion]:
        """按创建顺序返回派生主张（derived claim）的所有版本。"""

        return [
            self._claim_versions[version_id]
            for version_id in self._claim_version_ids_by_key.get(claim_key, [])
        ]

    def get_current_claims(self) -> list[ClaimVersion]:
        """按照创建顺序返回全部当前派生主张（derived claim）。"""

        return [
            claim_version
            for claim_version in self._claim_versions.values()
            if self._current_claim_version_id.get(claim_version.claim_key)
            == claim_version.id
        ]

    def get_supporting_fact_ids(self, claim_version_id: str) -> list[str]:
        """递归展开一个主张版本实际使用的原子事实（atomic fact）。"""

        claim_version = self._claim_versions[claim_version_id]
        return self._supporting_fact_ids_from_justification(
            claim_version.materialized_from_justification_id
        )

    def get_current_supporting_fact_ids(self, claim_key: str) -> list[str]:
        """展开一个当前主张正在使用的原子事实（atomic fact）。"""

        return self._supporting_fact_ids_from_justification(
            self._current_justification_id_by_claim[claim_key]
        )

    def current_supports_are_active(self, claim_key: str) -> bool:
        """判断当前成立依据（justification）的全部直接支持是否仍有效。"""

        justification = self.get_current_justification(claim_key)
        return all(
            self._is_reference_active(version_id)
            for version_id in justification.support_version_ids
        )

    def _supporting_fact_ids_from_justification(
        self,
        justification_id: str,
    ) -> list[str]:
        fact_ids: list[str] = []

        def expand_reference(version_id: str) -> None:
            claim_version = self._claim_versions.get(version_id)
            if claim_version is None:
                fact_ids.append(version_id)
                return
            justification = self._justifications[
                claim_version.materialized_from_justification_id
            ]
            for support_id in justification.support_version_ids:
                expand_reference(support_id)

        justification = self._justifications[justification_id]
        for support_id in justification.support_version_ids:
            expand_reference(support_id)
        return list(dict.fromkeys(fact_ids))

    def to_dict(self) -> dict:
        """返回可以直接写入 JSON 的依据与主张状态。"""

        return {
            "justifications": [
                item.model_dump(mode="json")
                for item in self._justifications.values()
            ],
            "claim_versions": [
                item.model_dump(mode="json")
                for item in self._claim_versions.values()
            ],
            "current_justification_id_by_claim": dict(
                self._current_justification_id_by_claim
            ),
            "current_claim_version_id": dict(self._current_claim_version_id),
        }

    @classmethod
    def from_dict(
        cls,
        fact_store: OurMemStore,
        data: dict,
    ) -> ClaimMemory:
        """从 JSON 对象恢复成立依据（justification）与主张状态。"""

        memory = cls(fact_store)
        for raw_justification in data["justifications"]:
            justification = Justification.model_validate(raw_justification)
            memory._justifications[justification.id] = justification
        for raw_claim_version in data["claim_versions"]:
            claim_version = ClaimVersion.model_validate(raw_claim_version)
            memory._claim_versions[claim_version.id] = claim_version
            memory._claim_version_ids_by_key[claim_version.claim_key].append(
                claim_version.id
            )
        memory._current_justification_id_by_claim.update(
            data["current_justification_id_by_claim"]
        )
        memory._current_claim_version_id.update(data["current_claim_version_id"])

        for justification_id in memory._current_justification_id_by_claim.values():
            justification = memory._justifications[justification_id]
            for version_id in (
                justification.support_version_ids
                + justification.explicit_defeater_version_ids
            ):
                memory._justification_ids_by_reference[version_id].add(
                    justification.id
                )
        return memory

    def get_claim_level(self, claim_key: str) -> int | None:
        """返回当前单一支持路径形成的只读层级。"""

        justification = self.get_current_justification(claim_key)
        if (
            justification is None
            or justification.status is not JustificationStatus.ACTIVE
        ):
            return None
        return self._active_claim_level(claim_key, {})

    def get_justification_depth(self, justification_id: str) -> int | None:
        """返回当前有效成立依据（justification）的派生深度。"""

        justification = self._justifications[justification_id]
        if (
            self._current_justification_id_by_claim.get(
                justification.conclusion_key
            )
            != justification.id
            or justification.status is not JustificationStatus.ACTIVE
        ):
            return None
        return 1 + max(
            self._reference_level(version_id, {})
            for version_id in justification.support_version_ids
        )

    def _validate_justification(
        self,
        justification: Justification,
        replacing_claim_key: str | None = None,
    ) -> None:
        if justification.id in self._justifications:
            raise ValueError(f"Justification '{justification.id}' already exists")
        if set(justification.support_version_ids).intersection(
            justification.explicit_defeater_version_ids
        ):
            raise ValueError("A version cannot be both support and defeater")
        if not all(
            self._is_reference_active(version_id)
            for version_id in justification.support_version_ids
        ):
            raise ValueError("A new justification must use current active supports")
        for version_id in justification.explicit_defeater_version_ids:
            self._is_reference_active(version_id)

        for version_id in justification.support_version_ids:
            support_claim_key = self._claim_key_for_version(version_id)
            if support_claim_key is None:
                continue
            if support_claim_key == justification.conclusion_key or self._has_path(
                justification.conclusion_key,
                support_claim_key,
                excluding_claim_key=replacing_claim_key,
            ):
                raise ValueError("The justification would create a cycle")

    def _register_justification(self, justification: Justification) -> None:
        self._justifications[justification.id] = justification
        self._current_justification_id_by_claim[
            justification.conclusion_key
        ] = justification.id
        for version_id in (
            justification.support_version_ids
            + justification.explicit_defeater_version_ids
        ):
            self._justification_ids_by_reference[version_id].add(justification.id)

    def _unregister_justification(self, justification: Justification) -> None:
        for version_id in (
            justification.support_version_ids
            + justification.explicit_defeater_version_ids
        ):
            self._justification_ids_by_reference[version_id].remove(justification.id)

    def _refresh_justification_status(
        self,
        justification: Justification,
    ) -> None:
        supports_active = all(
            self._is_reference_active(version_id)
            for version_id in justification.support_version_ids
        )
        has_active_defeater = any(
            self._is_reference_active(version_id)
            for version_id in justification.explicit_defeater_version_ids
        )
        justification.status = (
            JustificationStatus.ACTIVE
            if supports_active and not has_active_defeater
            else JustificationStatus.INVALID
        )

    def _propagate_claim_changes(
        self,
        initial_claim_keys: list[str],
        report: MaintenanceReport,
    ) -> None:
        pending = set(initial_claim_keys)
        topological_ranks = self._topological_ranks()
        while pending:
            claim_key = min(
                pending,
                key=lambda key: (topological_ranks[key], key),
            )
            pending.remove(claim_key)

            justification = self._justifications[
                self._current_justification_id_by_claim[claim_key]
            ]
            old_version = self.get_current_claim_version(claim_key)
            new_status = (
                ClaimStatus.VALID
                if justification.status is JustificationStatus.ACTIVE
                else ClaimStatus.INVALID
            )
            if (
                old_version is not None
                and old_version.clause == justification.clause
                and old_version.status is new_status
            ):
                continue

            claim_version = ClaimVersion(
                claim_key=claim_key,
                kind=justification.conclusion_kind,
                clause=justification.clause,
                status=new_status,
                materialized_from_justification_id=justification.id,
                supersedes_version_id=(
                    old_version.id if old_version is not None else None
                ),
            )
            self._claim_versions[claim_version.id] = claim_version
            self._claim_version_ids_by_key[claim_key].append(claim_version.id)
            self._current_claim_version_id[claim_key] = claim_version.id
            report.changed_claim_keys.append(claim_key)
            report.created_claim_version_ids.append(claim_version.id)

            if old_version is None:
                continue
            for justification_id in sorted(
                self._justification_ids_by_reference.get(old_version.id, set())
            ):
                downstream = self._justifications[justification_id]
                self._refresh_justification_status(downstream)
                report.touched_justification_ids.append(justification_id)
                pending.add(downstream.conclusion_key)

    def _is_reference_active(self, version_id: str) -> bool:
        claim_version = self._claim_versions.get(version_id)
        if claim_version is not None:
            return (
                self._current_claim_version_id.get(claim_version.claim_key)
                == version_id
                and claim_version.status is ClaimStatus.VALID
            )
        fact = self.fact_store.get_fact(version_id)
        return fact.status is FactStatus.ACTIVE

    def _claim_key_for_version(self, version_id: str) -> str | None:
        claim_version = self._claim_versions.get(version_id)
        return claim_version.claim_key if claim_version is not None else None

    def _claim_adjacency(
        self,
        excluding_claim_key: str | None = None,
    ) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for claim_key, justification_id in self._current_justification_id_by_claim.items():
            if claim_key == excluding_claim_key:
                continue
            justification = self._justifications[justification_id]
            adjacency.setdefault(claim_key, set())
            for version_id in justification.support_version_ids:
                support_key = self._claim_key_for_version(version_id)
                if support_key is not None:
                    adjacency[support_key].add(claim_key)
        return adjacency

    def _has_path(
        self,
        source_key: str,
        target_key: str,
        excluding_claim_key: str | None = None,
    ) -> bool:
        adjacency = self._claim_adjacency(excluding_claim_key)
        stack = [source_key]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target_key:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, set()))
        return False

    def _topological_ranks(self) -> dict[str, int]:
        adjacency = self._claim_adjacency()
        indegree = {claim_key: 0 for claim_key in adjacency}
        for downstream_keys in adjacency.values():
            for downstream_key in downstream_keys:
                indegree[downstream_key] += 1

        queue = deque(
            sorted(
                claim_key
                for claim_key, degree in indegree.items()
                if degree == 0
            )
        )
        ranks = {claim_key: 0 for claim_key in queue}
        while queue:
            claim_key = queue.popleft()
            for downstream_key in sorted(adjacency[claim_key]):
                ranks[downstream_key] = max(
                    ranks.get(downstream_key, 0),
                    ranks[claim_key] + 1,
                )
                indegree[downstream_key] -= 1
                if indegree[downstream_key] == 0:
                    queue.append(downstream_key)
        return ranks

    def _active_claim_level(
        self,
        claim_key: str,
        memo: dict[str, int],
    ) -> int:
        if claim_key in memo:
            return memo[claim_key]
        justification = self._justifications[
            self._current_justification_id_by_claim[claim_key]
        ]
        level = 1 + max(
            self._reference_level(version_id, memo)
            for version_id in justification.support_version_ids
        )
        memo[claim_key] = level
        return level

    def _reference_level(
        self,
        version_id: str,
        memo: dict[str, int],
    ) -> int:
        claim_key = self._claim_key_for_version(version_id)
        if claim_key is None:
            return 0
        return self._active_claim_level(claim_key, memo)
