"""统一保存和恢复 OurMem 的事实、依据与派生主张。"""

from __future__ import annotations

import json
from pathlib import Path

from .maintenance import ClaimMemory
from .store import OurMemStore


class OurMemPersistence:
    """OurMem JSON 快照（JSON snapshot）的读写接口。"""

    @staticmethod
    def save(
        path: str | Path,
        fact_store: OurMemStore,
        claim_memory: ClaimMemory,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fact_store": fact_store.to_dict(),
            "claim_memory": claim_memory.to_dict(),
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> tuple[OurMemStore, ClaimMemory]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fact_store = OurMemStore.from_dict(data["fact_store"])
        claim_memory = ClaimMemory.from_dict(
            fact_store,
            data["claim_memory"],
        )
        return fact_store, claim_memory
