"""OurMem 当前正在迭代的核心数据模型（data model）。

这个包暂时只放我们自己方法的实现。对 MemBase 的适配层（adapter layer）
会在后续单独放入 ``membase.layers``，避免把研究方法和评测接口混在一起。
"""

from .extractor import FactExtractor
from .maintenance import ClaimMemory, MaintenanceReport
from .models import (
    AtomicFact,
    ClaimClause,
    ClaimKind,
    ClaimPolarity,
    ClaimStatus,
    ClaimValue,
    ClaimVersion,
    EvidenceQuote,
    FactStatus,
    FactUpdate,
    FactUpdateAction,
    Justification,
    JustificationStatus,
)
from .reconciler import FactReconciler
from .store import OurMemStore

__all__ = [
    "AtomicFact",
    "ClaimClause",
    "ClaimKind",
    "ClaimMemory",
    "ClaimPolarity",
    "ClaimStatus",
    "ClaimValue",
    "ClaimVersion",
    "EvidenceQuote",
    "FactExtractor",
    "FactReconciler",
    "FactStatus",
    "FactUpdate",
    "FactUpdateAction",
    "Justification",
    "JustificationStatus",
    "MaintenanceReport",
    "OurMemStore",
]
