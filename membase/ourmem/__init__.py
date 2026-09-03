"""OurMem 当前正在迭代的核心数据模型（data model）。

这个包只放我们自己方法的实现。对 MemBase 的适配层（adapter layer）位于
``membase.layers.ourmem``，避免把研究方法和评测接口混在一起。
"""

from .extractor import FactExtractor
from .inducer import JustificationInducer
from .maintenance import ClaimMemory, MaintenanceReport
from .models import (
    AtomicFact,
    ClaimClause,
    ClaimKind,
    ClaimPolarity,
    ClaimStatus,
    ClaimValue,
    ClaimVersion,
    FactStatus,
    FactUpdate,
    FactUpdateAction,
    Justification,
    JustificationStatus,
    SourceEvidence,
)
from .persistence import OurMemPersistence
from .reconciler import FactReconciler
from .reader import MemoryReader
from .retriever import (
    MemoryCandidateRetriever,
    OpenAIEmbedder,
    SemanticCandidate,
)
from .store import OurMemStore
from .system import OurMemSystem
from .writer import FactWriteResult, MemoryWriter

__all__ = [
    "AtomicFact",
    "ClaimClause",
    "ClaimKind",
    "ClaimMemory",
    "ClaimPolarity",
    "ClaimStatus",
    "ClaimValue",
    "ClaimVersion",
    "FactExtractor",
    "FactReconciler",
    "FactStatus",
    "FactUpdate",
    "FactUpdateAction",
    "Justification",
    "JustificationInducer",
    "JustificationStatus",
    "MaintenanceReport",
    "MemoryCandidateRetriever",
    "MemoryReader",
    "MemoryWriter",
    "FactWriteResult",
    "OpenAIEmbedder",
    "OurMemPersistence",
    "OurMemStore",
    "OurMemSystem",
    "SemanticCandidate",
    "SourceEvidence",
]
