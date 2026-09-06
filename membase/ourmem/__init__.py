"""OurMem 公共记录；不在基础导入时建立模型客户端。"""

from .models import (
    AnswerResult, ControlOperation, DependencyEffect, DependencyLink, EvidenceBundle,
    InputMessage, InputPolicy, MemoryStatus, MemoryVersion, PremiseRef, PreparedContext,
    ReferenceType, Resolution, Revision, Snapshot, Source, SourceSpan, TimePoint, TimeScope,
)


def __getattr__(name: str):
    if name == "OurMemSystem":
        from .system import OurMemSystem
        return OurMemSystem
    raise AttributeError(name)
