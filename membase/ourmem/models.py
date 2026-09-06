"""统一保存原文、记忆内容与支持关系；事实和结论不再使用两套生命周期。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimePoint(Record):
    """真实日期或输入顺序位置；before 表示变化发生之前的边界。"""

    date: str | None = None
    order: int | None = Field(default=None, ge=0)
    offset: int = Field(default=0, ge=0)
    side: Literal["before", "at", "after"] = "at"

    @model_validator(mode="after")
    def validate_date(self) -> TimePoint:
        if self.date is not None:
            datetime.fromisoformat(self.date.replace("Z", "+00:00"))
        return self


class TimeScope(Record):
    """事件发生期不是知识的保质期；状态才会因其适用期结束而到期。"""

    kind: Literal["state", "event", "unknown"] = "unknown"
    start: TimePoint | None = None
    end: TimePoint | None = None
    precision: Literal["second", "minute", "day", "month", "year", "order", "unknown"] = "unknown"
    text: str | None = None


class SourceSpan(Record):
    """程序定位的原文半开区间 [start, end)，不是模型猜测的坐标。"""

    source_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("A source span must have end > start")
        return self


class ReferenceType(str, Enum):
    SOURCE = "SOURCE"
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"


class PremiseRef(Record):
    type: ReferenceType
    id: str
    span: SourceSpan | None = None
    context_refs: list[SourceSpan] = Field(default_factory=list)
    at_time: TimePoint | None = None

    @model_validator(mode="after")
    def reference_shape(self) -> PremiseRef:
        if self.type is ReferenceType.SOURCE:
            if self.span is None or self.span.source_id != self.id:
                raise ValueError("SOURCE requires a matching source span")
            if self.at_time is not None:
                raise ValueError("SOURCE does not use at_time")
        elif self.span is not None or self.context_refs:
            raise ValueError("Memory references do not carry source spans")
        if self.type is ReferenceType.HISTORICAL and (self.at_time is None or
                self.at_time.date is None and self.at_time.order is None):
            raise ValueError("HISTORICAL requires an explicit time or order boundary")
        if self.type is ReferenceType.CURRENT and self.at_time is not None:
            raise ValueError("CURRENT is evaluated at the caller's time")
        return self


class InputMessage(Record):
    """适配层白名单输入：刻意没有可混入标准答案的任意 metadata 字段。"""

    message_id: str
    content: str
    speaker: str = "user"
    role: Literal["user", "assistant", "system", "tool"] = "user"
    conversation_id: str | None = None
    mention_time: str | None = None
    source_order: int | None = Field(default=None, ge=0)
    generation_refs: list[PremiseRef] = Field(default_factory=list)


class Source(Record):
    id: str = Field(default_factory=lambda: new_id("src"))
    namespace: str
    message_id: str
    content: str
    speaker: str
    role: Literal["user", "assistant", "system", "tool"] = "user"
    conversation_id: str | None = None
    source_order: int = Field(ge=0)
    mention_time: str | None = None
    generation_refs: list[PremiseRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class InputPolicy(Record):
    """公开输入约定；不含任务类别、标准答案或标准依赖图。"""

    update_priority: Literal["explicit", "newer_source"] = "explicit"
    control_roles: list[str] = Field(default_factory=lambda: ["user"])
    description: str = ""


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class Revision(Record):
    previous_version_id: str
    reason: Literal["update", "correction", "reconfirmation", "override"]
    effective_time: TimePoint | None = None
    evidence_refs: list[PremiseRef] = Field(min_length=1)


class MemoryVersion(Record):
    id: str = Field(default_factory=lambda: new_id("mem"))
    namespace: str
    memory_key: str = Field(default_factory=lambda: new_id("key"))
    content: str = Field(min_length=1)
    valid_time: TimeScope = Field(default_factory=TimeScope)
    modality: Literal["asserted", "uncertain", "planned", "conditional", "hypothetical"] = "asserted"
    created_at: str = Field(default_factory=utc_now)
    revision: Revision | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE


class DependencyEffect(str, Enum):
    SUPPORT = "SUPPORT"
    INVALIDATE = "INVALIDATE"


class DependencyLink(Record):
    id: str = Field(default_factory=lambda: new_id("dep"))
    namespace: str
    premise_refs: list[PremiseRef] = Field(min_length=1)
    effect: DependencyEffect = DependencyEffect.SUPPORT
    target_version_id: str
    effective_time: TimePoint | None = None
    created_at: str = Field(default_factory=utc_now)


class ControlOperation(Record):
    """追加式控制动作；关闭、撤回和删除不改写旧内容。"""

    id: str = Field(default_factory=lambda: new_id("op"))
    namespace: str
    kind: Literal["close", "supersede", "correct", "retract", "delete", "conflict", "resolve_conflict", "pending", "resolve_pending", "revoke_control"]
    target_id: str
    scope: Literal["version", "key", "source", "span", "dependency", "operation"] = "version"
    span: SourceSpan | None = None
    effective_time: TimePoint | None = None
    evidence_refs: list[PremiseRef] = Field(default_factory=list)
    reason: str = ""
    topic: str = ""
    replacement_id: str | None = None
    source_cutoff: int | None = None
    created_at: str = Field(default_factory=utc_now)


class Snapshot(Record):
    id: int
    namespace: str
    source_cutoff: int
    sequence: int
    maintenance_incomplete: bool = False
    created_at: str = Field(default_factory=utc_now)


class Resolution(Record):
    version_id: str
    usable: bool
    status: MemoryStatus
    reason: str = ""
    support_ids: list[str] = Field(default_factory=list)


class EvidenceBundle(Record):
    """完整支持路径展开；超限时不将半条路径伪装成完整证据。"""

    text: str = ""
    refs: list[PremiseRef] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)
    complete: bool = True
    reason: str = ""


class PreparedContext(Record):
    context: str = ""
    resolution_status: Literal["resolved", "unknown", "conflict", "deleted", "incomplete"] = "incomplete"
    reason: str = ""
    evidence_refs: list[PremiseRef] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    read_trace: list[dict[str, Any]] = Field(default_factory=list)


class AnswerResult(PreparedContext):
    answer_text: str


class MaintenanceReport(Record):
    changed_ids: list[str] = Field(default_factory=list)
    pending_ids: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    generation_calls: int = 0
    incomplete: bool = False
