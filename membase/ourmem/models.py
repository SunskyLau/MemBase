"""OurMem 最基础的记忆数据结构（data structure）。

当前文件描述 OurMem 已经实现的基础对象：

1. 证据原文（evidence quote）：从原始对话中摘录、直接支持某条事实的原文；
2. 原子事实（atomic fact）：从对话中抽取、能够独立变化的最小事实单元。

可以把它们想象成读书时的“荧光笔标记”和“事实卡片”：

- 证据原文（evidence quote）是书页上被标亮的原文；
- 原子事实（atomic fact）是根据原文整理出的卡片；
- 卡片内容被纠正时，不擦掉旧卡片，而是增加新卡片并保留版本关系。

派生主张（derived claim）不伪装成用户原话，而是通过成立依据
（justification）递归指向原子事实（atomic fact）及其证据原文
（evidence quote）。
"""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _new_id(prefix: str) -> str:
    """生成便于人眼识别的随机标识。"""

    return f"{prefix}-{uuid4()}"


class FactStatus(str, Enum):
    """原子事实（atomic fact）在当前记忆中的状态。

    这里仅描述生命周期状态（lifecycle status），不描述事实内容是否确定。
    例如“会议可能改到周六，但尚未确认”本身可以是一条 ``ACTIVE`` 的
    原子事实（atomic fact），因为不确定语义（uncertain semantics）已经包含
    在 ``content`` 中。

    ``ACTIVE``
        当前仍可作为有效事实使用。

    ``SUPERSEDED``
        已经被更新版本替代。旧事实仍然保留，用于历史查询和证据追溯。

    ``RETRACTED``
        信息提供者明确撤回了这条事实。撤回不等于删除，原始记录仍然存在。

    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class OurMemModel(BaseModel):
    """OurMem 数据模型（data model）的共同基础配置。

    禁止未声明字段，避免把拼错的字段静默写入持久化数据。
    """

    model_config = ConfigDict(extra="forbid")


class EvidenceQuote(OurMemModel):
    """从原始消息中摘录的直接证据。

    证据原文（evidence quote）不是系统总结，也不是系统推断。它表示：
    “原始对话中的这一段文字，是某条原子事实（atomic fact）的直接来源。”

    一条消息可以产生多条证据原文（evidence quote），例如：

    ``父亲不能走太远，母亲不吃辣。``

    可以分别截取 ``父亲不能走太远`` 和 ``母亲不吃辣``，从而支持两条能够
    独立更新的原子事实（atomic fact）。
    """

    id: str = Field(
        default_factory=lambda: _new_id("quote"),
        min_length=1,
        description="证据原文（evidence quote）的唯一标识。",
    )
    message_id: str = Field(
        min_length=1,
        description="该片段来自哪一条 MemBase 消息。",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "该消息所在的会话标识。MemBase 当前没有直接把它传给记忆层（memory layer），"
            "因此先允许为空。"
        ),
    )
    message_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "该消息（message）在所属会话（session）中的顺序位置，从 0 开始。它不表示"
            "由用户消息和助手消息组成的交互轮（exchange）；暂时无法获得时可以为空。"
        ),
    )
    speaker: str = Field(
        min_length=1,
        description="原始文本的说话者。",
    )
    quote: str = Field(
        min_length=1,
        description="真正支持事实的原始文本，而不是整段对话摘要。",
    )
    timestamp: str = Field(
        min_length=1,
        description="这段原始文本在对话中被提及的时间。",
    )


class AtomicFact(OurMemModel):
    """能够被独立更新的最小事实单元。

    原子事实（atomic fact）像一张带出处的事实卡片。例如：

    ``Hotel A 距离地铁站约 200 米``

    当用户后来纠正为 2 公里时，不直接改写这张旧卡片。系统创建一张新卡片，
    让新原子事实（atomic fact）的 ``supersedes_fact_id`` 指向旧原子事实
    （atomic fact），同时把旧原子事实（atomic fact）的状态改为
    ``SUPERSEDED``。这样当前状态和历史状态都不会丢失。
    """

    id: str = Field(
        default_factory=lambda: _new_id("fact"),
        min_length=1,
        description="原子事实（atomic fact）的唯一标识。",
    )
    content: str = Field(
        min_length=1,
        description="事实本身的自然语言表达，一条记录尽量只表达一个可独立变化的事实。",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="事实涉及的人物、地点、物品或其他实体，用于后续定位相关记忆。",
    )
    evidence_quote_id: str = Field(
        min_length=1,
        description="直接支持这条事实的证据原文（evidence quote）标识。",
    )
    mention_time: str = Field(
        min_length=1,
        description="事实在对话中被提及的时间，通常来自 MemBase 消息时间戳。",
    )
    event_time: str | None = Field(
        default=None,
        description=(
            "事实真实发生的时间。例如，用户今天提到去年的旅行，提及时间（mention time）"
            "是今天，事件时间（event time）是去年。无法可靠判断时保持为空。"
        ),
    )
    status: FactStatus = Field(
        default=FactStatus.ACTIVE,
        description="原子事实（atomic fact）当前有效、已被替代或已被撤回。",
    )
    supersedes_fact_id: str | None = Field(
        default=None,
        description=(
            "如果这是一条修正后的新事实，这里指向被它替代的旧原子事实"
            "（atomic fact）。普通新增事实保持为空。"
        ),
    )
    retracted_by_evidence_quote_id: str | None = Field(
        default=None,
        description="明确撤回这条事实的证据原文（evidence quote）标识。",
    )

    @field_validator("entities")
    @classmethod
    def _deduplicate_list_values(cls, values: list[str]) -> list[str]:
        """去除重复值，同时保留第一次出现的顺序。"""

        if any(not value.strip() for value in values):
            raise ValueError("列表中不能包含空字符串")
        return list(dict.fromkeys(values))


class FactUpdateAction(str, Enum):
    """新原子事实（atomic fact）相对当前事实产生的版本操作。"""

    ADD = "add"
    DUPLICATE = "duplicate"
    SUPERSEDE = "supersede"
    RETRACT = "retract"


class FactUpdate(OurMemModel):
    """事实协调器（fact reconciler）输出的确定性操作。"""

    action: FactUpdateAction
    target_fact_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_target(self) -> FactUpdate:
        if self.action is FactUpdateAction.ADD:
            if self.target_fact_id is not None:
                raise ValueError("ADD cannot have a target fact")
        elif self.target_fact_id is None:
            raise ValueError(f"{self.action.value.upper()} requires a target fact")
        return self


class ClaimKind(str, Enum):
    """首版派生主张（derived claim）的两种层次。"""

    STATE = "state"
    DECISION = "decision"


class ClaimPolarity(str, Enum):
    """主张值是肯定还是否定。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ClaimStatus(str, Enum):
    """派生主张（derived claim）当前是否仍被其唯一依据支持。"""

    VALID = "valid"
    INVALID = "invalid"


class JustificationStatus(str, Enum):
    """成立依据（justification）的生命周期状态。"""

    ACTIVE = "active"
    INVALID = "invalid"
    SUPERSEDED = "superseded"


class ClaimValue(OurMemModel):
    """不包含时间范围的规范主张值。"""

    proposition: str = Field(
        min_length=1,
        description="主张在当前身份下表达的规范值。",
    )
    polarity: ClaimPolarity = Field(
        default=ClaimPolarity.POSITIVE,
        description="主张值的肯定或否定极性。",
    )


class ClaimClause(OurMemModel):
    """一条成立依据（justification）支持的规范结论。"""

    value: ClaimValue


class ClaimVersion(OurMemModel):
    """派生主张（derived claim）对外语义的一次不可变版本。"""

    id: str = Field(
        default_factory=lambda: _new_id("claim-version"),
        min_length=1,
    )
    claim_key: str = Field(
        min_length=1,
        description="派生主张（derived claim）的稳定语义身份。",
    )
    kind: ClaimKind
    clause: ClaimClause
    status: ClaimStatus
    materialized_from_justification_id: str = Field(min_length=1)
    supersedes_version_id: str | None = None


class Justification(OurMemModel):
    """一组直接支持如何共同推出一条派生主张（derived claim）。

    ``support_version_ids`` 可以引用原子事实（atomic fact）版本，也可以引用
    当前派生主张（derived claim）版本。派生来源最终沿这些引用展开到原始证据，
    因此这里不重复保存伪造的原文片段。
    """

    id: str = Field(
        default_factory=lambda: _new_id("justification"),
        min_length=1,
    )
    conclusion_key: str = Field(min_length=1)
    conclusion_kind: ClaimKind
    support_version_ids: list[str] = Field(min_length=2, max_length=3)
    explicit_defeater_version_ids: list[str] = Field(default_factory=list)
    clause: ClaimClause
    status: JustificationStatus = JustificationStatus.ACTIVE
    supersedes_justification_id: str | None = None

    @field_validator(
        "support_version_ids",
        "explicit_defeater_version_ids",
    )
    @classmethod
    def _deduplicate_version_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Version references cannot contain duplicates")
        return values
