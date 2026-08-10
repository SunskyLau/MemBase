import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Self

from .base import MemBaseDataset
from ..model_types.dataset import (
    Message,
    QuestionAnswerPair,
    Session,
    Trajectory,
)


class MobileMemOmni(MemBaseDataset):
    """Dataset wrapper for LoCoMo-shaped MobileMem-Omni data."""

    CATEGORY_ID_TO_TYPE: ClassVar[dict[int, str]] = {
        1: "multi-hop",
        2: "temporal-reasoning",
        3: "abstention",
        4: "single-hop",
        5: "implicit-preference",
        6: "visual-reasoning",
        7: "knowledge-update",
    }

    @staticmethod
    def _parse_session_timestamp(value: str, sample_id: str, session_idx: int) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError(
                f"Sample '{sample_id}' session {session_idx} has no timestamp."
            )

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass

        for fmt in (
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %b, %Y",
        ):
            try:
                return datetime.strptime(value, fmt).isoformat()
            except ValueError:
                continue

        raise ValueError(
            f"Sample '{sample_id}' session {session_idx} has an unsupported "
            f"timestamp: {value!r}."
        )

    @staticmethod
    def _normalise_answer(answer: Any) -> list[str]:
        if isinstance(answer, list):
            values = [str(item) for item in answer if item is not None]
            return values or [""]
        if answer is None:
            return [""]
        return [str(answer)]

    @classmethod
    def read_raw_data(cls, path: str) -> Self:
        raw_path = Path(path)
        with raw_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data:
            raise ValueError(
                "MobileMem-Omni data must be a non-empty JSON object or array."
            )

        trajectories = []
        qa_pair_lists = []
        seen_sample_ids = set()

        for sample_idx, sample in enumerate(data, start=1):
            sample_id = str(sample.get("sample_id") or f"uuid_{sample_idx}")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate MobileMem-Omni sample_id: {sample_id!r}.")
            seen_sample_ids.add(sample_id)

            conversation = sample.get("conversation")
            if not isinstance(conversation, dict):
                raise ValueError(f"Sample '{sample_id}' has no conversation object.")

            speaker_a = str(conversation.get("speaker_a") or "user")
            speaker_b = str(conversation.get("speaker_b") or "assistant")
            session_summaries = sample.get("session_summary") or {}
            session_observations = sample.get("observation") or {}
            event_summaries = sample.get("event_summary") or {}

            session_indices = sorted(
                int(key.removeprefix("session_"))
                for key, value in conversation.items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
                and key.removeprefix("session_").isdigit()
                and isinstance(value, list)
            )
            if not session_indices:
                raise ValueError(f"Sample '{sample_id}' contains no sessions.")

            sessions = []
            for session_idx in session_indices:
                session_key = f"session_{session_idx}"
                raw_messages = conversation[session_key]
                if not raw_messages:
                    continue
                timestamp = cls._parse_session_timestamp(
                    conversation.get(f"{session_key}_date_time", ""),
                    sample_id,
                    session_idx,
                )

                messages = []
                for message_idx, raw_message in enumerate(raw_messages, start=1):
                    speaker = str(raw_message.get("speaker") or "")
                    if speaker == speaker_a:
                        role = "user"
                    elif speaker == speaker_b:
                        role = "assistant"
                    else:
                        role = "user"

                    metadata = {
                        key: value
                        for key, value in raw_message.items()
                        if key not in {"speaker", "text", "dia_id"}
                    }
                    metadata["speaker_tag"] = (
                        "speaker_a" if speaker == speaker_a else "speaker_b"
                    )
                    messages.append(
                        Message(
                            id=str(
                                raw_message.get("dia_id")
                                or f"{sample_id}:S{session_idx}:M{message_idx}"
                            ),
                            name=speaker or (speaker_a if role == "user" else speaker_b),
                            role=role,
                            content=str(raw_message.get("text") or ""),
                            timestamp=timestamp,
                            metadata=metadata,
                        )
                    )

                sessions.append(
                    Session(
                        id=f"{sample_id}-session-{session_idx}",
                        messages=messages,
                        metadata={
                            "session_summary": session_summaries.get(
                                f"{session_key}_summary", ""
                            ),
                            "session_observation": session_observations.get(
                                f"{session_key}_observation", {}
                            ),
                            "event_summary": event_summaries.get(
                                f"{session_key}_event_summary", {}
                            ),
                        },
                    )
                )

            if not sessions:
                raise ValueError(f"Sample '{sample_id}' contains no non-empty sessions.")

            trajectories.append(
                Trajectory(
                    id=f"mobilemem-omni-{sample_id}",
                    sessions=sorted(sessions),
                    metadata={
                        "sample_id": sample_id,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                    },
                )
            )

            default_question_timestamp = sessions[-1].ended_at
            qa_pairs = []
            for question_idx, raw_qa in enumerate(sample.get("qa") or [], start=1):
                category_id = raw_qa.get("category")
                if category_id not in cls.CATEGORY_ID_TO_TYPE:
                    raise ValueError(
                        f"Sample '{sample_id}' question {question_idx} has unsupported "
                        f"category {category_id!r}; expected an integer from 1 to 7."
                    )

                question_timestamp = raw_qa.get("timestamp") or default_question_timestamp
                try:
                    question_timestamp = datetime.fromisoformat(
                        str(question_timestamp).replace("Z", "+00:00")
                    ).isoformat()
                except ValueError as exc:
                    raise ValueError(
                        f"Sample '{sample_id}' question {question_idx} has an invalid "
                        f"timestamp: {question_timestamp!r}."
                    ) from exc

                evidence = raw_qa.get("evidence") or []
                if not isinstance(evidence, list):
                    evidence = [evidence]

                qa_pairs.append(
                    QuestionAnswerPair(
                        id=str(
                            raw_qa.get("question_id")
                            or raw_qa.get("qid")
                            or f"{sample_id}-qa-{question_idx}"
                        ),
                        question=str(raw_qa.get("question") or ""),
                        golden_answers=cls._normalise_answer(raw_qa.get("answer")),
                        timestamp=question_timestamp,
                        metadata={
                            "question_type": cls.CATEGORY_ID_TO_TYPE[category_id],
                            "category_id": category_id,
                            "question_format": raw_qa.get("question_format"),
                            "evidence": [str(item) for item in evidence],
                            "sample_id": sample_id,
                        },
                    )
                )

            qa_pair_lists.append(qa_pairs)

        return cls(trajectories=trajectories, qa_pair_lists=qa_pair_lists)

    def _generate_metadata(self) -> dict[str, Any]:
        metadata = {
            "name": "MobileMem-Omni",
            "codebase_url": "https://github.com/zjunlp/MobileMem",
            "size": len(self),
            "total_sessions": 0,
            "total_messages": 0,
            "total_questions": 0,
            "question_type_stats": {},
        }
        for trajectory, qa_pairs in self:
            metadata["total_sessions"] += len(trajectory)
            metadata["total_messages"] += sum(len(session) for session in trajectory)
            metadata["total_questions"] += len(qa_pairs)
            for qa_pair in qa_pairs:
                question_type = qa_pair.metadata["question_type"]
                stats = metadata["question_type_stats"]
                stats[question_type] = stats.get(question_type, 0) + 1
        return metadata

    @classmethod
    def get_judge_template_name(cls, qa_pair: QuestionAnswerPair) -> str:
        return "mobilemem-omni-judge"

    @classmethod
    def parse_judge_response(cls, content: str) -> float:
        content = content.strip()
        try:
            parsed = json.loads(content)
            label = str(parsed.get("label", ""))
        except (json.JSONDecodeError, AttributeError):
            label = content
        normalised = label.strip().upper()
        if "WRONG" in normalised:
            return 0.0
        return float("CORRECT" in normalised)
