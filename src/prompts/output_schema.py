"""结构化输出格式定义 — 约束模型以 JSON Schema 格式返回情绪/氛围分析结果"""
from __future__ import annotations

import json
from dataclasses import dataclass

VALID_EMOTIONS: list[str] = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "neutral",
    "contemptuous",
]


def _validate_unit_interval(value: float, field_name: str) -> None:
    """校验浮点值是否落在 0.0-1.0 区间。"""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value}.")


@dataclass
class EmotionResult:
    """单个人物情绪分析结果（简化版，仅 primary_emotion + secondary_emotion）。

    person_id 由 pipeline 根据 person_idx 赋值，用于内部追踪，不由模型输出。
    """

    person_id: str
    primary_emotion: str
    secondary_emotion: str | None

    def __post_init__(self) -> None:
        """执行字段合法性校验。"""
        if self.primary_emotion not in VALID_EMOTIONS:
            raise ValueError(
                f"primary_emotion must be one of {VALID_EMOTIONS}, got {self.primary_emotion}."
            )
        if self.secondary_emotion is not None and self.secondary_emotion not in VALID_EMOTIONS:
            raise ValueError(
                f"secondary_emotion must be one of {VALID_EMOTIONS}, got {self.secondary_emotion}."
            )
        if not self.person_id.strip():
            raise ValueError("person_id must not be empty.")


@dataclass
class AtmosphereResult:
    """多人场景的群体氛围分析结果（简化版）。"""

    overall_mood: str
    tension_level: float
    engagement_level: float
    individual_emotions: list[EmotionResult]

    def __post_init__(self) -> None:
        """执行字段合法性校验。"""
        _validate_unit_interval(self.tension_level, "tension_level")
        _validate_unit_interval(self.engagement_level, "engagement_level")
        if not self.overall_mood.strip():
            raise ValueError("overall_mood must not be empty.")
        if not isinstance(self.individual_emotions, list):
            raise ValueError("individual_emotions must be a list of EmotionResult.")
        if not all(isinstance(item, EmotionResult) for item in self.individual_emotions):
            raise ValueError("individual_emotions must contain EmotionResult only.")


# 模型输出 schema（简化：仅 primary_emotion + secondary_emotion）
_EMOTION_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["primary_emotion", "secondary_emotion"],
    "properties": {
        "primary_emotion": {"type": "string", "enum": VALID_EMOTIONS},
        "secondary_emotion": {
            "anyOf": [
                {"type": "string", "enum": VALID_EMOTIONS},
                {"type": "null"},
            ]
        },
    },
}

SINGLE_PERSON_SCHEMA: str = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": list(_EMOTION_RESULT_SCHEMA["required"]),
        "properties": dict(_EMOTION_RESULT_SCHEMA["properties"]),
    },
    indent=2,
)

MULTI_PERSON_SCHEMA: str = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "overall_mood",
            "tension_level",
            "engagement_level",
            "individual_emotions",
        ],
        "properties": {
            "overall_mood": {"type": "string", "minLength": 1},
            "tension_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "engagement_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "individual_emotions": {
                "type": "array",
                "items": _EMOTION_RESULT_SCHEMA,
            },
        },
    },
    indent=2,
)

__all__ = [
    "AtmosphereResult",
    "EmotionResult",
    "MULTI_PERSON_SCHEMA",
    "SINGLE_PERSON_SCHEMA",
    "VALID_EMOTIONS",
]
