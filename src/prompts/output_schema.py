"""结构化输出格式定义 — 约束模型以 JSON Schema 返回动作决策结果"""
from __future__ import annotations

import json
from dataclasses import dataclass

VALID_EMOTIONS: list[str] = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "neutral",
]

VALID_ACTIONS: list[str] = [
    "idle",
#    "neutral.scan.default.mid",
#    "neutral.scan.inquiring.mid",
    "neutral.surprise.quick.low",
    "neutral.deny.default.mid",
    "shy.deny.inquiring.mid",
    "shy.affirm.inquiring.mid",
    "neutral.affirm.default.mid",
#    "neutral.scan.odor.mid",
#    "neutral.defend.defensive.mid",
#    "neutral.scan.sweep.low",
#    "neutral.scan.micro_scan.mid",
#    "neutral.scan.inquiring.mid",
    "neutral.recover.default.mid",
    "neutral.affirm.expressive.high",
    "neutral.attention.default.mid",
#    "neutral.bored.default.low",
    "neutral.deny.quick.low",
    "neutral.alarm.expressive.high",
    "shy.affirm.inquiring.low",
    "neutral.dialogue.default.mid",
    "neutral.think.murmur.low",
    "neutral.think.muted.low",
    "neutral.think.animated.mid",
    "neutral.question.default.low",
    "neutral.question.default.mid",
    "neutral.question.default.high",
    "neutral.affirm.playful.low",
    "neutral.affirm.default.high",
    "neutral.apology.default.low",
    "angry.deny.default.high",
#    "angry.vent.expressive.high",
    "angry.affirm.default.high",
    "angry.greet.default.mid",
    "angry.deny.quick.mid",
    "sad.sigh.default.low",
    "sad.deny.default.low",
    "sad.affirm.default.low",
]

ACTION_LIBRARY: list[tuple[str, str]] = [
    ("idle", "保持不动"),
    ("neutral.affirm.playful.low", "俏皮点头"),
    ("neutral.affirm.default.high", "用力点头"),
    ("neutral.deny.quick.low", "快速摇头"),
    ("neutral.deny.default.mid", "正常摇头"),
    ("neutral.attention.default.mid", "认真注视"),
    ("neutral.recover.default.mid", "鼓励"),
    ("neutral.dialogue.default.mid", "对话回应"),
    ("neutral.think.animated.mid", "活跃地思索"),
    ("neutral.question.default.mid", "正常提问"),
    ("neutral.surprise.quick.low", "轻微惊讶"),
]


def format_action_library() -> str:
    """将动作列表格式化为 prompt 可嵌入的紧凑文本。"""
    return " / ".join(f"{name}({desc})" for name, desc in ACTION_LIBRARY)




def _validate_unit_interval(value: float, field_name: str) -> None:
    """校验浮点值是否落在 0.0-1.0 区间。"""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value}.")


@dataclass
class EmotionResult:
    """单个人物情绪与互动结果。

    person_id 由 pipeline 根据 person_idx 赋值，用于内部追踪，不由模型输出。
    """

    person_id: str
    action: str
    reason: str = ""
    detected_emotion: str = "neutral"
    action_confidence: float = 0.5
    hold_seconds: float = 1.0

    def __post_init__(self) -> None:
        """执行字段合法性校验。"""
        if self.detected_emotion not in VALID_EMOTIONS:
            raise ValueError(
                f"detected_emotion must be one of {VALID_EMOTIONS}, got {self.detected_emotion}."
            )
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {VALID_ACTIONS}, got {self.action}."
            )
        _validate_unit_interval(self.action_confidence, "action_confidence")
        if not 0.5 <= self.hold_seconds <= 3.0:
            raise ValueError(
                f"hold_seconds must be in [0.5, 3.0], got {self.hold_seconds}."
            )
        if not self.person_id.strip():
            raise ValueError("person_id must not be empty.")
        self.reason = str(self.reason).strip()


_EMOTION_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reason"],
    "properties": {
        "action": {"type": "string", "enum": VALID_ACTIONS},
        "reason": {"type": "string", "minLength": 1, "maxLength": 120},
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

__all__ = [
    "ACTION_LIBRARY",
    "EmotionResult",
    "SINGLE_PERSON_SCHEMA",
    "VALID_ACTIONS",
    "VALID_EMOTIONS",
    "format_action_library",
]
