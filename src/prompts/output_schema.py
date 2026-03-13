"""结构化输出格式定义 — 约束模型以 JSON Schema 格式返回情绪/互动分析结果"""
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

VALID_ACTIONS: list[str] = [
    "scan_01",
    "scan_curious",
    "sneeze",
    "standard_no",
    "shy_no",
    "shy_yes",
    "standard_yes",
    "scan_sniff_stinky",
    "turtle_pose",
    "scan_raster",
    "scan_edge",
    "dance_twist",
    "curious",
    "perk_up",
    "hell_yes",
    "attention",
    "bored_snore",
    "jump_scare",
    "laugh_big",
    "laugh_giggle",
    "neutral_no",
    "scream_scare",
    "shy_trill",
    "dialogue_1",
    "babble_1",
    "babble_2",
    "babble_3",
    "question_1",
    "question_2",
    "question_3",
    "yes_1",
    "yes_2",
    "uhoh",
    "angry_no",
    "tantrum",
    "rant",
    "angry_yes",
    "angry_greeting",
    "angry_no_short",
    "sad_sigh",
    "sad_no",
    "sad_yes",
]

ACTION_GUIDE: list[str] = [
    "scan_01: 白眼+手电筒扫描四角（环境感知）",
    "scan_curious: 左右歪头后再扫描（好奇）",
    "sneeze: 快速向前下打喷嚏（意外生理反应）",
    "standard_no: 摇头+失落音效（拒绝）",
    "shy_no: 缩脖摇头+失落音效（害羞拒绝）",
    "shy_yes: 低头三次点头（害羞肯定）",
    "standard_yes: 平视三次点头（肯定）",
    "scan_sniff_stinky: 扫描闻到臭味后快速摇头（厌恶）",
    "turtle_pose: 缩脖蜷缩闭眼（防御/害怕）",
    "scan_raster: 快速左右摇头扫描（快速探测）",
    "scan_edge: 小角度四向扫描（细致观察）",
    "dance_twist: 头身扭动舞蹈（娱乐）",
    "curious: 缩脖歪头+伸脖再歪头+快速扫描（好奇探索）",
    "perk_up: 先低头再抬头伸懒腰（恢复活力）",
    "hell_yes: 抬头兴奋叫+不停点头（兴奋肯定）",
    "attention: 伸脖立正+天线竖起（专注）",
    "bored_snore: 低头丧气蜷缩（无聊/委屈）",
    "jump_scare: 受惊小跳（惊吓）",
    "laugh_big: 蹲前摇后大幅抬头笑（大笑）",
    "laugh_giggle: 蹲着点头笑（轻快笑）",
    "neutral_no: 快速摇头一次（中性拒绝）",
    "scream_scare: 仰头尖叫（强烈惊吓）",
    "shy_trill: 天线收起点头眨眼（害羞肯定）",
    "dialogue_1: 耳朵前摇准备对话（对话开始）",
    "babble_1: 耳朵小摆+眼闪（自言自语）",
    "babble_2: 缩脖微动+眼闪（轻度自言自语）",
    "babble_3: 耳朵前后摆+眼熄灯（自言自语）",
    "question_1: 疑问音效+耳朵摆动（轻度疑惑）",
    "question_2: 低头再抬头+疑问音调（疑惑）",
    "question_3: 下沉闪眼再抬头+高音调（强疑惑）",
    "yes_1: 点头眨眼+轻快音效（肯定）",
    "yes_2: 点头眨眼+音色更宽（肯定）",
    "uhoh: 缩脖左右摆头+叹息（做错事/道歉）",
    "angry_no: 愤怒拒绝（生气）",
    "tantrum: 低头蓄力后仰天咆哮（爆发）",
    "rant: 左右摇头咆哮（愤怒表达）",
    "angry_yes: 红眼快速点头（愤怒肯定）",
    "angry_greeting: 低头抬头+机关枪音效（愤怒回应）",
    "angry_no_short: 红眼快速摇头（愤怒拒绝）",
    "sad_sigh: 低头前倾叹息（伤心）",
    "sad_no: 低头前倾摇头（伤心拒绝）",
    "sad_yes: 低头前倾天线摆动（伤心肯定）",
]


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
    detected_emotion: str
    self_emotion: str
    action: str

    def __post_init__(self) -> None:
        """执行字段合法性校验。"""
        if self.detected_emotion not in VALID_EMOTIONS:
            raise ValueError(
                f"detected_emotion must be one of {VALID_EMOTIONS}, got {self.detected_emotion}."
            )
        if self.self_emotion not in VALID_EMOTIONS:
            raise ValueError(
                f"self_emotion must be one of {VALID_EMOTIONS}, got {self.self_emotion}."
            )
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {VALID_ACTIONS}, got {self.action}."
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


# 模型输出 schema（仅情绪识别 + 自身情绪 + 动作）
_EMOTION_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["detected_emotion", "self_emotion", "action"],
    "properties": {
        "detected_emotion": {"type": "string", "enum": VALID_EMOTIONS},
        "self_emotion": {"type": "string", "enum": VALID_EMOTIONS},
        "action": {"type": "string", "enum": VALID_ACTIONS},
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
            "individual_emotions": {"type": "array", "items": _EMOTION_RESULT_SCHEMA},
        },
    },
    indent=2,
)

__all__ = [
    "AtmosphereResult",
    "EmotionResult",
    "MULTI_PERSON_SCHEMA",
    "SINGLE_PERSON_SCHEMA",
    "ACTION_GUIDE",
    "VALID_ACTIONS",
    "VALID_EMOTIONS",
]
