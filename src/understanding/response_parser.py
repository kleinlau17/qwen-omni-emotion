"""响应解析 — 将模型文本输出解析为结构化动作数据"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.prompts.output_schema import EmotionResult, VALID_ACTIONS

LOGGER = logging.getLogger(__name__)


_FALLBACK_ACTION_BY_KEYWORD: tuple[tuple[tuple[str, ...], str], ...] = (
    (("idle", "none", "no_action", "wait", "hold", "standby"), "idle"),
    (("affirm", "agree", "yes", "nod"), "neutral.affirm.default.mid"),
    (("deny", "reject", "refuse", "no", "shake"), "neutral.deny.default.mid"),
    (("attention", "listen", "focus"), "neutral.attention.default.mid"),
    (("dialogue", "chat", "talk", "speak"), "neutral.dialogue.default.mid"),
    (("think", "ponder", "consider", "murmur"), "neutral.think.muted.low"),
    (("question", "ask", "curious", "inquire", "inquiring"), "neutral.question.default.mid"),
    (("surprise", "startle", "astonish"), "neutral.surprise.quick.low"),
    (("alarm", "panic", "scare"), "neutral.alarm.expressive.high"),
    (("apology", "sorry", "embarrass"), "neutral.apology.default.low"),
    (("recover", "encourage", "cheer"), "neutral.recover.default.mid"),
    (("sigh",), "sad.sigh.default.low"),
)


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    """将任意输入转浮点并夹紧到给定区间。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def _normalize_action(action: Any) -> str:
    """归一化模型动作名，提升解析鲁棒性。"""
    normalized = str(action).strip().strip(".,;:!?'\"，。；：！？")
    if normalized in VALID_ACTIONS:
        return normalized
    return _fallback_action(normalized)


def _fallback_action(action: str) -> str:
    """将库外动作按关键词回退到最接近的合法动作。"""
    lowered = action.lower()
    tokens = {token for token in re.split(r"[^a-z]+", lowered) if token}
    for keywords, fallback in _FALLBACK_ACTION_BY_KEYWORD:
        if any(keyword in tokens for keyword in keywords):
            return fallback
    return "idle"


def _normalize_reason(reason: Any) -> str:
    """归一化动作原因文本。"""
    normalized = str(reason).strip()
    if normalized:
        return normalized
    return "未提供动作原因。"


def _extract_json_candidates(raw_text: str) -> list[str]:
    """从原始文本中提取可能的 JSON 对象子串。"""
    stripped_text = raw_text.strip()
    candidates: list[str] = []

    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        candidates.append(stripped_text)

    # 先尝试大块，再尝试小块，提升容错命中率。
    greedy_matches = re.findall(r"\{[\s\S]*\}", raw_text)
    non_greedy_matches = re.findall(r"\{[\s\S]*?\}", raw_text)
    candidates.extend(greedy_matches)
    candidates.extend(non_greedy_matches)

    # 去重并按长度降序，优先解析信息最完整的候选。
    deduplicated = list(dict.fromkeys(candidates))
    deduplicated.sort(key=len, reverse=True)
    return deduplicated


def _try_parse_json(raw_text: str) -> dict[str, Any] | None:
    """尝试从原始文本解析 JSON 对象。"""
    for candidate in _extract_json_candidates(raw_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_emotion_response(raw_text: str) -> EmotionResult | None:
    """解析单人动作 JSON 响应。"""
    payload = _try_parse_json(raw_text)
    if payload is None:
        LOGGER.warning("Failed to parse emotion JSON. Raw output: %s", raw_text)
        return None

    try:
        action = _normalize_action(payload["action"])
        if action != str(payload["action"]).strip():
            LOGGER.debug(
                "Action normalized: raw=%s normalized=%s",
                payload["action"],
                action,
            )
        return EmotionResult(
            person_id="person_0",
            action=action,
            reason=_normalize_reason(payload.get("reason", payload.get("action_reason", ""))),
            detected_emotion=str(payload.get("detected_emotion", "neutral")),
            action_confidence=_clamp_float(
                payload.get("action_confidence", 0.5),
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
            hold_seconds=_clamp_float(
                payload.get("hold_seconds", 1.0),
                default=1.0,
                minimum=0.5,
                maximum=3.0,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Invalid emotion payload: %s, raw: %s", exc, raw_text)
        return None


__all__ = ["parse_emotion_response"]
