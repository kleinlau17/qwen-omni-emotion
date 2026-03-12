"""响应解析 — 将模型文本输出解析为结构化情绪/氛围数据"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.prompts.output_schema import AtmosphereResult, EmotionResult

LOGGER = logging.getLogger(__name__)


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
    """解析单人情绪 JSON 响应。"""
    payload = _try_parse_json(raw_text)
    if payload is None:
        LOGGER.warning("Failed to parse emotion JSON. Raw output: %s", raw_text)
        return None

    try:
        # 精简版 schema 只要求 primary_emotion / secondary_emotion，其余字段在这里补齐默认值，
        # 以兼容下游使用完整 EmotionResult 的代码。
        person_id = str(payload.get("person_id", "person_0"))
        emotion_intensity_raw = payload.get("emotion_intensity", 0.5)
        confidence_raw = payload.get("confidence", 0.5)
        description_raw = payload.get("description", "no description")
        return EmotionResult(
            person_id=person_id,
            primary_emotion=str(payload["primary_emotion"]),
            emotion_intensity=float(emotion_intensity_raw),
            secondary_emotion=(
                str(payload["secondary_emotion"])
                if payload.get("secondary_emotion") is not None
                else None
            ),
            confidence=float(confidence_raw),
            description=str(description_raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Invalid emotion payload: %s, raw: %s", exc, raw_text)
        return None


def parse_atmosphere_response(raw_text: str) -> AtmosphereResult | None:
    """解析多人氛围 JSON 响应。"""
    payload = _try_parse_json(raw_text)
    if payload is None:
        LOGGER.warning("Failed to parse atmosphere JSON. Raw output: %s", raw_text)
        return None

    try:
        individuals_raw = payload["individual_emotions"]
        if not isinstance(individuals_raw, list):
            raise ValueError("individual_emotions must be a list.")

        individuals: list[EmotionResult] = []
        for item in individuals_raw:
            individuals.append(
                EmotionResult(
                    # 精简版 schema 只强制 primary_emotion / secondary_emotion，
                    # 其余字段在此补默认值，兼容旧数据。
                    person_id=str(item.get("person_id", "person_0")),
                    primary_emotion=str(item["primary_emotion"]),
                    emotion_intensity=float(item.get("emotion_intensity", 0.5)),
                    secondary_emotion=(
                        str(item["secondary_emotion"])
                        if item.get("secondary_emotion") is not None
                        else None
                    ),
                    confidence=float(item.get("confidence", 0.5)),
                    description=str(item.get("description", "no description")),
                )
            )

        return AtmosphereResult(
            overall_mood=str(payload["overall_mood"]),
            tension_level=float(payload.get("tension_level", 0.5)),
            engagement_level=float(payload.get("engagement_level", 0.5)),
            individual_emotions=individuals,
            description=str(payload.get("description", "no description")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Invalid atmosphere payload: %s, raw: %s", exc, raw_text)
        return None


__all__ = ["parse_atmosphere_response", "parse_emotion_response"]
