"""分析任务指令模板 — 单人情绪 / 多人氛围 / 场景切换的 prompt 构建"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.prompts.output_schema import MULTI_PERSON_SCHEMA, SINGLE_PERSON_SCHEMA


def build_single_person_prompt() -> str:
    """构建单人情绪分析任务指令。"""
    return (
        "请对当前单人场景进行情绪分析，并返回严格 JSON。"
        "输出字段必须与以下 JSON Schema 一致，不得缺失或增加字段：\n"
        f"{SINGLE_PERSON_SCHEMA}\n"
        "要求：primary_emotion 必须来自 schema 枚举；secondary_emotion 可为 null。"
    )


def build_multi_person_prompt(person_count: int) -> str:
    """构建多人氛围分析任务指令。"""
    if person_count <= 0:
        raise ValueError("person_count must be positive.")

    return (
        f"请对当前约 {person_count} 人的多人场景进行群体氛围分析，并返回严格 JSON。"
        "输出字段必须与以下 JSON Schema 一致，不得缺失或增加字段：\n"
        f"{MULTI_PERSON_SCHEMA}\n"
        "要求：individual_emotions 中每个人物给出 primary_emotion 与 secondary_emotion；"
        "tension_level 与 engagement_level 为 0.0-1.0 浮点数。"
    )


def build_conversation(
    system_prompt: dict[str, Any],
    task_prompt: str,
    frames: list[Image.Image],
    audio: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """构建 transformers conversation 格式消息。"""
    if not frames:
        raise ValueError("frames must not be empty.")
    if not task_prompt.strip():
        raise ValueError("task_prompt must not be empty.")

    user_content: list[dict[str, Any]] = [{"type": "video", "video": frames}]
    if audio is not None:
        user_content.append({"type": "audio", "audio": audio})
    user_content.append({"type": "text", "text": task_prompt})

    return [
        system_prompt,
        {
            "role": "user",
            "content": user_content,
        },
    ]


__all__ = [
    "build_conversation",
    "build_multi_person_prompt",
    "build_single_person_prompt",
]
