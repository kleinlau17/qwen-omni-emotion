"""分析任务指令模板 — 单人情绪 / 多人氛围 / 场景切换的 prompt 构建"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.model.base import InferenceRequest
from src.prompts.output_schema import MULTI_PERSON_SCHEMA, SINGLE_PERSON_SCHEMA


def build_single_person_prompt() -> str:
    """构建通用情绪分析任务指令（每次针对一个人物或主体）。"""
    return (
        "请对当前时间段内画面中需要你重点关注的这一位人物（或主要情绪主体）进行情绪分析，"
        "尤其关注其面部表情随时间的细微变化、眼神和注视方向、肩颈与上半身的紧张或放松程度，"
        "以及是否存在不自觉的小动作（例如搓手、跺脚、身体前倾或后仰等），据此综合判断对方情绪。"
        "无需假定场景中只有一个人，但本次回答只针对你聚焦的这一位人物。"
        "请返回严格 JSON，输出字段必须与以下 JSON Schema 一致，不得缺失或增加字段：\n"
        f"{SINGLE_PERSON_SCHEMA}\n"
        "要求：\n"
        "1) 只需给出 primary_emotion 与 secondary_emotion（可为 null）；\n"
        "2) primary_emotion 必须来自 schema 中给出的枚举集合；\n"
        "3) secondary_emotion 若存在，也必须来自同一枚举集合。"
    )


def build_multi_person_prompt(person_count: int) -> str:
    """构建多人氛围分析任务指令。"""
    if person_count <= 0:
        raise ValueError("person_count must be positive.")

    return (
        f"请对当前约 {person_count} 人的多人场景进行群体氛围分析，并返回严格 JSON。"
        "输出字段必须与以下 JSON Schema 一致，不得缺失或增加字段：\n"
        f"{MULTI_PERSON_SCHEMA}\n"
        "要求：\n"
        "1) individual_emotions 中每个人物都给出 person_id 与情绪结果；\n"
        "2) tension_level 与 engagement_level 为 0.0-1.0 浮点数；\n"
        "3) description 总结整体氛围及关键线索。"
    )


def build_conversation(
    system_prompt: dict[str, Any],
    task_prompt: str,
    frames: list[Image.Image],
    audio: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """构建 transformers conversation 格式消息。"""
    return build_inference_request(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        frames=frames,
        audio=audio,
    ).to_conversation()


def build_inference_request(
    system_prompt: dict[str, Any] | str,
    task_prompt: str,
    frames: list[Image.Image],
    audio: np.ndarray | None = None,
    use_audio: bool = True,
    metadata: dict[str, Any] | None = None,
) -> InferenceRequest:
    """构建后端无关的推理请求对象。"""
    if not frames:
        raise ValueError("frames must not be empty.")
    if not task_prompt.strip():
        raise ValueError("task_prompt must not be empty.")
    if isinstance(system_prompt, dict):
        content = system_prompt.get("content", [])
        if not isinstance(content, list) or not content:
            raise ValueError("system_prompt content must be a non-empty list.")
        text = content[0].get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("system_prompt text must not be empty.")
        system_prompt_text = text
    else:
        system_prompt_text = system_prompt
    return InferenceRequest(
        system_prompt=system_prompt_text,
        task_prompt=task_prompt,
        frames=frames,
        audio=audio,
        use_audio=use_audio,
        metadata=metadata or {},
    )


__all__ = [
    "build_conversation",
    "build_inference_request",
    "build_multi_person_prompt",
    "build_single_person_prompt",
]
