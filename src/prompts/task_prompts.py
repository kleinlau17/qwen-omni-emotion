"""分析任务指令模板 — 单人动作决策 prompt 构建"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def build_single_person_prompt() -> str:
    """构建单人情绪分析任务指令（精简版，减少 token 数）。"""
    return (
        "如果画面中有人，请仔细观察人物的情绪、肢体动作，以及语音内容/语气来判断是否需要机器人执行回应动作。\n"
        "规则：\n"
        "- 必须从动作库选择 action。\n"
        "- 若人物信息不足，优先输出 idle 动作。\n"
        '- 输出必须包含两个字段：action 和 reason（<=10字）。\n'
        "参考示例：\n"
        '{"action":"idle","reason":"用户在思考。"}\n'
        '{"action":"neutral.recover.default.mid","reason":"用户很沮丧。"}\n'
        '{"action":"neutral.affirm.playful.low","reason":"用户很开心，在打招呼。"}\n'
        "仅输出 JSON。"
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
    "build_single_person_prompt",
]
