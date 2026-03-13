"""分析任务指令模板 — 单人情绪 / 多人氛围 / 场景切换的 prompt 构建"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.prompts.output_schema import ACTION_GUIDE, MULTI_PERSON_SCHEMA, SINGLE_PERSON_SCHEMA


def build_single_person_prompt() -> str:
    """构建单人情绪分析任务指令。"""
    return (
        "请对当前单人场景进行情绪识别与互动决策，并返回严格 JSON。"
        "输出字段必须与以下 JSON Schema 一致，不得缺失或增加字段：\n"
        f"{SINGLE_PERSON_SCHEMA}\n"
        "动作释义（action → 含义）：\n"
        + "\n".join(ACTION_GUIDE)
        + "\n"
        "动作选择原则："
        "1) 动作需匹配识别情绪，同时结合互动者的动作与表情（例如对方靠近/挥手/退缩/皱眉/微笑等）；"
        "2) 行为需与情境与 BDX 性格一致（怂萌、谨慎、尊重、被冷落会收敛）。\n"
        "要求：detected_emotion 为识别出的对象情绪；self_emotion 为 BDX 自身情绪；"
        "action 为 BDX 交互执行的动作，必须来自 schema 枚举。"
        "必须输出完整 JSON（包含 action 值并以 } 结束），建议单行输出，不要省略字段。"
    )


def build_multi_person_prompt(person_count: int) -> str:
    """构建多人氛围分析任务指令。"""
    if person_count <= 0:
        raise ValueError("person_count must be positive.")

    return (
        f"请对当前约 {person_count} 人的多人场景进行群体氛围分析，并给出 BDX 的互动决策，返回严格 JSON。"
        "输出字段必须与以下 JSON Schema 一致，不得缺失或增加字段：\n"
        f"{MULTI_PERSON_SCHEMA}\n"
        "动作释义（action → 含义）：\n"
        + "\n".join(ACTION_GUIDE)
        + "\n"
        "动作选择原则："
        "1) 动作需匹配识别情绪，同时结合互动者的动作与表情（例如对方靠近/挥手/退缩/皱眉/微笑等）；"
        "2) 仅当明显需要“娱乐/庆祝”时才用 dance_twist；"
        "3) 若情绪为 happy/neutral，优先选择轻量互动（yes_1/yes_2/hell_yes/laugh_giggle/curious/perk_up/attention/standard_yes）；"
        "4) 避免连续多次输出同一 action（尤其是 dance_twist），若无强理由请改用语义接近但不同的动作；"
        "5) 行为需与情境与 BDX 性格一致（怂萌、谨慎、尊重、被冷落会收敛）。\n"
        "要求：individual_emotions 中每个人物给出 detected_emotion、self_emotion、action；"
        "tension_level 与 engagement_level 为 0.0-1.0 浮点数。"
        "必须输出完整 JSON（包含 action 值并以 } 结束），建议单行输出，不要省略字段。"
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
