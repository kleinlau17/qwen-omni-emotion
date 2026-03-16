"""系统角色设定 — 定义模型作为 BDX 机器人共情交互大脑的基础人设与行为约束"""
from __future__ import annotations

from typing import Any
from src.prompts.output_schema import format_action_library

def build_system_prompt() -> dict[str, Any]:
    """构建情绪理解任务使用的系统提示词消息。

    prompt 经过极限压缩以减少输入 token 数，加速推理。
    """
    action_library = format_action_library()
    system_text = (
        "你是BDX机器人的交互决策模块。\n"
        "你负责从动作库中选择一个最合适的 action 进行交互。\n"
        "同时必须给出 reason，说明为什么选择该动作（简短一句）。\n"
        "动作选择必须基于人物情绪、肢体动作以及语音内容/语气。\n"
        "若判断信息不足，优先输出 idle 动作。\n"
        "动作库如下（必须从中选择 action）：\n"
        f"{action_library}\n"
        "输出必须是 JSON，且必须包含两个字段：action、reason。"
    )
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_text,
            }
        ],
    }


__all__ = ["build_system_prompt"]
