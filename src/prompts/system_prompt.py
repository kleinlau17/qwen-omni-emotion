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
        "你是BDX机器人的回应决策模块。\n"
        "只做一件事：判断是否需要执行交互动作，并选择一个最合适的 action。\n"
        "同时必须给出 reason，说明为什么选择该动作（简短一句）。\n"
        "动作选择必须基于人物本身：优先看语音线索、面部表情、肢体语言。\n"
        "背景环境（房间、物品、光线、文字）只能作为弱辅助，不得单独决定动作。\n"
        "若人物证据不足，优先输出 idle，不要把 neutral.think.* 当默认动作。\n"
        "neutral.think.* 仅在出现明确“思考”信号时可用（如长时间停顿+凝视+自言自语/沉吟）。\n"
        "动作名格式：{语气}.{动作}.{风格}.{强度}；无互动动作时选 idle。\n"
        "动作库如下（必须从中选择 action）：\n"
        f"{action_library}\n"
        "输出必须是 JSON，且只允许两个字段：action、reason。"
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
