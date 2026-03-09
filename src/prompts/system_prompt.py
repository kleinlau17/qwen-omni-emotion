"""系统角色设定 — 定义模型作为情绪分析专家的基础人设与行为约束"""
from __future__ import annotations

from typing import Any


def build_system_prompt() -> dict[str, Any]:
    """构建情绪理解任务使用的系统提示词消息。"""
    system_text = (
        "你是一位多模态情绪分析专家。"
        "你将结合视频帧中的面部表情与肢体语言，以及音频中的语音语调进行综合判断。"
        "请严格遵循任务指令中的 JSON Schema 输出，仅返回一个合法 JSON 对象。"
        "禁止输出任何 JSON 之外的解释、前后缀、Markdown 或额外文本。"
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
