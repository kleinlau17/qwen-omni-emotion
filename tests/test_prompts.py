"""提示词工程测试"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.prompts.output_schema import (
    EmotionResult,
    SINGLE_PERSON_SCHEMA,
    VALID_ACTIONS,
    VALID_EMOTIONS,
)
from src.prompts.system_prompt import build_system_prompt
from src.prompts.task_prompts import (
    build_conversation,
    build_single_person_prompt,
)


def test_emotion_result_creation_success() -> None:
    """应能创建合法 EmotionResult。"""
    result = EmotionResult(
        person_id="person_0",
        detected_emotion="happy",
        action=VALID_ACTIONS[0],
    )
    assert result.detected_emotion in VALID_EMOTIONS


def test_emotion_result_invalid_emotion_raises() -> None:
    """detected_emotion 非法时应抛出 ValueError。"""
    with pytest.raises(ValueError):
        EmotionResult(
            person_id="person_0",
            detected_emotion="invalid_emotion",
            action=VALID_ACTIONS[0],
        )


def test_schema_strings_are_valid_json() -> None:
    """Schema 文本应是可解析 JSON。"""
    single_schema = json.loads(SINGLE_PERSON_SCHEMA)
    assert single_schema["type"] == "object"
    assert "action" in single_schema["properties"]
    assert "reason" in single_schema["properties"]


def test_build_system_prompt_format() -> None:
    """system prompt 返回值应符合 conversation 结构。"""
    prompt = build_system_prompt()
    assert prompt["role"] == "system"
    assert isinstance(prompt["content"], list)
    assert prompt["content"][0]["type"] == "text"


def test_build_task_prompts_contain_key_elements() -> None:
    """system prompt 含动作库；单人 prompt 含 action few-shot。"""
    system_text = build_system_prompt()["content"][0]["text"]
    assert "action" in system_text
    assert "动作库" in system_text
    assert "优先看语音线索、面部表情、肢体语言" in system_text
    assert "不要把 neutral.think.* 当默认动作" in system_text
    single = build_single_person_prompt()
    assert '"action"' in single
    assert '"reason"' in single
    assert "参考示例" in single
    assert "环境信息仅作弱参考" in single
    assert "证据不足或冲突时，输出 idle" in single


def test_build_conversation_with_audio() -> None:
    """带音频时应构建 video+audio+text。"""
    frames = [Image.new("RGB", (32, 32), color=(255, 0, 0))]
    audio = np.zeros(16000, dtype=np.float32)
    conversation = build_conversation(
        system_prompt=build_system_prompt(),
        task_prompt=build_single_person_prompt(),
        frames=frames,
        audio=audio,
    )

    assert len(conversation) == 2
    assert conversation[1]["role"] == "user"
    content = conversation[1]["content"]
    assert content[0]["type"] == "video"
    assert content[1]["type"] == "audio"
    assert content[2]["type"] == "text"


def test_build_conversation_without_audio() -> None:
    """无音频时应仅包含 video+text。"""
    frames = [Image.new("RGB", (16, 16), color=(0, 255, 0))]
    conversation = build_conversation(
        system_prompt=build_system_prompt(),
        task_prompt="analyze",
        frames=frames,
    )
    content = conversation[1]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "video"
    assert content[1]["type"] == "text"
