"""提示词工程测试"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.prompts.output_schema import (
    AtmosphereResult,
    EmotionResult,
    MULTI_PERSON_SCHEMA,
    SINGLE_PERSON_SCHEMA,
    VALID_EMOTIONS,
)
from src.prompts.system_prompt import build_system_prompt
from src.prompts.task_prompts import (
    build_conversation,
    build_multi_person_prompt,
    build_single_person_prompt,
)


def test_emotion_result_creation_success() -> None:
    """应能创建合法 EmotionResult（简化版）。"""
    result = EmotionResult(
        person_id="person_0",
        primary_emotion="happy",
        secondary_emotion="neutral",
    )
    assert result.primary_emotion in VALID_EMOTIONS
    assert result.secondary_emotion == "neutral"


def test_emotion_result_invalid_emotion_raises() -> None:
    """primary_emotion 非法时应抛出 ValueError。"""
    with pytest.raises(ValueError):
        EmotionResult(
            person_id="person_0",
            primary_emotion="invalid_emotion",
            secondary_emotion=None,
        )


def test_atmosphere_result_creation_success() -> None:
    """应能创建合法 AtmosphereResult。"""
    item = EmotionResult(
        person_id="person_1",
        primary_emotion="neutral",
        secondary_emotion=None,
    )
    result = AtmosphereResult(
        overall_mood="focused",
        tension_level=0.3,
        engagement_level=0.75,
        individual_emotions=[item],
    )
    assert result.individual_emotions[0].person_id == "person_1"


def test_schema_strings_are_valid_json() -> None:
    """Schema 文本应是可解析 JSON。"""
    single_schema = json.loads(SINGLE_PERSON_SCHEMA)
    multi_schema = json.loads(MULTI_PERSON_SCHEMA)
    assert single_schema["type"] == "object"
    assert multi_schema["type"] == "object"
    assert "individual_emotions" in multi_schema["properties"]


def test_build_system_prompt_format() -> None:
    """system prompt 返回值应符合 conversation 结构。"""
    prompt = build_system_prompt()
    assert prompt["role"] == "system"
    assert isinstance(prompt["content"], list)
    assert prompt["content"][0]["type"] == "text"


def test_build_task_prompts_contain_schema() -> None:
    """任务 prompt 应包含 schema 片段。"""
    single = build_single_person_prompt()
    multi = build_multi_person_prompt(person_count=3)
    assert '"primary_emotion"' in single
    assert '"individual_emotions"' in multi


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
