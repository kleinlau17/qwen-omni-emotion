"""后处理层测试 — 响应解析与状态追踪"""
from __future__ import annotations

from src.prompts.output_schema import AtmosphereResult, EmotionResult
from src.understanding.response_parser import (
    parse_atmosphere_response,
    parse_emotion_response,
)
from src.understanding.state_tracker import EmotionStateTracker


def test_parse_emotion_response_valid_json() -> None:
    """合法 JSON 应解析成功。"""
    raw_text = """
    {
      "person_id": "person_0",
      "primary_emotion": "happy",
      "emotion_intensity": 0.82,
      "secondary_emotion": "surprised",
      "confidence": 0.91,
      "description": "smiling with energetic tone"
    }
    """
    result = parse_emotion_response(raw_text)
    assert isinstance(result, EmotionResult)
    assert result.primary_emotion == "happy"


def test_parse_emotion_response_embedded_json() -> None:
    """JSON 被包裹在其它文本中时也应能提取。"""
    raw_text = (
        'analysis result >>> {"person_id":"person_1","primary_emotion":"neutral",'
        '"emotion_intensity":0.31,"secondary_emotion":null,"confidence":0.7,'
        '"description":"calm and steady"} <<< end'
    )
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.person_id == "person_1"


def test_parse_emotion_response_non_json_returns_none() -> None:
    """非 JSON 文本应返回 None。"""
    result = parse_emotion_response("this is not json at all")
    assert result is None


def test_parse_atmosphere_response_valid_json() -> None:
    """合法多人 JSON 应解析为 AtmosphereResult。"""
    raw_text = """
    {
      "overall_mood": "focused",
      "tension_level": 0.2,
      "engagement_level": 0.88,
      "individual_emotions": [
        {
          "person_id": "person_0",
          "primary_emotion": "neutral",
          "emotion_intensity": 0.4,
          "secondary_emotion": null,
          "confidence": 0.83,
          "description": "watching attentively"
        }
      ],
      "description": "group is attentive and collaborative"
    }
    """
    result = parse_atmosphere_response(raw_text)
    assert isinstance(result, AtmosphereResult)
    assert result.individual_emotions[0].person_id == "person_0"


def test_parse_atmosphere_response_invalid_payload_returns_none() -> None:
    """字段不完整时应返回 None。"""
    raw_text = '{"overall_mood":"tense","tension_level":0.9}'
    result = parse_atmosphere_response(raw_text)
    assert result is None


def test_state_tracker_update_and_get_current_state() -> None:
    """更新后应能读取最新状态。"""
    tracker = EmotionStateTracker(max_history=20)
    first = EmotionResult(
        person_id="person_0",
        primary_emotion="neutral",
        emotion_intensity=0.2,
        secondary_emotion=None,
        confidence=0.8,
        description="baseline",
    )
    tracker.update(first, timestamp=1000.0)
    current = tracker.get_current_state("person_0")
    assert current is not None
    assert current.primary_emotion == "neutral"


def test_state_tracker_get_trend_and_detect_change() -> None:
    """趋势查询与突变检测应符合阈值逻辑。"""
    tracker = EmotionStateTracker(max_history=20)
    tracker.update(
        EmotionResult(
            person_id="person_0",
            primary_emotion="neutral",
            emotion_intensity=0.1,
            secondary_emotion=None,
            confidence=0.7,
            description="calm",
        ),
        timestamp=1001.0,
    )
    tracker.update(
        EmotionResult(
            person_id="person_0",
            primary_emotion="angry",
            emotion_intensity=0.65,
            secondary_emotion=None,
            confidence=0.85,
            description="strong voice and tense posture",
        ),
        timestamp=1002.0,
    )

    trend = tracker.get_trend("person_0", window_count=5)
    assert len(trend) == 2
    assert tracker.detect_change("person_0", threshold=0.3)


def test_state_tracker_update_with_atmosphere_result() -> None:
    """AtmosphereResult 更新应写入每个人物状态。"""
    tracker = EmotionStateTracker(max_history=20)
    atmosphere = AtmosphereResult(
        overall_mood="engaged",
        tension_level=0.4,
        engagement_level=0.9,
        individual_emotions=[
            EmotionResult(
                person_id="person_2",
                primary_emotion="happy",
                emotion_intensity=0.76,
                secondary_emotion=None,
                confidence=0.88,
                description="smiling and nodding",
            )
        ],
        description="positive collaboration",
    )
    tracker.update(atmosphere, timestamp=1003.0)
    assert tracker.get_current_state("person_2") is not None
