"""后处理层测试 — 响应解析与状态追踪"""
from __future__ import annotations

from src.prompts.output_schema import AtmosphereResult, EmotionResult
from src.understanding.response_parser import (
    parse_atmosphere_response,
    parse_emotion_response,
)
from src.understanding.state_tracker import EmotionStateTracker


def test_parse_emotion_response_valid_json() -> None:
    """合法 JSON 应解析成功（简化 schema：仅 primary_emotion + secondary_emotion）。"""
    raw_text = """
    {
      "primary_emotion": "happy",
      "secondary_emotion": "surprised"
    }
    """
    result = parse_emotion_response(raw_text)
    assert isinstance(result, EmotionResult)
    assert result.primary_emotion == "happy"
    assert result.secondary_emotion == "surprised"
    assert result.person_id == "person_0"


def test_parse_emotion_response_embedded_json() -> None:
    """JSON 被包裹在其它文本中时也应能提取。"""
    raw_text = (
        'analysis result >>> {"primary_emotion":"neutral","secondary_emotion":null} <<< end'
    )
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.person_id == "person_0"


def test_parse_emotion_response_non_json_returns_none() -> None:
    """非 JSON 文本应返回 None。"""
    result = parse_emotion_response("this is not json at all")
    assert result is None


def test_parse_atmosphere_response_valid_json() -> None:
    """合法多人 JSON 应解析为 AtmosphereResult（简化 schema）。"""
    raw_text = """
    {
      "overall_mood": "focused",
      "tension_level": 0.2,
      "engagement_level": 0.88,
      "individual_emotions": [
        {
          "primary_emotion": "neutral",
          "secondary_emotion": null
        }
      ]
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
        secondary_emotion=None,
    )
    tracker.update(first, timestamp=1000.0)
    current = tracker.get_current_state("person_0")
    assert current is not None
    assert current.primary_emotion == "neutral"


def test_state_tracker_get_trend_and_detect_change() -> None:
    """趋势查询与 primary_emotion 突变检测。"""
    tracker = EmotionStateTracker(max_history=20)
    tracker.update(
        EmotionResult(
            person_id="person_0",
            primary_emotion="neutral",
            secondary_emotion=None,
        ),
        timestamp=1001.0,
    )
    tracker.update(
        EmotionResult(
            person_id="person_0",
            primary_emotion="angry",
            secondary_emotion=None,
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
                person_id="person_0",
                primary_emotion="happy",
                secondary_emotion=None,
            )
        ],
    )
    tracker.update(atmosphere, timestamp=1003.0)
    assert tracker.get_current_state("person_0") is not None
