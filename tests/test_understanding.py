"""后处理层测试 — 响应解析与状态追踪"""
from __future__ import annotations

from src.prompts.output_schema import EmotionResult, VALID_ACTIONS
from src.understanding.response_parser import parse_emotion_response
from src.understanding.state_tracker import EmotionStateTracker


def test_parse_emotion_response_valid_json() -> None:
    """合法 JSON 应解析成功。"""
    raw_text = """
    {
      "action": "neutral.surprise.quick.low",
      "reason": "用户突然提高音量，机器人给出轻微惊讶反馈。"
    }
    """
    result = parse_emotion_response(raw_text)
    assert isinstance(result, EmotionResult)
    assert result.detected_emotion == "neutral"
    assert result.action == "neutral.surprise.quick.low"
    assert result.action_confidence == 0.5
    assert result.hold_seconds == 1.0
    assert result.person_id == "person_0"
    assert result.reason == "用户突然提高音量，机器人给出轻微惊讶反馈。"


def test_parse_emotion_response_embedded_json() -> None:
    """JSON 被包裹在其它文本中时也应能提取。"""
    raw_text = (
        'analysis result >>> {"action":"neutral.surprise.quick.low","reason":"检测到短暂惊讶线索。"} <<< end'
    )
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.person_id == "person_0"
    assert result.reason == "检测到短暂惊讶线索。"


def test_parse_emotion_response_unknown_action_with_affirm_keyword_fallback() -> None:
    """库外动作包含 affirm 关键词时应回退到合法动作。"""
    raw_text = '{"action":"neutral.affirm.default.low","reason":"用户表达同意，进行认同反馈。"}'
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.action == "neutral.affirm.default.mid"
    assert result.reason == "用户表达同意，进行认同反馈。"


def test_parse_emotion_response_unknown_action_keyword_fallback() -> None:
    """库外动作名应按关键词回退到合法动作。"""
    raw_text = (
        '{"action":"neutral.scan.inquiring.mid","reason":"用户停顿后看向机器人，像是在等追问。"}'
    )
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.action == "neutral.question.default.mid"


def test_parse_emotion_response_unknown_action_without_keyword_to_idle() -> None:
    """无法识别的动作应安全回退为 idle。"""
    raw_text = '{"action":"totally.unknown.motion","reason":"动作名不在库内。"}'
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.action == "idle"


def test_parse_emotion_response_non_json_returns_none() -> None:
    """非 JSON 文本应返回 None。"""
    result = parse_emotion_response("this is not json at all")
    assert result is None


def test_parse_emotion_response_with_action_fields() -> None:
    """附加字段存在时也应可解析。"""
    raw_text = (
        '{"action":"neutral.dialogue.default.mid","reason":"用户持续说话，保持对话态势。",'
        '"action_confidence":0.77,"hold_seconds":1.6}'
    )
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.action_confidence == 0.77
    assert result.hold_seconds == 1.6
    assert result.reason == "用户持续说话，保持对话态势。"


def test_parse_emotion_response_missing_reason_uses_default() -> None:
    """缺少 reason 时应回退默认原因文本。"""
    raw_text = '{"action":"neutral.dialogue.default.mid"}'
    result = parse_emotion_response(raw_text)
    assert result is not None
    assert result.reason == "未提供动作原因。"


def test_state_tracker_update_and_get_current_state() -> None:
    """更新后应能读取最新状态。"""
    tracker = EmotionStateTracker(max_history=20)
    first = EmotionResult(
        person_id="person_0",
        detected_emotion="neutral",
        action=VALID_ACTIONS[0],
    )
    tracker.update(first, timestamp=1000.0)
    current = tracker.get_current_state("person_0")
    assert current is not None
    assert current.detected_emotion == "neutral"


def test_state_tracker_get_trend_and_detect_change() -> None:
    """趋势查询与 detected_emotion 突变检测。"""
    tracker = EmotionStateTracker(max_history=20)
    tracker.update(
        EmotionResult(
            person_id="person_0",
            detected_emotion="neutral",
            action=VALID_ACTIONS[0],
        ),
        timestamp=1001.0,
    )
    tracker.update(
        EmotionResult(
            person_id="person_0",
            detected_emotion="angry",
            action=VALID_ACTIONS[0],
        ),
        timestamp=1002.0,
    )

    trend = tracker.get_trend("person_0", window_count=5)
    assert len(trend) == 2
    assert tracker.detect_change("person_0", threshold=0.3)


