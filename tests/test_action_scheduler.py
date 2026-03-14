"""动作调度层测试。"""
from __future__ import annotations

from src.prompts.output_schema import EmotionResult
from src.robot.action_scheduler import ActionScheduler


def _result(
    action: str,
    *,
    person_id: str = "person_0",
    confidence: float = 0.7,
    hold_seconds: float = 1.2,
) -> EmotionResult:
    return EmotionResult(
        person_id=person_id,
        detected_emotion="neutral",
        action=action,
        action_confidence=confidence,
        hold_seconds=hold_seconds,
    )


def test_scheduler_first_action_should_send() -> None:
    """首次动作应直接发送。"""
    scheduler = ActionScheduler(config={"send_interval_seconds": 1.0})
    actions = scheduler.submit(_result("neutral.question.default.low"), timestamp=1.0)
    assert actions == ["question_1"]


def test_scheduler_same_action_should_be_throttled() -> None:
    """同动作短时间重复应被抑制。"""
    scheduler = ActionScheduler(
        config={"send_interval_seconds": 1.0, "repeat_interval_seconds": 5.0}
    )
    scheduler.submit(_result("neutral.question.default.low"), timestamp=1.0)
    actions = scheduler.submit(_result("neutral.question.default.low"), timestamp=2.0)
    assert actions == []


def test_scheduler_switch_action_low_confidence_should_wait() -> None:
    """动作切换但低置信度时应等待最小时长。"""
    scheduler = ActionScheduler(
        config={
            "send_interval_seconds": 1.0,
            "min_hold_seconds": 1.5,
            "force_switch_confidence": 0.85,
        }
    )
    scheduler.submit(_result("neutral.question.default.low"), timestamp=1.0)
    actions = scheduler.submit(
        _result("neutral.deny.default.mid", confidence=0.7, hold_seconds=1.6),
        timestamp=1.8,
    )
    assert actions == []


def test_scheduler_switch_action_high_confidence_can_interrupt() -> None:
    """高置信度动作切换应允许提前打断。"""
    scheduler = ActionScheduler(
        config={
            "send_interval_seconds": 1.0,
            "min_hold_seconds": 1.5,
            "force_switch_confidence": 0.85,
        }
    )
    scheduler.submit(_result("neutral.question.default.low"), timestamp=1.0)
    actions = scheduler.submit(
        _result("neutral.deny.default.mid", confidence=0.92, hold_seconds=1.8),
        timestamp=1.3,
    )
    assert actions == ["standard_no"]
