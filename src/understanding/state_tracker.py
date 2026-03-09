"""状态追踪 — 跨推理窗口的情绪趋势累积与变化检测"""
from __future__ import annotations

from collections import deque
from threading import Lock

from src.prompts.output_schema import AtmosphereResult, EmotionResult


class EmotionStateTracker:
    """跨推理窗口维护人物情绪时序状态。"""

    def __init__(self, max_history: int = 20) -> None:
        """初始化状态追踪器。"""
        if max_history <= 1:
            raise ValueError("max_history must be greater than 1.")
        self._max_history = max_history
        self._history: dict[str, deque[tuple[EmotionResult, float]]] = {}
        self._lock = Lock()

    def update(self, result: EmotionResult | AtmosphereResult, timestamp: float) -> None:
        """写入一次新的推理结果。"""
        with self._lock:
            if isinstance(result, EmotionResult):
                self._append_result(result, timestamp)
                return
            for emotion in result.individual_emotions:
                self._append_result(emotion, timestamp)

    def get_trend(self, person_id: str, window_count: int = 5) -> list[EmotionResult]:
        """获取指定人物最近 N 次情绪结果。"""
        if window_count <= 0:
            raise ValueError("window_count must be positive.")
        with self._lock:
            person_history = self._history.get(person_id)
            if person_history is None:
                return []
            return [item[0] for item in list(person_history)[-window_count:]]

    def detect_change(self, person_id: str, threshold: float = 0.3) -> bool:
        """检测人物最近两次情绪强度变化是否超过阈值。"""
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        with self._lock:
            person_history = self._history.get(person_id)
            if person_history is None or len(person_history) < 2:
                return False
            latest = person_history[-1][0]
            previous = person_history[-2][0]
            return abs(latest.emotion_intensity - previous.emotion_intensity) > threshold

    def get_current_state(self, person_id: str) -> EmotionResult | None:
        """获取人物当前最新情绪状态。"""
        with self._lock:
            person_history = self._history.get(person_id)
            if not person_history:
                return None
            return person_history[-1][0]

    def _append_result(self, result: EmotionResult, timestamp: float) -> None:
        """向人物历史中追加一次情绪结果。"""
        person_history = self._history.get(result.person_id)
        if person_history is None:
            person_history = deque(maxlen=self._max_history)
            self._history[result.person_id] = person_history
        person_history.append((result, timestamp))


__all__ = ["EmotionStateTracker"]
