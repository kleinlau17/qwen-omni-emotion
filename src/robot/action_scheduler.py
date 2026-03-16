"""动作调度层 — 在模型输出与机器人发送之间进行调度

接收 EmotionResult，经调度策略处理后输出待执行动作列表。
支持节流：按可配置间隔（默认 5 秒）才将动作发送给机器人，避免频繁切换。
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from src.prompts.output_schema import EmotionResult
from src.robot.action_name_mapping import map_action_to_legacy

LOGGER = logging.getLogger(__name__)


class ActionScheduler:
    """动作调度器。

    将模型输出的 EmotionResult 转为待发送给机器人的动作序列。
    支持节流：仅在距上次发送超过 send_interval_seconds 时才会输出动作。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """初始化调度器。

        Args:
            config: 调度配置，支持:
                - send_interval_seconds: 动作切换最小间隔（秒），默认 5.0
                - repeat_interval_seconds: 同动作重复发送间隔（秒），默认 6.0
                - min_hold_seconds: 动作最小驻留时长（秒），默认 1.0
                - force_switch_confidence: 高置信度动作强制切换阈值，默认 0.8
                - pass_through: 若为 True 则忽略节流、透传（兼容旧行为），默认 False
        """
        self._config = config or {}
        self._send_interval = float(
            self._config.get("send_interval_seconds", 5.0)
        )
        self._repeat_interval = float(
            self._config.get("repeat_interval_seconds", 3.0)
        )
        self._min_hold_seconds = float(
            self._config.get("min_hold_seconds", 0.8)
        )
        self._force_switch_confidence = float(
            self._config.get("force_switch_confidence", 0.65)
        )
        self._pass_through = bool(self._config.get("pass_through", False))
        self._last_by_person: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        result: EmotionResult,
        timestamp: float,
    ) -> list[str]:
        """提交一次推理结果，经调度后返回待执行的动作列表。

        Args:
            result: 单次推理得到的情绪与动作结果
            timestamp: 推理窗口结束时间戳

        Returns:
            待发送给机器人的动作名称列表，空列表表示本帧因节流无需发送。
        """
        actions = self._schedule(result, timestamp)
        mapped_actions = [map_action_to_legacy(action) for action in actions]
        if mapped_actions:
            LOGGER.debug(
                "调度输出: person=%s action=%s -> %s",
                result.person_id,
                result.action,
                mapped_actions,
            )
        return mapped_actions

    def _schedule(
        self,
        result: EmotionResult,
        timestamp: float,
    ) -> list[str]:
        """调度策略：按人物状态机节流，优先处理动作切换。"""
        if result.action == "idle":
            LOGGER.debug(
                "idle 动作跳过发送: person=%s", result.person_id,
            )
            with self._lock:
                self._last_by_person.setdefault(result.person_id, {
                    "last_action": "idle",
                    "last_send_ts": timestamp,
                }).update({"last_action": "idle", "last_send_ts": timestamp})
            return []

        if self._pass_through:
            return [result.action]

        with self._lock:
            person_state = self._last_by_person.get(result.person_id)
            if person_state is None:
                self._last_by_person[result.person_id] = {
                    "last_action": result.action,
                    "last_send_ts": timestamp,
                }
                return [result.action]

            last_action = str(person_state["last_action"])
            elapsed = float(timestamp - float(person_state["last_send_ts"]))
            hold_seconds = max(
                self._min_hold_seconds,
                float(result.hold_seconds),
            )

            # 动作不变：仅做低频重复发送，避免机器人动作抖动。
            if result.action == last_action:
                if elapsed < self._repeat_interval:
                    return []
                person_state["last_send_ts"] = timestamp
                return [result.action]

            # 动作变化：短时间内仅在高置信度下强制切换。
            if (
                elapsed < hold_seconds
                and result.action_confidence < self._force_switch_confidence
            ):
                return []
            if (
                elapsed < self._send_interval
                and result.action_confidence < self._force_switch_confidence
            ):
                return []

            person_state["last_action"] = result.action
            person_state["last_send_ts"] = timestamp
        return [result.action]


__all__ = ["ActionScheduler"]
