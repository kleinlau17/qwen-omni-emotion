"""动作调度层 — 在模型输出与机器人发送之间进行调度

接收 EmotionResult，经调度策略处理后输出待执行动作列表。
支持节流：按可配置间隔（默认 5 秒）才将动作发送给机器人，避免频繁切换。
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from src.prompts.output_schema import EmotionResult

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
                - send_interval_seconds: 两次发送之间的最小间隔（秒），默认 5.0
                - pass_through: 若为 True 则忽略节流、透传（兼容旧行为），默认 False
        """
        self._config = config or {}
        self._send_interval = float(
            self._config.get("send_interval_seconds", 5.0)
        )
        self._pass_through = bool(self._config.get("pass_through", False))
        self._last_send_ts: float = 0.0
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
        if actions:
            LOGGER.debug(
                "调度输出: person=%s action=%s -> %s",
                result.person_id,
                result.action,
                actions,
            )
        return actions

    def _schedule(
        self,
        result: EmotionResult,
        timestamp: float,
    ) -> list[str]:
        """调度策略：节流 — 仅当距上次发送超过 send_interval_seconds 时输出动作。"""
        if self._pass_through:
            return [result.action]

        with self._lock:
            elapsed = timestamp - self._last_send_ts
            if elapsed < self._send_interval:
                return []
            self._last_send_ts = timestamp
        return [result.action]


__all__ = ["ActionScheduler"]
