"""帧采样策略 — 从 30fps 原始流中智能选帧，平衡精度与延迟"""
from __future__ import annotations

import logging
from typing import Final

import numpy as np

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


class FrameSampler:
    """带时间戳帧序列采样器。"""

    def __init__(self, strategy: str = "uniform", max_frames: int = 4) -> None:
        """
        初始化帧采样器。

        Args:
            strategy: 采样策略，目前仅支持 ``uniform``。
            max_frames: 单窗口最多保留的帧数。
        """
        if max_frames <= 0:
            raise ValueError("max_frames 必须大于 0")
        if strategy != "uniform":
            raise ValueError(f"不支持的采样策略: {strategy}")

        self._strategy: str = strategy
        self._max_frames: int = max_frames

    def sample(self, frames: list[tuple[np.ndarray, float]]) -> list[np.ndarray]:
        """
        从带时间戳帧列表中采样，返回最多 ``max_frames`` 帧。

        Args:
            frames: 原始帧序列，元素为 ``(frame, timestamp)``。

        Returns:
            采样后的 RGB 帧列表。
        """
        if not frames:
            return []

        if len(frames) <= self._max_frames:
            return [frame for frame, _ in frames]

        if self._strategy == "uniform":
            indices = self._uniform_indices(total=len(frames), count=self._max_frames)
            return [frames[index][0] for index in indices]

        LOGGER.warning("未知策略，回退到前 max_frames 帧: %s", self._strategy)
        return [frame for frame, _ in frames[: self._max_frames]]

    def _uniform_indices(self, total: int, count: int) -> list[int]:
        """计算等间隔采样索引，保证包含首帧与尾帧。"""
        if count == 1:
            return [0]

        step = max(1, total // (count - 1))
        indices: list[int] = [min(index * step, total - 1) for index in range(count - 1)]
        indices.append(total - 1)
        return indices
