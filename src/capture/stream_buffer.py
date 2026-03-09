"""时间窗口缓冲 — 将连续的视频帧和音频块按推理窗口组织，供 pipeline 周期性取用"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Final

import numpy as np

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass
class InferenceWindow:
    """单个推理窗口的数据载体。"""

    frames: list[tuple[np.ndarray, float]] = field(default_factory=list)
    audio_chunks: list[tuple[np.ndarray, float]] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0

    def get_audio_array(self) -> np.ndarray:
        """将窗口中的音频块拼接成一个连续的一维数组。"""
        if not self.audio_chunks:
            return np.empty(0, dtype=np.float32)

        audio_arrays = [chunk.astype(np.float32, copy=False) for chunk, _ in self.audio_chunks]
        return np.concatenate(audio_arrays, axis=0)


class StreamBuffer:
    """采集层与推理层之间的线程安全窗口缓冲。"""

    def __init__(self, window_duration: float = 1.5, max_windows: int = 3) -> None:
        """
        初始化缓冲组件。

        Args:
            window_duration: 单个推理窗口时长（秒）。
            max_windows: 最多缓存的已完成窗口数量，超出后丢弃最旧窗口。
        """
        if window_duration <= 0:
            raise ValueError("window_duration 必须大于 0")
        if max_windows <= 0:
            raise ValueError("max_windows 必须大于 0")

        self._window_duration: float = window_duration
        self._max_windows: int = max_windows
        self._lock: Lock = Lock()
        self._current_window: InferenceWindow = InferenceWindow()
        self._ready_windows: list[InferenceWindow] = []
        self._current_started: bool = False

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """线程安全地写入一帧 RGB 图像。"""
        with self._lock:
            self._ensure_window_started_locked(timestamp=timestamp)
            self._current_window.frames.append((frame, timestamp))
            self._current_window.end_ts = max(self._current_window.end_ts, timestamp)
            self._rollover_if_needed_locked()

    def push_audio(self, chunk: np.ndarray, timestamp: float) -> None:
        """线程安全地写入一段音频块。"""
        with self._lock:
            self._ensure_window_started_locked(timestamp=timestamp)
            self._current_window.audio_chunks.append((chunk, timestamp))
            self._current_window.end_ts = max(self._current_window.end_ts, timestamp)
            self._rollover_if_needed_locked()

    def get_window(self) -> InferenceWindow | None:
        """
        非阻塞获取一个可推理窗口。

        Returns:
            可用窗口；若当前无完整窗口则返回 ``None``。
        """
        with self._lock:
            self._rollover_if_needed_locked()
            if not self._ready_windows:
                return None
            return self._ready_windows.pop(0)

    def get_windows_batch(self, max_count: int) -> list[InferenceWindow]:
        """非阻塞获取至多 *max_count* 个可推理窗口。

        Args:
            max_count: 本次最多取出的窗口数量。

        Returns:
            可用窗口列表（可能为空，最多 *max_count* 个）。
        """
        with self._lock:
            self._rollover_if_needed_locked()
            n = min(max_count, len(self._ready_windows))
            if n == 0:
                return []
            batch = self._ready_windows[:n]
            self._ready_windows = self._ready_windows[n:]
            return batch

    def reset(self) -> None:
        """清空当前与已完成窗口缓存。"""
        with self._lock:
            self._current_window = InferenceWindow()
            self._ready_windows.clear()
            self._current_started = False

    def _ensure_window_started_locked(self, timestamp: float) -> None:
        """在第一次写入时初始化窗口起止时间。"""
        if not self._current_started:
            self._current_window.start_ts = timestamp
            self._current_window.end_ts = timestamp
            self._current_started = True

    def _rollover_if_needed_locked(self) -> None:
        """当窗口达到设定时长时封存窗口并切换到新窗口。"""
        if not self._has_data_locked():
            return
        duration = self._current_window.end_ts - self._current_window.start_ts
        if duration < self._window_duration:
            return

        self._ready_windows.append(self._current_window)
        self._current_window = InferenceWindow()
        self._current_started = False

        if len(self._ready_windows) > self._max_windows:
            self._ready_windows.pop(0)
            LOGGER.warning("StreamBuffer 窗口缓存超限，已丢弃最旧窗口")

    def _has_data_locked(self) -> bool:
        return bool(self._current_window.frames or self._current_window.audio_chunks)
