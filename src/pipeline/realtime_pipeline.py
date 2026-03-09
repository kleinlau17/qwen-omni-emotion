"""端到端实时流水线 — 串联采集→预处理→推理→理解，控制延迟 <400ms"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from typing import Any

import numpy as np

from PIL import Image

from src.capture.audio_capture import AudioCapture
from src.capture.stream_buffer import StreamBuffer
from src.capture.video_capture import VideoCapture
from src.model.qwen_omni import QwenOmniModel
from src.preprocessing.frame_sampler import FrameSampler
from src.preprocessing.roi_extractor import ROIExtractor
from src.prompts.output_schema import EmotionResult
from src.prompts.system_prompt import build_system_prompt
from src.prompts.task_prompts import build_conversation, build_single_person_prompt
from src.understanding.response_parser import parse_emotion_response
from src.understanding.state_tracker import EmotionStateTracker

LOGGER = logging.getLogger(__name__)


class RealtimePipeline:
    """实时情绪理解主管道。"""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        根据配置初始化所有子组件。

        Args:
            config: 合并后的全局配置字典。
        """
        LOGGER.info("  [1/6] 解析配置 ...")
        self._config: dict[str, Any] = config

        capture_cfg = config.get("capture", {})
        video_cfg = capture_cfg.get("video", {})
        audio_cfg = capture_cfg.get("audio", {})

        stream_cfg = config.get("stream_buffer", {})
        preprocess_cfg = config.get("preprocessing", {})
        sampling_cfg = preprocess_cfg.get("frame_sampling", {})
        roi_cfg = preprocess_cfg.get("roi", {})

        model_cfg = config.get("model", {})
        infer_cfg = config.get("inference", {})
        perf_cfg = config.get("performance", {})
        understanding_cfg = config.get("understanding", {})

        video_resolution_raw = video_cfg.get("resolution", [1920, 1080])
        video_resolution = tuple(int(v) for v in video_resolution_raw)

        LOGGER.info("  [2/6] 初始化视频采集 (VideoCapture %dx%d@%dfps) ...",
                    video_resolution[0], video_resolution[1], int(video_cfg.get("fps", 30)))
        self.video_capture = VideoCapture(
            resolution=(video_resolution[0], video_resolution[1]),
            fps=int(video_cfg.get("fps", 30)),
        )
        LOGGER.info("  [3/6] 初始化音频采集 (AudioCapture %dHz) ...",
                    int(audio_cfg.get("sample_rate", 16000)))
        self.audio_capture = AudioCapture(
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            channels=int(audio_cfg.get("channels", 1)),
        )
        self.stream_buffer = StreamBuffer(
            window_duration=float(stream_cfg.get("window_duration_seconds", 1.5)),
            max_windows=int(stream_cfg.get("max_windows_in_buffer", 3)),
        )
        LOGGER.info("  [4/6] 初始化预处理 (FrameSampler + ROIExtractor) ...")
        self.frame_sampler = FrameSampler(
            strategy=str(sampling_cfg.get("strategy", "uniform")),
            max_frames=int(sampling_cfg.get("max_frames_per_window", 4)),
        )
        self.roi_extractor = ROIExtractor(
            padding_ratio=float(roi_cfg.get("padding_ratio", 0.2)),
            detection_backend=str(roi_cfg.get("detection_backend", "vision")),
        )

        model_path = str(model_cfg.get("local_path", "")).strip()
        LOGGER.info("  [5/6] 初始化模型配置 (QwenOmniModel: %s) ...", model_path)
        self.model = QwenOmniModel(
            model_path=str(Path(model_path).expanduser()),
            torch_dtype=str(infer_cfg.get("torch_dtype", "bfloat16")),
            attn_implementation=str(infer_cfg.get("attn_implementation", "sdpa")),
            max_new_tokens=int(infer_cfg.get("max_new_tokens", 512)),
        )
        LOGGER.info("  [6/6] 初始化状态追踪器 (EmotionStateTracker) ...")
        self.tracker = EmotionStateTracker(
            max_history=int(understanding_cfg.get("max_history", 20))
        )

        self._use_audio_in_video: bool = bool(infer_cfg.get("use_audio_in_video", True))
        self._roi_enabled: bool = bool(roi_cfg.get("enabled", True))
        self._inference_interval: float = float(perf_cfg.get("inference_interval_seconds", 0.1))
        self._latency_budget_single: float = float(
            perf_cfg.get("latency_budget_ms", {}).get("single_person", 400)
        )
        self._latency_budget_multi: float = float(
            perf_cfg.get("latency_budget_ms", {}).get("multi_person", 500)
        )

        self._running = False
        self._state_lock = Lock()
        self._latest_states: dict[str, EmotionResult] = {}
        self._loop_thread: Thread | None = None

        self._frame_lock = Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_ts: float = 0.0

        self._metrics_lock = Lock()
        self._metrics_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._person_count: int = 0
        self._inference_count: int = 0

    def start(self) -> None:
        """加载模型并启动采集与推理循环。"""
        with self._state_lock:
            if self._running:
                return
            self._running = True

        LOGGER.info("正在加载模型（首次可能需要下载权重）...")
        self.model.load()

        LOGGER.info("正在启动视频采集 ...")
        self.video_capture.set_frame_callback(self._on_frame)
        self.audio_capture.set_audio_callback(self._on_audio)

        try:
            self.video_capture.start()
            LOGGER.info("视频采集已启动")
        except Exception:
            LOGGER.exception("启动视频采集失败，进入降级模式（仅音频）")

        try:
            self.audio_capture.start()
            LOGGER.info("音频采集已启动")
        except Exception:
            LOGGER.exception("启动音频采集失败，进入降级模式（仅视频）")

        self._loop_thread = Thread(
            target=self._inference_loop,
            name="realtime-inference",
            daemon=True,
        )
        self._loop_thread.start()
        LOGGER.info("推理循环已启动")
        LOGGER.info("RealtimePipeline 已就绪")

    def stop(self) -> None:
        """停止推理循环并释放采集资源。"""
        with self._state_lock:
            if not self._running:
                return
            self._running = False

        loop_thread = self._loop_thread
        if loop_thread is not None:
            loop_thread.join(timeout=max(1.0, self._inference_interval * 2))
            self._loop_thread = None

        try:
            self.video_capture.stop()
        except Exception:
            LOGGER.exception("停止视频采集失败")

        try:
            self.audio_capture.stop()
        except Exception:
            LOGGER.exception("停止音频采集失败")

        LOGGER.info("RealtimePipeline 已停止")

    def _inference_loop(self) -> None:
        """持续从缓冲区拉取窗口并执行推理。"""
        while self._is_running():
            window = self.stream_buffer.get_window()
            if window is None:
                time.sleep(self._inference_interval)
                continue

            loop_start = perf_counter()
            try:
                sampled_frames = self.frame_sampler.sample(window.frames)
                if not sampled_frames:
                    continue

                audio = window.get_audio_array()
                audio_input = audio if audio.size > 0 else None

                has_audio = audio_input is not None
                person_frames = self._prepare_person_inputs(sampled_frames)
                for person_index, frames in enumerate(person_frames):
                    conversation = build_conversation(
                        system_prompt=build_system_prompt(),
                        task_prompt=build_single_person_prompt(),
                        frames=[Image.fromarray(frame) for frame in frames],
                        audio=audio_input,
                    )
                    response = self.model.infer(
                        conversation=conversation,
                        use_audio_in_video=self._use_audio_in_video and has_audio,
                    )
                    result = parse_emotion_response(response)
                    if result is None:
                        LOGGER.warning("模型输出无法解析，已跳过当前人物结果")
                        continue
                    normalized = EmotionResult(
                        person_id=f"person_{person_index}",
                        primary_emotion=result.primary_emotion,
                        emotion_intensity=result.emotion_intensity,
                        secondary_emotion=result.secondary_emotion,
                        confidence=result.confidence,
                        description=result.description,
                    )
                    self.tracker.update(normalized, timestamp=window.end_ts)
                    with self._state_lock:
                        self._latest_states[normalized.person_id] = normalized

                elapsed_ms = (perf_counter() - loop_start) * 1000.0
                person_count = len(person_frames)
                budget_ms = (
                    self._latency_budget_multi if person_count > 1 else self._latency_budget_single
                )

                with self._metrics_lock:
                    self._person_count = person_count
                    self._inference_count += 1
                    self._metrics_history.append({
                        "latency_ms": elapsed_ms,
                        "person_count": person_count,
                        "budget_ms": budget_ms,
                        "within_budget": elapsed_ms <= budget_ms,
                        "timestamp": time.time(),
                    })

                LOGGER.info(
                    "窗口推理完成: people=%d latency=%.1fms budget=%.1fms status=%s",
                    person_count,
                    elapsed_ms,
                    budget_ms,
                    "ok" if elapsed_ms <= budget_ms else "exceeded",
                )
            except Exception:
                LOGGER.exception("推理循环异常，跳过当前窗口")

    def get_current_state(self) -> dict[str, dict[str, Any]]:
        """返回当前已知人物情绪状态快照。"""
        with self._state_lock:
            return {
                person_id: {
                    "person_id": item.person_id,
                    "primary_emotion": item.primary_emotion,
                    "emotion_intensity": item.emotion_intensity,
                    "secondary_emotion": item.secondary_emotion,
                    "confidence": item.confidence,
                    "description": item.description,
                }
                for person_id, item in self._latest_states.items()
            }

    def get_latest_frame(self) -> tuple[np.ndarray | None, float]:
        """返回最新的摄像头原始帧及其时间戳。"""
        with self._frame_lock:
            return self._latest_frame, self._latest_frame_ts

    def get_performance_metrics(self) -> dict[str, Any]:
        """返回推理性能指标快照。"""
        with self._metrics_lock:
            history = list(self._metrics_history)
            person_count = self._person_count
            inference_count = self._inference_count

        if not history:
            return {
                "last_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "person_count": 0,
                "inference_count": 0,
                "within_budget": True,
            }

        latencies = [h["latency_ms"] for h in history]
        latest = history[-1]
        return {
            "last_latency_ms": round(latest["latency_ms"], 1),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "person_count": person_count,
            "inference_count": inference_count,
            "within_budget": latest["within_budget"],
        }

    def get_emotion_trends(self, window_count: int = 10) -> dict[str, list[dict[str, Any]]]:
        """返回所有已知人物的情绪趋势数据。"""
        with self._state_lock:
            person_ids = list(self._latest_states.keys())

        trends: dict[str, list[dict[str, Any]]] = {}
        for person_id in person_ids:
            results = self.tracker.get_trend(person_id, window_count=window_count)
            trends[person_id] = [
                {
                    "primary_emotion": r.primary_emotion,
                    "emotion_intensity": r.emotion_intensity,
                    "confidence": r.confidence,
                }
                for r in results
            ]
        return trends

    def _prepare_person_inputs(self, sampled_frames: list[Any]) -> list[list[Any]]:
        """
        根据 ROI 开关与检测结果准备逐人输入帧序列。

        单人场景返回整帧序列；多人场景返回每个人物对应的帧序列（当前使用最新帧 ROI）。
        """
        if not self._roi_enabled:
            return [sampled_frames]
        rois = self.roi_extractor.extract(sampled_frames[-1])
        if len(rois) <= 1:
            return [sampled_frames]
        return [[roi] for roi in rois]

    def _on_frame(self, frame: Any, timestamp: float) -> None:
        """视频回调：将帧压入缓冲区，同时缓存最新帧供可视化使用。"""
        try:
            self.stream_buffer.push_frame(frame=frame, timestamp=timestamp)
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_ts = timestamp
        except Exception:
            LOGGER.exception("写入视频帧到缓冲区失败")

    def _on_audio(self, chunk: Any, timestamp: float) -> None:
        """音频回调：将音频块压入缓冲区。"""
        try:
            self.stream_buffer.push_audio(chunk=chunk, timestamp=timestamp)
        except Exception:
            LOGGER.exception("写入音频块到缓冲区失败")

    def _is_running(self) -> bool:
        with self._state_lock:
            return self._running
