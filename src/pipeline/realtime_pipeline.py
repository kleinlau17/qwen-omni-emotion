"""端到端实时流水线 — 双线程流水线：预处理 ∥ 推理，完成即显示

架构::

    StreamBuffer → [PrepThread] → PrepQueue(FIFO) → [InferThread] → Dashboard
                     ~0.5s/窗口                        ~4.5s/推理     即时更新

- PrepThread 持续从 StreamBuffer 拉取窗口并预处理（帧采样/ROI/resize/构建 conversation），
  结果推入线程安全的 FIFO 队列。
- InferThread 从队列取出 1~batch_size 个 prepared item 执行推理，每完成一批立即更新状态。
- FIFO 保证时间轴顺序；预处理与推理并行消除了预处理等待。
"""
from __future__ import annotations

import logging
import queue
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
        audio_sample_rate = int(audio_cfg.get("sample_rate", 16000))
        audio_channels = int(audio_cfg.get("channels", 1))
        LOGGER.info("  [3/6] 初始化音频采集 (AudioCapture %dHz) ...", audio_sample_rate)
        self.audio_capture = AudioCapture(
            sample_rate=audio_sample_rate,
            channels=audio_channels,
        )
        self._batch_size: int = int(perf_cfg.get("batch_size", 1))

        buffer_max = max(
            int(stream_cfg.get("max_windows_in_buffer", 3)),
            self._batch_size + 2,
        )
        self.stream_buffer = StreamBuffer(
            window_duration=float(stream_cfg.get("window_duration_seconds", 1.5)),
            max_windows=buffer_max,
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
            max_new_tokens=int(infer_cfg.get("max_new_tokens", 128)),
            min_pixels=int(infer_cfg.get("min_pixels", 128 * 28 * 28)),
            max_pixels=int(infer_cfg.get("max_pixels", 256 * 28 * 28)),
        )
        LOGGER.info("  [6/6] 初始化状态追踪器 (EmotionStateTracker) ...")
        self.tracker = EmotionStateTracker(
            max_history=int(understanding_cfg.get("max_history", 20))
        )

        self._use_audio_in_video: bool = bool(infer_cfg.get("use_audio_in_video", True))
        self._roi_enabled: bool = bool(roi_cfg.get("enabled", True))

        infer_res_raw = preprocess_cfg.get("inference_resolution", [480, 270])
        self._inference_resolution: tuple[int, int] = (
            int(infer_res_raw[0]),
            int(infer_res_raw[1]),
        )

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

        self._prep_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=self._batch_size * 3,
        )
        self._prep_thread: Thread | None = None
        self._infer_thread: Thread | None = None

        self._frame_lock = Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_ts: float = 0.0

        self._metrics_lock = Lock()
        self._metrics_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._person_count: int = 0
        self._inference_count: int = 0

        self._history_lock = Lock()
        self._history: deque[dict[str, Any]] = deque(
            maxlen=int(understanding_cfg.get("history_max_items", 32))
        )
        self._next_history_id: int = 1

        self._audio_sample_rate: int = audio_sample_rate
        self._audio_channels: int = audio_channels

    def start(self) -> None:
        """加载模型并启动采集与双线程流水线。"""
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

        self._prep_thread = Thread(
            target=self._prep_loop,
            name="realtime-prep",
            daemon=True,
        )
        self._infer_thread = Thread(
            target=self._infer_loop,
            name="realtime-infer",
            daemon=True,
        )
        self._prep_thread.start()
        self._infer_thread.start()
        LOGGER.info(
            "流水线已启动 (prep→queue→infer, batch_size=%d)",
            self._batch_size,
        )
        LOGGER.info("RealtimePipeline 已就绪")

    def stop(self) -> None:
        """停止流水线并释放采集资源。"""
        with self._state_lock:
            if not self._running:
                return
            self._running = False

        for thread in (self._prep_thread, self._infer_thread):
            if thread is not None:
                thread.join(timeout=10.0)
        self._prep_thread = None
        self._infer_thread = None

        try:
            self.video_capture.stop()
        except Exception:
            LOGGER.exception("停止视频采集失败")

        try:
            self.audio_capture.stop()
        except Exception:
            LOGGER.exception("停止音频采集失败")

        LOGGER.info("RealtimePipeline 已停止")

    # ── 流水线阶段 1: 预处理线程 ──────────────────────────────────

    def _prep_loop(self) -> None:
        """持续从 StreamBuffer 拉取窗口，预处理后推入 _prep_queue。

        处理速度远快于推理 (~0.5s vs ~4.5s)，因此队列中通常有
        若干已就绪的 item 等待 InferThread 消费。
        """
        system_prompt = build_system_prompt()
        task_prompt = build_single_person_prompt()
        target_w, target_h = self._inference_resolution
        window_serial: int = 0

        while self._is_running():
            window = self.stream_buffer.get_window()
            if window is None:
                time.sleep(self._inference_interval)
                continue

            try:
                sampled_frames = self.frame_sampler.sample(window.frames)
                if not sampled_frames:
                    continue

                audio = window.get_audio_array()
                audio_input = audio if audio.size > 0 else None
                if audio_input is not None:
                    try:
                        duration_seconds = float(audio_input.shape[0]) / max(
                            float(self._audio_sample_rate), 1.0
                        )
                    except Exception:
                        duration_seconds = -1.0
                    LOGGER.info(
                        "预处理窗口 #%d: audio_samples=%d, sample_rate=%d, duration=%.3fs",
                        window_serial,
                        int(audio_input.shape[0]),
                        int(self._audio_sample_rate),
                        duration_seconds,
                    )
                person_frames = self._prepare_person_inputs(sampled_frames)

                for person_idx, frames in enumerate(person_frames):
                    resized = [
                        Image.fromarray(f).resize(
                            (target_w, target_h), Image.LANCZOS,
                        )
                        for f in frames
                    ]
                    conversation = build_conversation(
                        system_prompt=system_prompt,
                        task_prompt=task_prompt,
                        frames=resized,
                        audio=audio_input,
                    )
                    item: dict[str, Any] = {
                        "conversation": conversation,
                        "window_serial": window_serial,
                        "person_idx": person_idx,
                        "end_ts": window.end_ts,
                        "audio": audio_input,
                        "prep_done_ts": perf_counter(),
                        "frames": frames,
                    }
                    while self._is_running():
                        try:
                            self._prep_queue.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue

                window_serial += 1
            except Exception:
                LOGGER.exception("预处理异常，跳过窗口 #%d", window_serial)

    # ── 流水线阶段 2: 推理线程 ────────────────────────────────────

    def _infer_loop(self) -> None:
        """从 _prep_queue 取出 prepared item 执行推理，完成即更新状态。

        每轮从队列中取 1~batch_size 个 item，调用 batch_infer，
        结果逐条更新 tracker 和仪表盘状态。FIFO 取出保证时间轴顺序。
        """
        while self._is_running():
            batch_items: list[dict[str, Any]] = []

            try:
                first = self._prep_queue.get(timeout=1.0)
                batch_items.append(first)
            except queue.Empty:
                continue

            for _ in range(self._batch_size - 1):
                try:
                    batch_items.append(self._prep_queue.get_nowait())
                except queue.Empty:
                    break

            infer_start = perf_counter()
            try:
                has_audio = any(item["audio"] is not None for item in batch_items)
                conversations = [item["conversation"] for item in batch_items]

                responses = self.model.batch_infer(
                    conversations=conversations,
                    use_audio_in_video=self._use_audio_in_video and has_audio,
                )

                for item, response in zip(batch_items, responses):
                    LOGGER.info(
                        "模型原始输出 (完整):\n%s",
                        response if isinstance(response, str) else repr(response),
                    )
                    result = parse_emotion_response(response)
                    if result is None:
                        LOGGER.warning(
                            "模型输出无法解析，已跳过 (window=#%d person=%d)",
                            item["window_serial"], item["person_idx"],
                        )
                        continue
                    LOGGER.info(
                        "解析结果: primary_emotion=%s secondary_emotion=%s",
                        result.primary_emotion,
                        result.secondary_emotion,
                    )
                    normalized = EmotionResult(
                        person_id=f"person_{item['person_idx']}",
                        primary_emotion=result.primary_emotion,
                        secondary_emotion=result.secondary_emotion,
                    )
                    self.tracker.update(normalized, timestamp=item["end_ts"])
                    with self._state_lock:
                        self._latest_states[normalized.person_id] = normalized
                    self._append_history(
                        result=normalized,
                        end_ts=item["end_ts"],
                        audio=item["audio"],
                        frames=item.get("frames", []),
                    )

                infer_ms = (perf_counter() - infer_start) * 1000.0
                total_items = len(batch_items)
                e2e_ms = (perf_counter() - batch_items[0]["prep_done_ts"]) * 1000.0
                budget_ms = self._latency_budget_single

                with self._metrics_lock:
                    self._person_count = total_items
                    self._inference_count += 1
                    self._metrics_history.append({
                        "latency_ms": infer_ms,
                        "person_count": total_items,
                        "budget_ms": budget_ms,
                        "within_budget": infer_ms <= budget_ms,
                        "timestamp": time.time(),
                        "batch_items": total_items,
                        "e2e_ms": e2e_ms,
                    })

                LOGGER.info(
                    "推理完成: batch=%d infer=%.0fms e2e=%.0fms per_item=%.0fms",
                    total_items,
                    infer_ms,
                    e2e_ms,
                    infer_ms / max(total_items, 1),
                )
            except Exception:
                LOGGER.exception("推理异常，跳过当前批次")

    def _append_history(
        self,
        result: EmotionResult,
        end_ts: float,
        audio: Any | None,
        frames: list[Any],
    ) -> None:
        """将本次推理结果追加到内存历史，用于 Web 端回放多帧 + 音频。"""
        frame_list: list[Any] = []
        for f in frames:
            try:
                frame_list.append(np.asarray(f).copy())
            except Exception:
                continue

        audio_copy: Any | None = None
        if audio is not None:
            try:
                audio_copy = np.asarray(audio).copy()
            except Exception:
                audio_copy = None

        with self._history_lock:
            item_id = self._next_history_id
            self._next_history_id += 1
            self._history.append(
                {
                    "id": item_id,
                    "person_id": result.person_id,
                    "primary_emotion": result.primary_emotion,
                    "secondary_emotion": result.secondary_emotion,
                    "emotion_intensity": None,
                    "confidence": None,
                    "description": None,
                    "timestamp": end_ts,
                    "frames": frame_list,
                    "audio": audio_copy,
                }
            )

    def get_current_state(self) -> dict[str, dict[str, Any]]:
        """返回当前已知人物情绪状态快照。

        保留 emotion_intensity、confidence、description 字段（值为 None），
        供 Web 端保持结构但不刷新展示。
        """
        with self._state_lock:
            return {
                person_id: {
                    "person_id": item.person_id,
                    "primary_emotion": item.primary_emotion,
                    "secondary_emotion": item.secondary_emotion,
                    "emotion_intensity": None,
                    "confidence": None,
                    "description": None,
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
                "running": self._is_running(),
            }

        latencies = [h["latency_ms"] for h in history]
        latest = history[-1]
        return {
            "last_latency_ms": round(latest["latency_ms"], 1),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "person_count": person_count,
            "inference_count": inference_count,
            "within_budget": latest["within_budget"],
            "running": self._is_running(),
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
                    "emotion_intensity": None,
                    "confidence": None,
                }
                for r in results
            ]
        return trends

    def get_inference_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """返回最近若干次推理的概要历史（不包含原始媒体数据）。"""
        with self._history_lock:
            items = list(self._history)[-limit:]
        return [
            {
                "id": item["id"],
                "person_id": item["person_id"],
                "primary_emotion": item["primary_emotion"],
                "secondary_emotion": item["secondary_emotion"],
                "emotion_intensity": item["emotion_intensity"],
                "confidence": item["confidence"],
                "description": item["description"],
                "timestamp": item["timestamp"],
                "frame_count": len(item.get("frames") or []),
            }
            for item in items
        ]

    def get_history_media(
        self,
        item_id: int,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """根据历史条目 ID 获取对应的多帧和音频数组。"""
        with self._history_lock:
            for item in self._history:
                if item.get("id") == item_id:
                    frames = item.get("frames") or []
                    audio = item.get("audio")
                    frame_list: list[np.ndarray] = []
                    for f in frames:
                        try:
                            frame_list.append(np.asarray(f).copy())
                        except Exception:
                            continue
                    audio_arr = (
                        np.asarray(audio).copy()
                        if audio is not None
                        else None
                    )
                    return frame_list, audio_arr
        return [], None

    def get_audio_format(self) -> dict[str, int]:
        """返回音频格式信息，供 Web 端导出 WAV 使用。"""
        return {
            "sample_rate": self._audio_sample_rate,
            "channels": self._audio_channels,
        }

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
