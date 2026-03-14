"""管道编排测试"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import yaml

import main as app_main
from src.capture.stream_buffer import InferenceWindow
from src.pipeline import realtime_pipeline
from src.pipeline.realtime_pipeline import RealtimePipeline


class _FakeVideoCapture:
    def __init__(self, resolution: tuple[int, int], fps: int) -> None:
        self.resolution = resolution
        self.fps = fps
        self.started = False
        self.stopped = False
        self.callback = None

    def set_frame_callback(self, callback: Any) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FakeAudioCapture:
    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.started = False
        self.stopped = False
        self.callback = None

    def set_audio_callback(self, callback: Any) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FakeStreamBuffer:
    def __init__(self, window_duration: float, max_windows: int) -> None:
        self.window_duration = window_duration
        self.max_windows = max_windows
        self._windows: list[InferenceWindow] = [
            InferenceWindow(
                frames=[
                    (np.zeros((8, 8, 3), dtype=np.uint8), 10.0),
                    (np.ones((8, 8, 3), dtype=np.uint8), 10.2),
                ],
                audio_chunks=[(np.zeros(1600, dtype=np.float32), 10.1)],
                start_ts=10.0,
                end_ts=11.6,
            )
        ]

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        del frame, timestamp

    def push_audio(self, chunk: np.ndarray, timestamp: float) -> None:
        del chunk, timestamp

    def get_window(self) -> InferenceWindow | None:
        if self._windows:
            return self._windows.pop(0)
        return None


class _FakeFrameSampler:
    def __init__(self, strategy: str, max_frames: int) -> None:
        self.strategy = strategy
        self.max_frames = max_frames

    def sample(self, frames: list[tuple[np.ndarray, float]]) -> list[np.ndarray]:
        return [item[0] for item in frames[: self.max_frames]]


class _FakeROIExtractor:
    def __init__(self, padding_ratio: float, detection_backend: str) -> None:
        self.padding_ratio = padding_ratio
        self.detection_backend = detection_backend

    def extract(self, frame: np.ndarray) -> list[np.ndarray]:
        return [frame]


class _FakeModel:
    def __init__(
        self,
        model_path: str,
        torch_dtype: str,
        attn_implementation: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        min_pixels: int,
        max_pixels: int,
        torch_compile: bool,
    ) -> None:
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.torch_compile = torch_compile
        self.load_called = False
        self.batch_infer_called = 0

    def load(self) -> None:
        self.load_called = True

    def batch_infer(
        self,
        conversations: list[list[dict[str, Any]]],
        use_audio_in_video: bool = True,
    ) -> list[str]:
        del use_audio_in_video
        self.batch_infer_called += 1
        return [
            '{"action":"neutral.surprise.quick.low","reason":"检测到突发刺激信号，先给出轻微惊讶回应。"}'
            for _ in conversations
        ]


class _FakeTracker:
    def __init__(self, max_history: int) -> None:
        self.max_history = max_history
        self.updated: list[tuple[Any, float]] = []

    def update(self, result: Any, timestamp: float) -> None:
        self.updated.append((result, timestamp))


def _build_test_config() -> dict[str, Any]:
    return {
        "system": {"log_level": "INFO"},
        "performance": {
            "latency_budget_ms": {"single_person": 400, "multi_person": 500},
            "inference_interval_seconds": 0.01,
        },
        "model": {"local_path": "~/.cache/modelscope/Qwen2.5-Omni-3B"},
        "inference": {
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "use_audio_in_video": True,
        },
        "capture": {
            "video": {"resolution": [640, 480], "fps": 30},
            "audio": {"sample_rate": 16000, "channels": 1},
        },
        "preprocessing": {
            "frame_sampling": {"strategy": "uniform", "max_frames_per_window": 4},
            "roi": {"enabled": True, "detection_backend": "vision", "padding_ratio": 0.2},
        },
        "stream_buffer": {"window_duration_seconds": 1.5, "max_windows_in_buffer": 3},
    }


def _patch_pipeline_dependencies(monkeypatch: Any) -> None:
    monkeypatch.setattr(realtime_pipeline, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(realtime_pipeline, "AudioCapture", _FakeAudioCapture)
    monkeypatch.setattr(realtime_pipeline, "StreamBuffer", _FakeStreamBuffer)
    monkeypatch.setattr(realtime_pipeline, "FrameSampler", _FakeFrameSampler)
    monkeypatch.setattr(realtime_pipeline, "ROIExtractor", _FakeROIExtractor)
    monkeypatch.setattr(realtime_pipeline, "QwenOmniModel", _FakeModel)
    monkeypatch.setattr(realtime_pipeline, "EmotionStateTracker", _FakeTracker)


def test_pipeline_start_stop_lifecycle(monkeypatch: Any) -> None:
    """验证 pipeline start/stop 生命周期行为。"""
    _patch_pipeline_dependencies(monkeypatch)
    pipeline = RealtimePipeline(config=_build_test_config())

    pipeline.start()
    assert pipeline.model.load_called is True
    assert pipeline.video_capture.started is True
    assert pipeline.audio_capture.started is True

    pipeline.stop()
    assert pipeline.video_capture.stopped is True
    assert pipeline.audio_capture.stopped is True


def test_pipeline_orchestration_updates_state(monkeypatch: Any) -> None:
    """验证推理编排链路可写入当前状态。"""
    _patch_pipeline_dependencies(monkeypatch)
    pipeline = RealtimePipeline(config=_build_test_config())

    pipeline.start()
    time.sleep(0.05)
    pipeline.stop()

    assert pipeline.model.batch_infer_called >= 1
    assert len(pipeline.tracker.updated) >= 1
    latest_result, _ = pipeline.tracker.updated[-1]
    assert latest_result.action_confidence == 0.5
    assert latest_result.hold_seconds == 1.0
    current_state = pipeline.get_current_state()
    assert "person_0" in current_state
    assert current_state["person_0"]["detected_emotion"] == "neutral"
    assert current_state["person_0"]["description"] == "检测到突发刺激信号，先给出轻微惊讶回应。"


def test_load_merged_config(tmp_path: Any) -> None:
    """验证 default/model/pipeline 三份配置可按顺序合并。"""
    default_cfg = {
        "system": {"log_level": "INFO"},
        "performance": {"latency_budget_ms": {"single_person": 400}},
    }
    model_cfg = {
        "model": {"local_path": "/tmp/model"},
        "inference": {"torch_dtype": "bfloat16"},
    }
    pipeline_cfg = {"performance": {"inference_interval_seconds": 0.2}, "capture": {"video": {"fps": 25}}}

    (tmp_path / "default.yaml").write_text(yaml.safe_dump(default_cfg), encoding="utf-8")
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(model_cfg), encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(pipeline_cfg), encoding="utf-8")

    merged = app_main.load_merged_config(tmp_path)
    assert merged["system"]["log_level"] == "INFO"
    assert merged["model"]["local_path"] == "/tmp/model"
    assert merged["inference"]["torch_dtype"] == "bfloat16"
    assert merged["performance"]["latency_budget_ms"]["single_person"] == 400
    assert merged["performance"]["inference_interval_seconds"] == 0.2
    assert merged["capture"]["video"]["fps"] == 25
