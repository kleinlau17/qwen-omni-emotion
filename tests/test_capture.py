"""采集层测试"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.capture.audio_capture import AudioCapture
from src.capture.stream_buffer import InferenceWindow, StreamBuffer
from src.capture.video_capture import VideoCapture


def _has_pyobjc_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


HAS_AVFOUNDATION = _has_pyobjc_module("AVFoundation")

if HAS_AVFOUNDATION:
    import AVFoundation as AVF  # type: ignore

    HAS_VIDEO_DEVICE = AVF.AVCaptureDevice.defaultDeviceWithMediaType_(AVF.AVMediaTypeVideo) is not None
    HAS_AUDIO_DEVICE = AVF.AVCaptureDevice.defaultDeviceWithMediaType_(AVF.AVMediaTypeAudio) is not None
else:
    HAS_VIDEO_DEVICE = False
    HAS_AUDIO_DEVICE = False


def test_inference_window_concat_audio() -> None:
    """验证音频块拼接结果。"""
    window = InferenceWindow(
        audio_chunks=[
            (np.array([0.1, 0.2], dtype=np.float32), 1.0),
            (np.array([0.3, 0.4, 0.5], dtype=np.float32), 1.1),
        ]
    )

    audio = window.get_audio_array()
    np.testing.assert_allclose(audio, np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    assert audio.dtype == np.float32


def test_stream_buffer_push_and_get_window() -> None:
    """验证窗口填满后可非阻塞取出。"""
    buffer = StreamBuffer(window_duration=1.5, max_windows=3)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    audio = np.array([0.1, -0.2], dtype=np.float32)

    buffer.push_frame(frame, timestamp=10.0)
    buffer.push_audio(audio, timestamp=10.2)
    assert buffer.get_window() is None

    buffer.push_frame(frame, timestamp=11.6)
    ready = buffer.get_window()

    assert ready is not None
    assert ready.start_ts == pytest.approx(10.0)
    assert ready.end_ts == pytest.approx(11.6)
    assert len(ready.frames) == 2
    assert len(ready.audio_chunks) == 1


def test_stream_buffer_drop_oldest_when_exceeds_max_windows() -> None:
    """验证超出 max_windows 时丢弃最旧窗口。"""
    buffer = StreamBuffer(window_duration=1.0, max_windows=2)

    for idx in range(3):
        ts_base = float(idx * 2)
        frame = np.full((2, 2, 3), fill_value=idx, dtype=np.uint8)
        buffer.push_frame(frame, timestamp=ts_base)
        buffer.push_frame(frame, timestamp=ts_base + 1.1)

    win1 = buffer.get_window()
    win2 = buffer.get_window()
    win3 = buffer.get_window()

    assert win1 is not None
    assert win2 is not None
    assert win3 is None
    assert int(win1.frames[0][0][0, 0, 0]) == 1
    assert int(win2.frames[0][0][0, 0, 0]) == 2


@pytest.mark.skipif(
    not HAS_AVFOUNDATION or not HAS_VIDEO_DEVICE,
    reason="无 AVFoundation 或无可用摄像头，跳过硬件测试",
)
def test_video_capture_start_stop() -> None:
    """验证视频采集启动/停止状态切换。"""
    capture = VideoCapture()
    assert capture.is_running is False
    capture.start()
    assert capture.is_running is True
    capture.stop()
    assert capture.is_running is False


@pytest.mark.skipif(
    not HAS_AVFOUNDATION or not HAS_AUDIO_DEVICE,
    reason="无 AVFoundation 或无可用麦克风，跳过硬件测试",
)
def test_audio_capture_start_stop() -> None:
    """验证音频采集启动/停止状态切换。"""
    capture = AudioCapture()
    assert capture.is_running is False
    capture.start()
    assert capture.is_running is True
    capture.stop()
    assert capture.is_running is False
