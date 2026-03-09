"""实时音频流采集"""
from __future__ import annotations

import ctypes
import logging
import time
from collections.abc import Callable
from threading import Lock
from typing import Final

import numpy as np

try:
    import AVFoundation
    import CoreMedia
    import objc
    from dispatch import DISPATCH_QUEUE_SERIAL, dispatch_queue_create
    from Foundation import NSObject

    PYOBJC_AVAILABLE = True
except ImportError:  # pragma: no cover - 非 macOS/PyObjC 环境
    AVFoundation = None
    CoreMedia = None
    objc = None
    NSObject = object
    DISPATCH_QUEUE_SERIAL = None
    dispatch_queue_create = None
    PYOBJC_AVAILABLE = False


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
AudioCallback = Callable[[np.ndarray, float], None]


def _extract_pcm_bytes(block_buffer) -> bytes:  # noqa: ANN001
    """兼容不同 PyObjC 绑定返回值，提取 CMBlockBuffer 内的 PCM 字节流。"""
    total_length = int(CoreMedia.CMBlockBufferGetDataLength(block_buffer))
    if total_length <= 0:
        return b""

    copy_fn = getattr(CoreMedia, "CMBlockBufferCopyDataBytes", None)
    if copy_fn is not None:
        copied = copy_fn(block_buffer, 0, total_length, None)
        if isinstance(copied, (bytes, bytearray, memoryview)):
            return bytes(copied)
        if isinstance(copied, tuple):
            for item in copied:
                if isinstance(item, (bytes, bytearray, memoryview)):
                    return bytes(item)

    pointer_result = CoreMedia.CMBlockBufferGetDataPointer(block_buffer, 0, None, None, None)
    if isinstance(pointer_result, tuple):
        pointer_value: int | None = None
        bytes_value: bytes | None = None
        for item in pointer_result:
            if isinstance(item, int):
                pointer_value = item
            elif isinstance(item, (bytes, bytearray, memoryview)):
                bytes_value = bytes(item)

        if bytes_value is not None:
            return bytes_value[:total_length]
        if pointer_value is not None:
            return ctypes.string_at(pointer_value, total_length)

    if isinstance(pointer_result, int):
        return ctypes.string_at(pointer_result, total_length)

    return b""


class AudioCaptureDelegate(NSObject):
    """AVCaptureAudioDataOutput 回调代理。"""

    def init(self):  # type: ignore[override]
        """初始化 delegate 内部状态。"""
        self = objc.super(AudioCaptureDelegate, self).init()
        if self is None:
            return None
        self._audio_callback: AudioCallback | None = None
        self._lock = Lock()
        return self

    def set_callback(self, callback: AudioCallback | None) -> None:
        """设置音频回调。"""
        with self._lock:
            self._audio_callback = callback

    def captureOutput_didOutputSampleBuffer_fromConnection_(  # noqa: N802
        self,
        output,  # noqa: ANN001
        sample_buffer,  # noqa: ANN001
        connection,  # noqa: ANN001
    ) -> None:
        """AVFoundation 音频回调：PCM16 转 float32。"""
        del output, connection
        with self._lock:
            callback = self._audio_callback
        if callback is None:
            return

        block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
        if block_buffer is None:
            return

        try:
            pcm_bytes = _extract_pcm_bytes(block_buffer)
            if not pcm_bytes:
                return

            audio_pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            if audio_pcm16.size == 0:
                return
            audio_float32 = (audio_pcm16.astype(np.float32) / 32768.0).copy()
            callback(audio_float32, time.time())
        except Exception:
            LOGGER.exception("音频块转换失败")


class AudioCapture:
    """麦克风实时采集封装。"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        """
        初始化音频采集器（不自动启动）。

        Args:
            sample_rate: 采样率。
            channels: 通道数。
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_callback: AudioCallback | None = None

        self._session = None
        self._delegate = None
        self._queue = None
        self._is_running = False
        self._lock = Lock()

    @property
    def is_running(self) -> bool:
        """返回采集是否正在运行。"""
        with self._lock:
            return self._is_running

    def set_audio_callback(self, callback: AudioCallback) -> None:
        """设置音频回调函数。"""
        with self._lock:
            self._audio_callback = callback
            delegate = self._delegate

        if delegate is not None:
            delegate.set_callback(callback)

    def start(self) -> None:
        """启动麦克风采集会话。"""
        if not PYOBJC_AVAILABLE:
            raise RuntimeError("PyObjC/AVFoundation 不可用，无法启动音频采集")

        with self._lock:
            if self._is_running:
                return

        session = AVFoundation.AVCaptureSession.alloc().init()

        device = AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(
            AVFoundation.AVMediaTypeAudio
        )
        if device is None:
            raise RuntimeError("未找到可用麦克风设备")

        input_device, input_error = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(
            device, None
        )
        if input_error is not None:
            raise RuntimeError(f"创建音频输入失败: {input_error}")
        if not session.canAddInput_(input_device):
            raise RuntimeError("音频输入无法添加到采集会话")
        session.addInput_(input_device)

        output = AVFoundation.AVCaptureAudioDataOutput.alloc().init()
        output.setAudioSettings_(
            {
                "AVSampleRateKey": self._sample_rate,
                "AVNumberOfChannelsKey": self._channels,
                "AVLinearPCMBitDepthKey": 16,
                "AVLinearPCMIsFloatKey": False,
                "AVFormatIDKey": 1819304813,  # kAudioFormatLinearPCM
            }
        )
        if not session.canAddOutput_(output):
            raise RuntimeError("音频输出无法添加到采集会话")
        session.addOutput_(output)

        delegate = AudioCaptureDelegate.alloc().init()
        if delegate is None:
            raise RuntimeError("初始化 AudioCaptureDelegate 失败")

        with self._lock:
            callback = self._audio_callback
        delegate.set_callback(callback)

        queue = dispatch_queue_create(b"omni3b.audio.capture.queue", DISPATCH_QUEUE_SERIAL)
        output.setSampleBufferDelegate_queue_(delegate, queue)

        session.startRunning()

        with self._lock:
            self._session = session
            self._delegate = delegate
            self._queue = queue
            self._is_running = True

    def stop(self) -> None:
        """停止麦克风采集会话。"""
        with self._lock:
            session = self._session
            if not self._is_running or session is None:
                return

        try:
            session.stopRunning()
        except Exception:
            LOGGER.exception("停止音频采集会话失败")
        finally:
            with self._lock:
                self._session = None
                self._delegate = None
                self._queue = None
                self._is_running = False
