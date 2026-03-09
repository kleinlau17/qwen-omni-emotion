"""AVFoundation 1080p/30fps 视频流采集"""
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
    try:
        import CoreVideo
    except ImportError:  # pragma: no cover - 部分 PyObjC 版本通过 Quartz 暴露 CoreVideo API
        import Quartz as CoreVideo
    import objc
    from dispatch import DISPATCH_QUEUE_SERIAL, dispatch_queue_create
    from Foundation import NSObject

    PYOBJC_AVAILABLE = True
except ImportError:  # pragma: no cover - 非 macOS/PyObjC 环境
    AVFoundation = None
    CoreMedia = None
    CoreVideo = None
    objc = None
    NSObject = object
    DISPATCH_QUEUE_SERIAL = None
    dispatch_queue_create = None
    PYOBJC_AVAILABLE = False


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
FrameCallback = Callable[[np.ndarray, float], None]
BGRA_PIXEL_FORMAT: Final[int] = 1111970369  # kCVPixelFormatType_32BGRA

# PyObjC 的 CVPixelBufferGetBaseAddress 在某些版本返回 objc.varlist 而非 int，
# 导致无法获取原始指针地址。通过 ctypes 直接调用 C 函数彻底绕过此问题。
_cv_lib = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/CoreVideo.framework/CoreVideo"
)
_cv_get_base_address = _cv_lib.CVPixelBufferGetBaseAddress
_cv_get_base_address.restype = ctypes.c_void_p
_cv_get_base_address.argtypes = [ctypes.c_void_p]


_AVAuthorizationStatusNotDetermined = 0
_AVAuthorizationStatusRestricted = 1
_AVAuthorizationStatusDenied = 2
_AVAuthorizationStatusAuthorized = 3


def _check_camera_authorization() -> None:
    """检查并请求摄像头访问权限，权限被拒绝时抛出异常。"""
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeVideo
    )
    if status == _AVAuthorizationStatusAuthorized:
        return

    if status == _AVAuthorizationStatusDenied:
        raise RuntimeError(
            "摄像头权限被拒绝，请前往 系统设置 → 隐私与安全性 → 摄像头 中授权当前终端应用"
        )

    if status == _AVAuthorizationStatusRestricted:
        raise RuntimeError("摄像头访问受限（可能由设备管理策略限制）")

    if status == _AVAuthorizationStatusNotDetermined:
        import threading

        granted_event = threading.Event()
        granted_result: list[bool] = [False]

        def _on_response(granted: bool) -> None:
            granted_result[0] = granted
            granted_event.set()

        LOGGER.info("首次使用摄像头，正在请求系统授权（请在弹窗中允许）...")
        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeVideo, _on_response
        )
        granted_event.wait(timeout=60)

        if not granted_result[0]:
            raise RuntimeError("用户拒绝了摄像头访问权限")
        LOGGER.info("摄像头权限已获取")


def _get_session_preset(resolution: tuple[int, int]) -> str | None:
    """按分辨率选择 AVFoundation 预设。"""
    if not PYOBJC_AVAILABLE:
        return None
    preset_map: dict[tuple[int, int], str] = {
        (640, 480): AVFoundation.AVCaptureSessionPreset640x480,
        (1280, 720): AVFoundation.AVCaptureSessionPreset1280x720,
        (1920, 1080): AVFoundation.AVCaptureSessionPreset1920x1080,
    }
    return preset_map.get(resolution)


class VideoCaptureDelegate(NSObject):
    """AVCaptureVideoDataOutput 回调代理。"""

    def init(self):  # type: ignore[override]
        """初始化 delegate 内部状态。"""
        self = objc.super(VideoCaptureDelegate, self).init()
        if self is None:
            return None
        self._frame_callback: FrameCallback | None = None
        self._lock = Lock()
        self._frame_count: int = 0
        return self

    def set_callback(self, callback: FrameCallback | None) -> None:
        """设置帧回调。"""
        with self._lock:
            self._frame_callback = callback

    def captureOutput_didOutputSampleBuffer_fromConnection_(  # noqa: N802
        self,
        output,  # noqa: ANN001
        sample_buffer,  # noqa: ANN001
        connection,  # noqa: ANN001
    ) -> None:
        """AVFoundation 帧回调：BGRA 像素缓冲转 RGB ndarray。"""
        del output, connection

        self._frame_count += 1
        if self._frame_count <= 3:
            LOGGER.info("delegate 帧回调 #%d 已触发", self._frame_count)

        with self._lock:
            callback = self._frame_callback
        if callback is None:
            if self._frame_count <= 3:
                LOGGER.warning("帧回调 #%d: callback 为 None，跳过", self._frame_count)
            return

        image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
        if image_buffer is None:
            return

        CoreVideo.CVPixelBufferLockBaseAddress(image_buffer, 0)
        try:
            width = int(CoreVideo.CVPixelBufferGetWidth(image_buffer))
            height = int(CoreVideo.CVPixelBufferGetHeight(image_buffer))
            bytes_per_row = int(CoreVideo.CVPixelBufferGetBytesPerRow(image_buffer))

            if self._frame_count <= 3:
                fmt = CoreVideo.CVPixelBufferGetPixelFormatType(image_buffer)
                LOGGER.info(
                    "帧 #%d 像素信息: %dx%d bpr=%d fmt=%d",
                    self._frame_count, width, height, bytes_per_row, fmt,
                )

            buffer_ptr = objc.pyobjc_id(image_buffer)
            pointer_value = _cv_get_base_address(buffer_ptr)
            if not pointer_value:
                LOGGER.warning("base_address 为 NULL，跳过帧 #%d", self._frame_count)
                return

            buffer_size = bytes_per_row * height
            raw_bytes = ctypes.string_at(pointer_value, buffer_size)
            frame_bgra = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                (height, bytes_per_row // 4, 4)
            )[:, :width, :]
            frame_rgb = frame_bgra[:, :, [2, 1, 0]].copy()

            if self._frame_count == 1:
                LOGGER.info("首帧转换成功 (%dx%d RGB)", width, height)
            elif self._frame_count % 300 == 0:
                LOGGER.debug("已接收 %d 帧", self._frame_count)

            callback(frame_rgb, time.time())
        except Exception:
            LOGGER.exception("视频帧转换失败 (帧 #%d)", self._frame_count)
        finally:
            CoreVideo.CVPixelBufferUnlockBaseAddress(image_buffer, 0)


class VideoCapture:
    """摄像头实时采集封装。"""

    def __init__(self, resolution: tuple[int, int] = (1920, 1080), fps: int = 30) -> None:
        """
        初始化视频采集器（不自动启动）。

        Args:
            resolution: 目标分辨率。
            fps: 目标帧率。
        """
        self._resolution = resolution
        self._fps = fps
        self._frame_callback: FrameCallback | None = None

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

    def set_frame_callback(self, callback: FrameCallback) -> None:
        """设置帧回调函数。"""
        with self._lock:
            self._frame_callback = callback
            delegate = self._delegate

        if delegate is not None:
            delegate.set_callback(callback)

    def start(self) -> None:
        """启动摄像头采集会话。"""
        if not PYOBJC_AVAILABLE:
            raise RuntimeError("PyObjC/AVFoundation 不可用，无法启动视频采集")

        with self._lock:
            if self._is_running:
                return

        _check_camera_authorization()

        device = AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
        if device is None:
            raise RuntimeError("未找到可用摄像头设备")
        LOGGER.info("使用摄像头设备: %s", device.localizedName())

        input_device, input_error = (
            AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        )
        if input_error is not None:
            raise RuntimeError(f"创建视频输入失败: {input_error}")

        session = AVFoundation.AVCaptureSession.alloc().init()
        session.beginConfiguration()

        if not session.canAddInput_(input_device):
            raise RuntimeError("视频输入无法添加到采集会话")
        session.addInput_(input_device)

        output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_({"PixelFormatType": BGRA_PIXEL_FORMAT})
        if not session.canAddOutput_(output):
            raise RuntimeError("视频输出无法添加到采集会话")
        session.addOutput_(output)

        preset = _get_session_preset(self._resolution)
        if preset is not None and session.canSetSessionPreset_(preset):
            session.setSessionPreset_(preset)
            LOGGER.info("Session 预设: %s", self._resolution)
        else:
            LOGGER.warning("摄像头不支持预设分辨率 %s，使用设备默认设置", self._resolution)

        session.commitConfiguration()

        delegate = VideoCaptureDelegate.alloc().init()
        if delegate is None:
            raise RuntimeError("初始化 VideoCaptureDelegate 失败")

        with self._lock:
            callback = self._frame_callback
        delegate.set_callback(callback)

        queue = dispatch_queue_create(b"omni3b.video.capture.queue", DISPATCH_QUEUE_SERIAL)
        output.setSampleBufferDelegate_queue_(delegate, queue)

        session.startRunning()
        if not session.isRunning():
            raise RuntimeError("视频采集 session 启动后未处于运行状态")

        with self._lock:
            self._session = session
            self._delegate = delegate
            self._queue = queue
            self._is_running = True

    def stop(self) -> None:
        """停止摄像头采集会话。"""
        with self._lock:
            session = self._session
            if not self._is_running or session is None:
                return

        try:
            session.stopRunning()
        except Exception:
            LOGGER.exception("停止视频采集会话失败")
        finally:
            with self._lock:
                self._session = None
                self._delegate = None
                self._queue = None
                self._is_running = False
