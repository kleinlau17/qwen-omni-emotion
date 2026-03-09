---
name: avfoundation-capture
description: macOS AVFoundation video and audio capture via PyObjC on Apple Silicon. Use when implementing camera capture, microphone recording, frame grabbing, or audio stream capture in the capture layer.
---

# macOS AVFoundation 音视频采集

通过 PyObjC 调用 AVFoundation 框架，在 macOS 上实现摄像头和麦克风的实时采集。

## 依赖导入

```python
import AVFoundation as AVF
import CoreMedia
from Foundation import NSObject
from Quartz import CIImage
import objc
import numpy as np
```

## 视频采集

### 核心流程

1. 创建 AVCaptureSession
2. 获取摄像头设备 → 创建 Input
3. 创建 VideoDataOutput → 设置像素格式 → 添加 Delegate
4. 启动 Session

### 完整视频采集模板

```python
import AVFoundation as AVF
import CoreMedia
import CoreVideo
from Foundation import NSObject
import objc
import numpy as np
import threading

# kCVPixelFormatType_32BGRA = 0x42475241 = 1111970369
BGRA_FORMAT = 1111970369


class VideoCaptureDelegate(NSObject):
    """AVCaptureVideoDataOutput 的回调代理"""

    def init(self):
        self = objc.super(VideoCaptureDelegate, self).init()
        if self is None:
            return None
        self._frame_callback = None
        self._lock = threading.Lock()
        return self

    def set_callback_(self, callback):
        with self._lock:
            self._frame_callback = callback

    def captureOutput_didOutputSampleBuffer_fromConnection_(
        self, output, sample_buffer, connection
    ):
        """每帧回调 — AVFoundation 在后台线程调用"""
        if self._frame_callback is None:
            return

        image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
        if image_buffer is None:
            return

        # 锁定像素数据
        CoreVideo.CVPixelBufferLockBaseAddress(image_buffer, 0)
        try:
            base_address = CoreVideo.CVPixelBufferGetBaseAddress(image_buffer)
            width = CoreVideo.CVPixelBufferGetWidth(image_buffer)
            height = CoreVideo.CVPixelBufferGetHeight(image_buffer)
            bytes_per_row = CoreVideo.CVPixelBufferGetBytesPerRow(image_buffer)

            # 将 BGRA 缓冲区转为 numpy 数组
            buf = (ctypes.c_uint8 * (bytes_per_row * height)).from_address(
                base_address.__int__()
            )
            frame_bgra = np.frombuffer(buf, dtype=np.uint8).reshape(
                (height, bytes_per_row // 4, 4)
            )[:, :width, :]

            # BGRA → RGB
            frame_rgb = frame_bgra[:, :, [2, 1, 0]].copy()

            timestamp = CoreMedia.CMSampleBufferGetPresentationTimeStamp(
                sample_buffer
            )
            ts_seconds = CoreMedia.CMTimeGetSeconds(timestamp)

            self._frame_callback(frame_rgb, ts_seconds)
        finally:
            CoreVideo.CVPixelBufferUnlockBaseAddress(image_buffer, 0)


def create_video_capture_session(
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    frame_callback=None,
):
    """创建并启动视频采集 session"""
    session = AVF.AVCaptureSession.alloc().init()
    session.setSessionPreset_(AVF.AVCaptureSessionPreset1920x1080)

    # 获取默认摄像头
    device = AVF.AVCaptureDevice.defaultDeviceWithMediaType_(
        AVF.AVMediaTypeVideo
    )
    if device is None:
        raise RuntimeError("未找到摄像头设备")

    # 配置帧率
    device.lockForConfiguration_(None)
    desired_range = None
    for fmt_range in device.activeFormat().videoSupportedFrameRateRanges():
        if fmt_range.maxFrameRate() >= fps:
            desired_range = fmt_range
            break
    if desired_range:
        device.setActiveVideoMinFrameDuration_(
            CoreMedia.CMTimeMake(1, fps)
        )
        device.setActiveVideoMaxFrameDuration_(
            CoreMedia.CMTimeMake(1, fps)
        )
    device.unlockForConfiguration()

    # 创建输入
    input_dev, error = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(
        device, None
    )
    if error:
        raise RuntimeError(f"创建视频输入失败: {error}")
    session.addInput_(input_dev)

    # 创建输出
    output = AVF.AVCaptureVideoDataOutput.alloc().init()
    output.setVideoSettings_({
        "PixelFormatType": BGRA_FORMAT,
    })
    output.setAlwaysDiscardsLateVideoFrames_(True)

    delegate = VideoCaptureDelegate.alloc().init()
    if frame_callback:
        delegate.set_callback_(frame_callback)

    from dispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL
    queue = dispatch_queue_create(b"video_capture_queue", DISPATCH_QUEUE_SERIAL)
    output.setSampleBufferDelegate_queue_(delegate, queue)

    session.addOutput_(output)
    session.startRunning()

    return session, delegate
```

### CVPixelBuffer → numpy 的替代方案

如果 `ctypes` 方式有问题，可以用 CoreImage 中转：

```python
from Quartz import CIImage, CIContext, kCIFormatRGBA8

ci_context = CIContext.contextWithOptions_(None)
ci_image = CIImage.imageWithCVPixelBuffer_(image_buffer)

width = int(ci_image.extent().size.width)
height = int(ci_image.extent().size.height)
bitmap = bytearray(width * height * 4)

ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
    ci_image, bitmap, width * 4,
    ci_image.extent(), kCIFormatRGBA8, None
)

frame = np.frombuffer(bitmap, dtype=np.uint8).reshape((height, width, 4))
frame_rgb = frame[:, :, :3].copy()  # 去掉 alpha
```

## 音频采集

### 完整音频采集模板

```python
import AVFoundation as AVF
import CoreMedia
from Foundation import NSObject
import objc
import numpy as np


class AudioCaptureDelegate(NSObject):
    """AVCaptureAudioDataOutput 的回调代理"""

    def init(self):
        self = objc.super(AudioCaptureDelegate, self).init()
        if self is None:
            return None
        self._audio_callback = None
        return self

    def set_callback_(self, callback):
        self._audio_callback = callback

    def captureOutput_didOutputSampleBuffer_fromConnection_(
        self, output, sample_buffer, connection
    ):
        if self._audio_callback is None:
            return

        block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
        if block_buffer is None:
            return

        length, data = CoreMedia.CMBlockBufferGetDataPointer(
            block_buffer, 0, None, None
        )

        # 假设 16-bit PCM，单声道
        audio_data = np.frombuffer(
            data[:length], dtype=np.int16
        ).astype(np.float32) / 32768.0

        timestamp = CoreMedia.CMSampleBufferGetPresentationTimeStamp(
            sample_buffer
        )
        ts_seconds = CoreMedia.CMTimeGetSeconds(timestamp)

        self._audio_callback(audio_data, ts_seconds)


def create_audio_capture_session(
    sample_rate: int = 16000,
    channels: int = 1,
    audio_callback=None,
):
    """创建并启动音频采集 session"""
    session = AVF.AVCaptureSession.alloc().init()

    device = AVF.AVCaptureDevice.defaultDeviceWithMediaType_(
        AVF.AVMediaTypeAudio
    )
    if device is None:
        raise RuntimeError("未找到麦克风设备")

    input_dev, error = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(
        device, None
    )
    if error:
        raise RuntimeError(f"创建音频输入失败: {error}")
    session.addInput_(input_dev)

    output = AVF.AVCaptureAudioDataOutput.alloc().init()
    # 设置音频格式
    audio_settings = {
        "AVSampleRateKey": sample_rate,
        "AVNumberOfChannelsKey": channels,
        "AVLinearPCMBitDepthKey": 16,
        "AVLinearPCMIsFloatKey": False,
        "AVFormatIDKey": 1819304813,  # kAudioFormatLinearPCM
    }
    output.setAudioSettings_(audio_settings)

    delegate = AudioCaptureDelegate.alloc().init()
    if audio_callback:
        delegate.set_callback_(audio_callback)

    from dispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL
    queue = dispatch_queue_create(b"audio_capture_queue", DISPATCH_QUEUE_SERIAL)
    output.setSampleBufferDelegate_queue_(delegate, queue)

    session.addOutput_(output)
    session.startRunning()

    return session, delegate
```

## 合并音视频采集

可以在同一个 AVCaptureSession 中同时添加视频和音频的 input/output：

```python
session = AVF.AVCaptureSession.alloc().init()

# 添加视频
video_device = AVF.AVCaptureDevice.defaultDeviceWithMediaType_(AVF.AVMediaTypeVideo)
video_input, _ = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(video_device, None)
session.addInput_(video_input)
session.addOutput_(video_output)  # 视频输出

# 添加音频
audio_device = AVF.AVCaptureDevice.defaultDeviceWithMediaType_(AVF.AVMediaTypeAudio)
audio_input, _ = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(audio_device, None)
session.addInput_(audio_input)
session.addOutput_(audio_output)  # 音频输出

session.startRunning()
```

## 关键注意事项

1. **回调在后台线程** — `captureOutput_didOutputSampleBuffer_fromConnection_` 由 AVFoundation 在 dispatch queue 上调用，写入共享数据需加锁
2. **CVPixelBuffer 必须 lock/unlock** — 访问像素数据前后必须调用 `LockBaseAddress` / `UnlockBaseAddress`
3. **帧拷贝** — numpy 数组必须 `.copy()` 否则 unlock 后数据失效
4. **内存压力** — 1080p/30fps 每帧约 6MB，及时释放或用环形 buffer
5. **dispatch queue** — 如果 `dispatch` 模块不可用，可用 `libdispatch` 或 `NSOperationQueue` 替代
6. **权限** — macOS 需要摄像头/麦克风使用权限，首次运行会弹出系统授权弹窗
