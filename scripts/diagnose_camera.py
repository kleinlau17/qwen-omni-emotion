#!/usr/bin/env python3
"""摄像头采集诊断脚本 — 隔离测试 AVFoundation 是否能正常交付帧"""
import sys
import time

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print()

# ── 1. 导入检查 ──────────────────────────────────────────────

try:
    import AVFoundation
    import CoreMedia

    try:
        import CoreVideo
    except ImportError:
        import Quartz as CoreVideo

    import objc
    from Foundation import NSDate, NSObject, NSRunLoop

    print("[OK] PyObjC / AVFoundation 导入成功")
except ImportError as exc:
    print(f"[FAIL] 导入失败: {exc}")
    sys.exit(1)

try:
    from dispatch import DISPATCH_QUEUE_SERIAL, dispatch_queue_create

    print("[OK] dispatch 模块可用")
    dispatch_ok = True
except ImportError:
    print("[WARN] dispatch 模块不可用，将使用 None (主队列)")
    dispatch_ok = False

print()

# ── 2. 权限检查 ──────────────────────────────────────────────

status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
    AVFoundation.AVMediaTypeVideo
)
status_names = {0: "NotDetermined", 1: "Restricted", 2: "Denied", 3: "Authorized"}
print(f"摄像头授权状态: {status} ({status_names.get(status, '未知')})")

if status == 0:
    print("  → 首次请求权限，请在弹窗中点击「允许」...")
    import threading

    ev = threading.Event()
    result = [False]

    def _cb(granted):
        result[0] = granted
        ev.set()

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeVideo, _cb
    )
    ev.wait(timeout=60)
    if not result[0]:
        print("[FAIL] 用户拒绝了摄像头权限")
        sys.exit(1)
    print("[OK] 权限已授予")
elif status == 2:
    print("[FAIL] 摄像头权限被拒绝！请在 系统设置 → 隐私与安全性 → 摄像头 中授权终端应用")
    sys.exit(1)
elif status != 3:
    print(f"[FAIL] 未知授权状态: {status}")
    sys.exit(1)

print()

# ── 3. 设备枚举 ──────────────────────────────────────────────

devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
    AVFoundation.AVMediaTypeVideo
)
print(f"可用摄像头数量: {len(devices)}")
for i, dev in enumerate(devices):
    print(f"  [{i}] {dev.localizedName()}  (ID: {dev.uniqueID()})")

default_dev = AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(
    AVFoundation.AVMediaTypeVideo
)
if default_dev is None:
    print("[FAIL] 无默认摄像头设备")
    sys.exit(1)
print(f"默认设备: {default_dev.localizedName()}")
print()

# ── 4. 尝试采集帧 ────────────────────────────────────────────

frame_count = [0]
first_frame_time = [None]


class DiagDelegate(NSObject):
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sb, conn):
        frame_count[0] += 1
        if frame_count[0] == 1:
            first_frame_time[0] = time.time()
            ib = CoreMedia.CMSampleBufferGetImageBuffer(sb)
            if ib:
                w = CoreVideo.CVPixelBufferGetWidth(ib)
                h = CoreVideo.CVPixelBufferGetHeight(ib)
                fmt = CoreVideo.CVPixelBufferGetPixelFormatType(ib)
                print(f"[OK] 首帧到达! 分辨率={w}x{h} 像素格式={fmt}")
            else:
                print("[OK] 首帧到达 (但无法获取 image buffer)")


# 4a. 创建 session
session = AVFoundation.AVCaptureSession.alloc().init()
print("Session 已创建")

# 4b. 使用 beginConfiguration/commitConfiguration
session.beginConfiguration()

# 4c. 添加输入
inp, err = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(default_dev, None)
if err:
    print(f"[FAIL] 创建输入失败: {err}")
    sys.exit(1)
if session.canAddInput_(inp):
    session.addInput_(inp)
    print("[OK] 输入已添加")
else:
    print("[FAIL] 无法添加输入到 session")
    sys.exit(1)

# 4d. 添加输出 (不指定像素格式，让系统自动选择)
vid_output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
vid_output.setAlwaysDiscardsLateVideoFrames_(True)

if session.canAddOutput_(vid_output):
    session.addOutput_(vid_output)
    print("[OK] 输出已添加")
else:
    print("[FAIL] 无法添加输出到 session")
    sys.exit(1)

# 4e. 在输入/输出都就绪后检查并设置预设
presets_to_try = [
    ("1920x1080", AVFoundation.AVCaptureSessionPreset1920x1080),
    ("1280x720", AVFoundation.AVCaptureSessionPreset1280x720),
    ("640x480", AVFoundation.AVCaptureSessionPreset640x480),
]
for name, preset in presets_to_try:
    if session.canSetSessionPreset_(preset):
        session.setSessionPreset_(preset)
        print(f"[OK] 使用预设: {name}")
        break
else:
    print("[WARN] 所有预设均不支持，使用设备默认")

session.commitConfiguration()
print("[OK] Session 配置已提交")

# 4f. 设置 delegate
delegate = DiagDelegate.alloc().init()
if delegate is None:
    print("[FAIL] delegate 初始化失败")
    sys.exit(1)

if dispatch_ok:
    queue = dispatch_queue_create(b"diag.camera.queue", DISPATCH_QUEUE_SERIAL)
    vid_output.setSampleBufferDelegate_queue_(delegate, queue)
    print("[OK] Delegate 已设置 (使用 dispatch serial queue)")
else:
    vid_output.setSampleBufferDelegate_queue_(delegate, None)
    print("[OK] Delegate 已设置 (使用 None/主队列)")

# 4g. 启动
print("\n正在启动 session...")
session.startRunning()
is_running = session.isRunning()
print(f"session.isRunning() = {is_running}")

if not is_running:
    print("[FAIL] Session 启动后未处于运行状态!")
    sys.exit(1)

# 4h. 等待帧 — 通过 NSRunLoop 驱动事件
print("\n等待帧到达 (最多 8 秒) ...")
start_t = time.time()
while time.time() - start_t < 8:
    NSRunLoop.currentRunLoop().runUntilDate_(
        NSDate.dateWithTimeIntervalSinceNow_(0.1)
    )
    if frame_count[0] >= 1:
        break

# 如果 dispatch queue 方式没帧，尝试 None 队列
if frame_count[0] == 0 and dispatch_ok:
    print("\n[WARN] dispatch queue 方式未收到帧，尝试使用 None 队列 ...")
    vid_output.setSampleBufferDelegate_queue_(delegate, None)
    start_t2 = time.time()
    while time.time() - start_t2 < 5:
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )
        if frame_count[0] >= 1:
            break

# 继续多收几帧
if frame_count[0] > 0:
    extra_start = time.time()
    while time.time() - extra_start < 2:
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

session.stopRunning()

# ── 5. 结果 ──────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"总接收帧数: {frame_count[0]}")
if frame_count[0] > 0:
    elapsed = time.time() - first_frame_time[0]
    fps = frame_count[0] / max(elapsed, 0.001)
    print(f"平均帧率: {fps:.1f} fps")
    print(f"\n[OK] 摄像头采集正常!")
else:
    print(f"\n[FAIL] 未收到任何帧!")
    print("可能原因:")
    print("  1. 摄像头权限未正确授权（检查 系统设置 → 隐私与安全性 → 摄像头）")
    print("  2. USB 摄像头驱动问题（尝试拔插或在 FaceTime 中测试）")
    print("  3. PyObjC delegate 回调不兼容")
