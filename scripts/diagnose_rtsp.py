"""RTSP 音视频接入诊断脚本。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.rtsp_capture import RTSPAudioCapture, RTSPOptions, RTSPVideoCapture  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSP 诊断工具")
    parser.add_argument("--url", required=True, help="RTSP 地址")
    parser.add_argument("--transport", default="udp", help="rtsp_transport: udp/tcp")
    parser.add_argument("--duration", type=float, default=6.0, help="采集时长(秒)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--probe-size", type=int, default=1000000)
    parser.add_argument("--analyze-duration", type=int, default=1000000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = RTSPOptions(
        url=args.url,
        rtsp_transport=args.transport,
        probe_size=args.probe_size,
        analyze_duration=args.analyze_duration,
    )

    frame_count = 0
    audio_samples = 0
    last_frame_ts = 0.0
    last_audio_ts = 0.0

    stop_event = Event()

    def on_frame(frame, ts) -> None:  # noqa: ANN001
        nonlocal frame_count, last_frame_ts
        frame_count += 1
        last_frame_ts = ts

    def on_audio(chunk, ts) -> None:  # noqa: ANN001
        nonlocal audio_samples, last_audio_ts
        audio_samples += len(chunk)
        last_audio_ts = ts

    video = RTSPVideoCapture(
        options=options,
        resolution=(args.width, args.height),
        fps=args.fps,
    )
    audio = RTSPAudioCapture(
        options=options,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_duration_ms=args.chunk_ms,
    )

    print("启动 RTSP 诊断...")
    video.set_frame_callback(on_frame)
    audio.set_audio_callback(on_audio)
    video.start()
    audio.start()

    start_ts = time.time()
    while time.time() - start_ts < args.duration:
        time.sleep(0.2)
        if stop_event.is_set():
            break

    video.stop()
    audio.stop()

    print("诊断完成:")
    print(f"  视频帧数: {frame_count}")
    if last_frame_ts > 0:
        print(f"  最新视频时间戳: {last_frame_ts:.3f}")
    print(f"  音频样本数: {audio_samples}")
    if last_audio_ts > 0:
        print(f"  最新音频时间戳: {last_audio_ts:.3f}")


if __name__ == "__main__":
    main()
