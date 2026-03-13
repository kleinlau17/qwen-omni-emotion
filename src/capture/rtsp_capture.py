"""RTSP (ffmpeg) 音视频采集。"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Final

import numpy as np

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
FrameCallback = Callable[[np.ndarray, float], None]
AudioCallback = Callable[[np.ndarray, float], None]


@dataclass(frozen=True)
class RTSPOptions:
    url: str
    rtsp_transport: str = "udp"
    ffmpeg_path: str = "ffmpeg"
    extra_args: list[str] | None = None
    log_tail: int = 20
    probe_size: int | None = 1000000
    analyze_duration: int | None = 1000000


def _ensure_ffmpeg_available(ffmpeg_path: str) -> None:
    if "/" in ffmpeg_path or "\\" in ffmpeg_path:
        return
    if shutil.which(ffmpeg_path) is None:
        raise RuntimeError(
            "未找到 ffmpeg，可通过 brew install ffmpeg 或配置 capture.video.ffmpeg_path"
        )


def _read_exact(stream, size: int, stop_event: Event) -> bytes:  # noqa: ANN001
    buf = bytearray()
    while len(buf) < size and not stop_event.is_set():
        chunk = stream.read(size - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)


def _should_warn_black_frame(frame: np.ndarray) -> bool:
    try:
        return frame.mean() < 1.0
    except Exception:
        return False


class RTSPVideoCapture:
    """通过 ffmpeg 拉取 RTSP 视频帧。"""

    def __init__(
        self,
        options: RTSPOptions,
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
    ) -> None:
        self._options = options
        self._resolution = resolution
        self._fps = fps
        self._frame_callback: FrameCallback | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: Thread | None = None
        self._stderr_thread: Thread | None = None
        self._stop_event = Event()
        self._lock = Lock()
        self._is_running = False
        self._stderr_lines: list[str] = []
        self._last_frame_ts: float | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def set_frame_callback(self, callback: FrameCallback) -> None:
        with self._lock:
            self._frame_callback = callback

    def start(self) -> None:
        _ensure_ffmpeg_available(self._options.ffmpeg_path)

        with self._lock:
            if self._is_running:
                return
            self._stop_event.clear()

        width, height = self._resolution
        cmd = [
            self._options.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            self._options.rtsp_transport,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
        ]

        if self._options.probe_size is not None:
            cmd += ["-probesize", str(self._options.probe_size)]
        if self._options.analyze_duration is not None:
            cmd += ["-analyzeduration", str(self._options.analyze_duration)]

        cmd += [
            "-i",
            self._options.url,
        ]

        if self._options.extra_args:
            cmd.extend(self._options.extra_args)

        cmd += [
            "-an",
            "-vf",
            f"scale={width}:{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self._fps),
            "-f",
            "rawvideo",
            "pipe:1",
        ]

        LOGGER.info("RTSP video ffmpeg: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("ffmpeg 未能打开 stdout/stderr 管道")

        self._proc = proc
        self._reader_thread = Thread(
            target=self._read_loop,
            name="rtsp-video-reader",
            daemon=True,
        )
        self._stderr_thread = Thread(
            target=self._stderr_loop,
            name="rtsp-video-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

        with self._lock:
            self._is_running = True

    def stop(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        self._stop_event.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                LOGGER.exception("终止 RTSP 视频 ffmpeg 失败")
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)

        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

        self._proc = None
        self._reader_thread = None
        self._stderr_thread = None

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > self._options.log_tail:
                        self._stderr_lines.pop(0)
                LOGGER.debug("ffmpeg(video): %s", line)
        except Exception:
            LOGGER.exception("读取 RTSP 视频 ffmpeg stderr 失败")

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        width, height = self._resolution
        frame_size = width * height * 3
        frame_count = 0

        start_ts = time.time()
        warned_black = False
        while not self._stop_event.is_set():
            data = _read_exact(proc.stdout, frame_size, self._stop_event)
            if not data:
                if proc.poll() is not None:
                    break
                if time.time() - start_ts > 5.0 and frame_count == 0:
                    LOGGER.warning("RTSP 视频 5s 内未收到帧，请检查 RTSP 地址/网络/传输协议")
                time.sleep(0.005)
                continue
            if len(data) != frame_size:
                LOGGER.debug(
                    "RTSP 视频读取到非完整帧 (%d/%d bytes)，已丢弃",
                    len(data),
                    frame_size,
                )
                continue

            frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            if not warned_black and _should_warn_black_frame(frame):
                LOGGER.warning("RTSP 视频首帧几乎为黑色，可能是码流异常或解码失败")
                warned_black = True
            with self._lock:
                callback = self._frame_callback
            if callback is not None:
                callback(frame, time.time())

            frame_count += 1
            if frame_count == 1:
                LOGGER.info("RTSP 视频首帧到达 (%dx%d)", width, height)
            elif frame_count % 300 == 0:
                LOGGER.debug("RTSP 已接收 %d 帧", frame_count)

        if proc.poll() is not None:
            LOGGER.warning(
                "RTSP 视频 ffmpeg 已退出 (code=%s). stderr_tail=%s",
                proc.returncode,
                " | ".join(self._stderr_lines),
            )


class RTSPAudioCapture:
    """通过 ffmpeg 拉取 RTSP 音频流。"""

    def __init__(
        self,
        options: RTSPOptions,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 100,
    ) -> None:
        self._options = options
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_duration_ms = max(int(chunk_duration_ms), 10)
        self._audio_callback: AudioCallback | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: Thread | None = None
        self._stderr_thread: Thread | None = None
        self._stop_event = Event()
        self._lock = Lock()
        self._is_running = False
        self._stderr_lines: list[str] = []

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def set_audio_callback(self, callback: AudioCallback) -> None:
        with self._lock:
            self._audio_callback = callback

    def start(self) -> None:
        _ensure_ffmpeg_available(self._options.ffmpeg_path)

        with self._lock:
            if self._is_running:
                return
            self._stop_event.clear()

        cmd = [
            self._options.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            self._options.rtsp_transport,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
        ]

        if self._options.probe_size is not None:
            cmd += ["-probesize", str(self._options.probe_size)]
        if self._options.analyze_duration is not None:
            cmd += ["-analyzeduration", str(self._options.analyze_duration)]

        cmd += [
            "-i",
            self._options.url,
        ]

        if self._options.extra_args:
            cmd.extend(self._options.extra_args)

        cmd += [
            "-vn",
            "-ac",
            str(self._channels),
            "-ar",
            str(self._sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]

        LOGGER.info("RTSP audio ffmpeg: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("ffmpeg 未能打开 stdout/stderr 管道")

        self._proc = proc
        self._reader_thread = Thread(
            target=self._read_loop,
            name="rtsp-audio-reader",
            daemon=True,
        )
        self._stderr_thread = Thread(
            target=self._stderr_loop,
            name="rtsp-audio-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

        with self._lock:
            self._is_running = True

    def stop(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        self._stop_event.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                LOGGER.exception("终止 RTSP 音频 ffmpeg 失败")
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)

        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

        self._proc = None
        self._reader_thread = None
        self._stderr_thread = None

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > self._options.log_tail:
                        self._stderr_lines.pop(0)
                LOGGER.debug("ffmpeg(audio): %s", line)
        except Exception:
            LOGGER.exception("读取 RTSP 音频 ffmpeg stderr 失败")

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        samples_per_chunk = int(self._sample_rate * self._chunk_duration_ms / 1000)
        bytes_per_chunk = samples_per_chunk * self._channels * 2

        start_ts = time.time()
        while not self._stop_event.is_set():
            data = _read_exact(proc.stdout, bytes_per_chunk, self._stop_event)
            if not data:
                if proc.poll() is not None:
                    break
                if time.time() - start_ts > 5.0:
                    LOGGER.warning("RTSP 音频 5s 内未收到数据，请确认码流包含音频轨道")
                time.sleep(0.005)
                continue

            pcm16 = np.frombuffer(data, dtype=np.int16)
            if pcm16.size == 0:
                continue
            audio_float32 = (pcm16.astype(np.float32) / 32768.0).copy()

            with self._lock:
                callback = self._audio_callback
            if callback is not None:
                callback(audio_float32, time.time())

        if proc.poll() is not None:
            LOGGER.warning(
                "RTSP 音频 ffmpeg 已退出 (code=%s). stderr_tail=%s",
                proc.returncode,
                " | ".join(self._stderr_lines),
            )
