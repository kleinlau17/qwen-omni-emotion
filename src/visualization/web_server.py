"""Web 可视化服务 — FastAPI 应用，提供 MJPEG 视频流、WebSocket 数据推送、静态仪表盘"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from src.pipeline.realtime_pipeline import RealtimePipeline

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(pipeline: RealtimePipeline, config: dict[str, Any]) -> FastAPI:
    """
    创建 FastAPI 应用实例，注入 pipeline 引用。

    Args:
        pipeline: 实时推理管道实例（只读访问）。
        config: visualization 配置段。
    """
    app = FastAPI(title="Qwen-Omni 情绪理解仪表盘", docs_url=None, redoc_url=None)

    mjpeg_quality: int = int(config.get("mjpeg_quality", 70))
    mjpeg_fps: int = int(config.get("mjpeg_fps", 15))
    ws_interval_s: float = int(config.get("ws_push_interval_ms", 300)) / 1000.0

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """返回仪表盘首页。"""
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    @app.get("/api/stream")
    async def mjpeg_stream() -> StreamingResponse:
        """MJPEG 视频流端点，浏览器可直接用 <img> 标签消费。"""
        return StreamingResponse(
            _generate_mjpeg(pipeline, mjpeg_quality, mjpeg_fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        """WebSocket 端点：周期性推送情绪状态、性能指标、趋势数据。"""
        await ws.accept()
        LOGGER.info("WebSocket 客户端已连接")
        try:
            while True:
                payload = _build_ws_payload(pipeline)
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
                await asyncio.sleep(ws_interval_s)
        except WebSocketDisconnect:
            LOGGER.info("WebSocket 客户端已断开")
        except Exception:
            LOGGER.exception("WebSocket 推送异常")

    @app.post("/api/pipeline/start")
    async def api_pipeline_start() -> dict[str, str]:
        """启动实时推理流水线。"""
        try:
            pipeline.start()
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - 运行时保护
            LOGGER.exception("启动流水线失败")
            raise HTTPException(status_code=500, detail="failed to start pipeline") from exc

    @app.post("/api/pipeline/pause")
    async def api_pipeline_pause() -> dict[str, str]:
        """暂停实时推理流水线。"""
        try:
            pipeline.stop()
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - 运行时保护
            LOGGER.exception("暂停流水线失败")
            raise HTTPException(status_code=500, detail="failed to pause pipeline") from exc

    @app.get("/api/history/frame/{item_id}/{index}")
    async def api_history_frame(item_id: int, index: int) -> Response:
        """根据推理历史条目 ID 和帧索引返回对应帧的 JPEG 图像。"""
        frames, _ = pipeline.get_history_media(item_id)
        if not frames or Image is None:
            raise HTTPException(status_code=404, detail="frame not found")
        idx = max(0, min(index, len(frames) - 1))
        jpg_bytes = _encode_frame_jpeg(frames[idx], mjpeg_quality)
        return Response(content=jpg_bytes, media_type="image/jpeg")

    @app.get("/api/history/audio/{item_id}")
    async def api_history_audio(item_id: int) -> Response:
        """根据推理历史条目 ID 返回对应音频的 WAV 文件。"""
        _, audio = pipeline.get_history_media(item_id)
        if audio is None:
            raise HTTPException(status_code=404, detail="audio not found")

        audio_cfg = pipeline.get_audio_format()
        wav_bytes = _encode_audio_wav(
            audio,
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            channels=int(audio_cfg.get("channels", 1)),
        )
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.get("/api/history")
    async def api_history(limit: int = 20) -> dict[str, Any]:
        """调试/查询接口：返回最近若干次推理历史概要。"""
        return {"items": pipeline.get_inference_history(limit=limit)}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


async def _generate_mjpeg(
    pipeline: RealtimePipeline,
    quality: int,
    target_fps: int,
) -> Any:
    """异步生成器：持续输出 MJPEG 帧字节。"""
    interval = 1.0 / max(1, target_fps)
    sent_first = False
    while True:
        frame, _ = pipeline.get_latest_frame()
        if frame is not None and Image is not None:
            if not sent_first:
                LOGGER.info("MJPEG 流首帧已发送 (shape=%s)", frame.shape)
                sent_first = True
            jpg_bytes = _encode_frame_jpeg(frame, quality)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
        await asyncio.sleep(interval)


def _encode_frame_jpeg(frame: Any, quality: int) -> bytes:
    """将 numpy RGB 帧编码为 JPEG 字节。"""
    pil_img = Image.fromarray(frame, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _encode_audio_wav(audio: Any, sample_rate: int, channels: int) -> bytes:
    """将 float32 音频数组编码为 16bit PCM WAV 字节。"""
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 1 and channels > 1:
        audio_np = np.tile(audio_np[:, None], (1, channels))
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (audio_np * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _build_ws_payload(pipeline: RealtimePipeline) -> dict[str, Any]:
    """汇聚 pipeline 各接口数据为一个 WebSocket 推送包。"""
    return {
        "timestamp": time.time(),
        "emotions": pipeline.get_current_state(),
        "metrics": pipeline.get_performance_metrics(),
        "trends": pipeline.get_emotion_trends(),
        "history": pipeline.get_inference_history(),
    }


__all__ = ["create_app"]
