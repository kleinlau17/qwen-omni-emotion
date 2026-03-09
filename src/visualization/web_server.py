"""Web 可视化服务 — FastAPI 应用，提供 MJPEG 视频流、WebSocket 数据推送、静态仪表盘"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
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


def _build_ws_payload(pipeline: RealtimePipeline) -> dict[str, Any]:
    """汇聚 pipeline 各接口数据为一个 WebSocket 推送包。"""
    return {
        "timestamp": time.time(),
        "emotions": pipeline.get_current_state(),
        "metrics": pipeline.get_performance_metrics(),
        "trends": pipeline.get_emotion_trends(),
    }


__all__ = ["create_app"]
