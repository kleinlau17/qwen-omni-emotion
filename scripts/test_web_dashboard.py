"""仪表盘独立测试 — 用模拟数据验证 Web 可视化，无需加载模型"""
from __future__ import annotations

import random
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from src.prompts.output_schema import VALID_ACTIONS, VALID_EMOTIONS, EmotionResult


class MockPipeline:
    """模拟 RealtimePipeline，提供与真实 pipeline 相同的只读接口。"""

    def __init__(self) -> None:
        self._emotions: dict[str, EmotionResult] = {}
        self._metrics: deque[dict[str, Any]] = deque(maxlen=100)
        self._trends: dict[str, deque[EmotionResult]] = {}
        self._frame: np.ndarray | None = None
        self._running = False

    def start(self) -> None:
        """启动模拟数据生成。"""
        self._running = True
        self._frame = self._make_gradient_frame()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        tick = 0
        while self._running:
            tick += 1
            person_count = random.choice([1, 1, 1, 2, 2, 3])
            latency = random.uniform(80, 350)
            budget = 400.0 if person_count == 1 else 500.0

            self._metrics.append({
                "latency_ms": latency,
                "person_count": person_count,
                "budget_ms": budget,
                "within_budget": latency <= budget,
                "timestamp": time.time(),
            })

            for i in range(person_count):
                pid = f"person_{i}"
                emotion = random.choice(VALID_EMOTIONS)
                result = EmotionResult(
                    person_id=pid,
                    detected_emotion=emotion,
                    action=random.choice(VALID_ACTIONS),
                )
                self._emotions[pid] = result
                if pid not in self._trends:
                    self._trends[pid] = deque(maxlen=20)
                self._trends[pid].append(result)

            self._frame = self._make_animated_frame(tick)
            time.sleep(1.0)

    def get_current_state(self) -> dict[str, dict[str, Any]]:
        return {
            pid: {
                "person_id": r.person_id,
                "detected_emotion": r.detected_emotion,
                "action": r.action,
                "emotion_intensity": None,
                "confidence": None,
                "description": None,
            }
            for pid, r in self._emotions.items()
        }

    def get_latest_frame(self) -> tuple[np.ndarray | None, float]:
        return self._frame, time.time()

    def get_performance_metrics(self) -> dict[str, Any]:
        history = list(self._metrics)
        if not history:
            return {
                "last_latency_ms": 0.0, "avg_latency_ms": 0.0,
                "person_count": 0, "inference_count": 0, "within_budget": True,
            }
        latencies = [h["latency_ms"] for h in history]
        latest = history[-1]
        return {
            "last_latency_ms": round(latest["latency_ms"], 1),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "person_count": latest["person_count"],
            "inference_count": len(history),
            "within_budget": latest["within_budget"],
        }

    def get_emotion_trends(self, window_count: int = 10) -> dict[str, list[dict[str, Any]]]:
        trends: dict[str, list[dict[str, Any]]] = {}
        for pid, history in self._trends.items():
            items = list(history)[-window_count:]
            trends[pid] = [
                {
                    "detected_emotion": r.detected_emotion,
                    "emotion_intensity": None,
                    "confidence": None,
                }
                for r in items
            ]
        return trends

    @staticmethod
    def _make_gradient_frame() -> np.ndarray:
        """生成一张渐变测试帧。"""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            frame[y, :, 0] = int(255 * y / h)
            frame[y, :, 2] = int(255 * (1 - y / h))
        return frame

    @staticmethod
    def _make_animated_frame(tick: int) -> np.ndarray:
        """生成带动态色带的测试帧。"""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        offset = (tick * 5) % h
        for y in range(h):
            shifted = (y + offset) % h
            frame[y, :, 0] = int(255 * shifted / h)
            frame[y, :, 1] = int(128 * abs(shifted - h / 2) / (h / 2))
            frame[y, :, 2] = int(255 * (1 - shifted / h))
        return frame


def _log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def main() -> None:
    _log("正在导入 Web 模块 ...")
    import uvicorn

    from src.visualization.web_server import create_app

    viz_config = {
        "host": "127.0.0.1",
        "port": 8080,
        "mjpeg_quality": 70,
        "mjpeg_fps": 15,
        "ws_push_interval_ms": 500,
    }

    _log("正在启动模拟数据生成器 ...")
    pipeline = MockPipeline()
    pipeline.start()

    _log("正在创建 FastAPI 应用 ...")
    app = create_app(pipeline=pipeline, config=viz_config)  # type: ignore[arg-type]

    _log(f"仪表盘测试服务已启动: http://{viz_config['host']}:{viz_config['port']}")
    _log("使用模拟数据，无需加载模型。按 Ctrl+C 退出。")

    uvicorn.run(app, host=viz_config["host"], port=viz_config["port"], log_level="warning")


if __name__ == "__main__":
    main()
