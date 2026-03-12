"""Qwen2.5-Omni 实时情绪理解系统 — 主入口"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def _setup_early_logging() -> None:
    """在配置加载前初始化基础日志，确保启动阶段的进度可见。"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 实时情绪理解系统")
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="配置目录路径，目录中应包含 default.yaml/model.yaml/pipeline.yaml",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="禁用 Web 可视化仪表盘",
    )
    return parser.parse_args()


def load_merged_config(config_dir: str | Path) -> dict[str, Any]:
    """
    加载并合并配置文件。

    合并顺序为：default.yaml -> model.yaml -> pipeline.yaml（后者覆盖前者同名键）。
    """
    base_path = Path(config_dir).expanduser().resolve()
    merged: dict[str, Any] = {}
    for file_name in ("default.yaml", "model.yaml", "pipeline.yaml"):
        file_path = base_path / file_name
        merged = _deep_merge_dict(merged, _load_yaml_file(file_path))
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """读取单个 YAML 配置文件。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件根节点必须是对象: {path}")
    return loaded


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，override 同名键覆盖 base。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _start_web_server(
    pipeline: Any,
    viz_config: dict[str, Any],
) -> threading.Thread | None:
    """
    在独立守护线程中启动 Web 可视化服务器。

    Returns:
        运行 uvicorn 的守护线程，若启动失败返回 None。
    """
    try:
        import uvicorn

        from src.visualization.web_server import create_app
    except ImportError:
        LOGGER.warning("uvicorn 或 fastapi 未安装，跳过 Web 仪表盘启动")
        return None

    host = str(viz_config.get("host", "127.0.0.1"))
    port = int(viz_config.get("port", 8080))
    app = create_app(pipeline=pipeline, config=viz_config)

    server_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(server_config)

    thread = threading.Thread(
        target=server.run,
        name="web-dashboard",
        daemon=True,
    )
    thread.start()
    LOGGER.info("Web 仪表盘已启动: http://%s:%d", host, port)
    return thread


def _run_main_loop() -> None:
    """
    保持主线程活跃。

    macOS 的 AVFoundation 需要主线程的 NSRunLoop 处理设备事件（尤其是 USB 摄像头），
    纯 time.sleep() 不会分派 Objective-C 事件，导致帧回调永远不触发。
    """
    try:
        from Foundation import NSDate, NSRunLoop

        LOGGER.info("主线程进入 NSRunLoop（macOS 事件循环）")
        while True:
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(1.0)
            )
    except ImportError:
        LOGGER.warning("Foundation 不可用，退回 time.sleep (摄像头帧回调可能不工作)")
        while True:
            time.sleep(1.0)


def main() -> None:
    """程序主入口。"""
    _setup_early_logging()

    args = parse_args()
    LOGGER.info("正在加载配置文件 (%s) ...", args.config)
    config = load_merged_config(args.config)

    log_level = str(config.get("system", {}).get("log_level", "INFO"))
    setup_logging(level=log_level)
    LOGGER.info("配置加载完成")

    LOGGER.info("正在导入推理框架（首次可能需要 1-2 分钟）...")
    from src.pipeline.realtime_pipeline import RealtimePipeline

    LOGGER.info("推理框架导入完成")

    LOGGER.info("正在初始化管道组件 ...")
    pipeline = RealtimePipeline(config=config)
    LOGGER.info("管道初始化完成")
    viz_config = config.get("visualization", {})
    web_enabled = bool(viz_config.get("enabled", True)) and not args.no_web
    if web_enabled:
        _start_web_server(pipeline, viz_config)
        LOGGER.info("等待 Web 仪表盘控制端触发推理 (请在浏览器中点击“开始推理”)")
    else:
        LOGGER.info("未启用 Web 仪表盘，直接启动流水线")
        pipeline.start()

    LOGGER.info("系统已就绪，按 Ctrl+C 退出")

    try:
        _run_main_loop()
    except KeyboardInterrupt:
        LOGGER.info("收到退出信号，开始优雅停止")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
