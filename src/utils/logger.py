"""日志工具 — 统一的日志配置与格式化"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
DEFAULT_LOG_LEVEL: str = "INFO"


def _load_log_level_from_config(config_path: Path = DEFAULT_CONFIG_PATH) -> str | None:
    """从默认配置文件读取日志等级。"""
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config: dict[str, Any] | None = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(config, dict):
        return None

    system_config: Any = config.get("system")
    if not isinstance(system_config, dict):
        return None

    log_level: Any = system_config.get("log_level")
    if isinstance(log_level, str):
        return log_level
    return None


def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """初始化全局日志配置。

    当 `level` 保持默认值时，会优先尝试读取 `configs/default.yaml` 中的 `system.log_level`。
    """
    configured_level: str | None = _load_log_level_from_config()
    use_config_level: bool = level == DEFAULT_LOG_LEVEL and configured_level is not None
    effective_level: str = configured_level if use_config_level else level
    logging_level: int = getattr(logging, effective_level.upper(), logging.INFO)

    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """返回指定名称的日志实例。"""
    return logging.getLogger(name)
