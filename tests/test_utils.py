"""工具层测试 — 日志"""
from __future__ import annotations

import logging
import re

from src.utils.logger import get_logger, setup_logging


def test_setup_logging_sets_root_level() -> None:
    """验证 setup_logging 可以设置 root logger 等级。"""
    setup_logging(level="DEBUG")
    root_logger: logging.Logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_get_logger_returns_named_logger() -> None:
    """验证 get_logger 返回指定名称的 logger。"""
    logger: logging.Logger = get_logger("pipeline.realtime")
    assert logger.name == "pipeline.realtime"


def test_logging_format_includes_time_level_and_name(capsys: object) -> None:
    """验证日志格式包含时间、级别和模块名。"""
    setup_logging(level="INFO")
    logger: logging.Logger = get_logger("tests.logger")
    logger.info("format check")

    captured = capsys.readouterr()
    output: str = captured.err.strip()

    assert output
    assert "[INFO]" in output
    assert "[tests.logger]" in output
    assert re.search(r"^\[\d{2}:\d{2}:\d{2}\]", output) is not None
