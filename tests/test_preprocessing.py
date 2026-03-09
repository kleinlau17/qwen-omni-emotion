"""输入准备层测试 — 帧采样与 ROI 裁剪"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from src.preprocessing.frame_sampler import FrameSampler
from src.preprocessing.roi_extractor import ROIExtractor

HAS_VISION = importlib.util.find_spec("Vision") is not None
IS_MACOS = sys.platform == "darwin"


def test_frame_sampler_uniform_sampling() -> None:
    """验证 uniform 策略按等间隔采样并保留尾帧。"""
    frames = [
        (np.full((2, 2, 3), fill_value=index, dtype=np.uint8), float(index))
        for index in range(10)
    ]
    sampler = FrameSampler(strategy="uniform", max_frames=4)

    sampled = sampler.sample(frames)

    assert len(sampled) == 4
    assert [int(frame[0, 0, 0]) for frame in sampled] == [0, 3, 6, 9]


def test_frame_sampler_returns_all_when_frames_insufficient() -> None:
    """验证当帧数不足 max_frames 时全部返回。"""
    frames = [
        (np.full((2, 2, 3), fill_value=index, dtype=np.uint8), float(index))
        for index in range(3)
    ]
    sampler = FrameSampler(strategy="uniform", max_frames=4)

    sampled = sampler.sample(frames)

    assert len(sampled) == 3
    assert [int(frame[0, 0, 0]) for frame in sampled] == [0, 1, 2]


@pytest.mark.skipif(
    not IS_MACOS or not HAS_VISION,
    reason="仅 macOS 且安装 Vision(PyObjC) 时执行",
)
def test_roi_extractor_detect_persons_returns_list() -> None:
    """验证人体检测接口可执行并返回列表结构。"""
    extractor = ROIExtractor(padding_ratio=0.2, detection_backend="vision")
    frame = np.zeros((128, 128, 3), dtype=np.uint8)

    detections = extractor.detect_persons(frame)

    assert isinstance(detections, list)
    if detections:
        first = detections[0]
        assert "bbox" in first
        assert "confidence" in first


@pytest.mark.skipif(
    not IS_MACOS or not HAS_VISION,
    reason="仅 macOS 且安装 Vision(PyObjC) 时执行",
)
def test_roi_extractor_extract_fallback_returns_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证未检测到人体时返回整帧兜底。"""
    extractor = ROIExtractor(padding_ratio=0.2, detection_backend="vision")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(extractor, "detect_persons", lambda _: [])

    rois = extractor.extract(frame)

    assert len(rois) >= 1
    np.testing.assert_array_equal(rois[0], frame)
