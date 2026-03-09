"""多人场景 ROI 裁剪 — 从全局画面中裁出各个人物区域，分别送入模型"""
from __future__ import annotations

import io
import logging
from typing import Any, Final

import numpy as np
from PIL import Image

try:
    import Vision  # type: ignore
    from Foundation import NSData  # type: ignore
    from Quartz import CIImage  # type: ignore

    HAS_VISION: bool = True
except ImportError:
    Vision = None  # type: ignore[assignment]
    NSData = None  # type: ignore[assignment]
    CIImage = None  # type: ignore[assignment]
    HAS_VISION = False

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


class ROIExtractor:
    """基于 macOS Vision 的人体 ROI 提取器。"""

    def __init__(self, padding_ratio: float = 0.2, detection_backend: str = "vision") -> None:
        """
        初始化 ROI 提取器。

        Args:
            padding_ratio: 对检测框每边扩展比例。
            detection_backend: 检测后端，目前仅支持 ``vision``。
        """
        if padding_ratio < 0:
            raise ValueError("padding_ratio 不能小于 0")
        if detection_backend != "vision":
            raise ValueError(f"不支持的检测后端: {detection_backend}")

        self._padding_ratio: float = padding_ratio
        self._detection_backend: str = detection_backend

    def extract(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        对输入帧执行人体检测并裁剪 ROI。

        若未检测到人体，返回整帧作为兜底结果。
        """
        self._validate_frame(frame)
        frame_height, frame_width = frame.shape[:2]

        detections = self.detect_persons(frame)
        if not detections:
            return [frame]

        rois: list[np.ndarray] = []
        for detection in detections:
            bbox = detection.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue

            y1, y2, x1, x2 = self._vision_bbox_to_pixel(
                bbox=bbox,
                frame_height=frame_height,
                frame_width=frame_width,
                padding_ratio=self._padding_ratio,
            )
            if y2 <= y1 or x2 <= x1:
                continue

            roi = frame[y1:y2, x1:x2].copy()
            if roi.size > 0:
                rois.append(roi)

        return rois if rois else [frame]

    def detect_persons(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        检测输入帧中的人体矩形。

        Args:
            frame: RGB 帧，shape 为 ``(H, W, 3)``。

        Returns:
            检测结果列表，元素形如
            ``{"bbox": (x, y, w, h), "confidence": float}``。
        """
        self._validate_frame(frame)
        if self._detection_backend != "vision":
            return []
        if not HAS_VISION:
            LOGGER.warning("Vision 框架不可用，跳过人体检测")
            return []

        ci_image = self._numpy_to_ciimage(frame)
        request = Vision.VNDetectHumanRectanglesRequest.alloc().init()
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)

        success, error = handler.performRequests_error_([request], None)
        if not success or error is not None:
            LOGGER.warning("Vision 人体检测失败: %s", error)
            return []

        observations = request.results()
        if observations is None:
            return []

        detections: list[dict[str, Any]] = []
        for observation in observations:
            bounding_box = observation.boundingBox()
            detections.append(
                {
                    "bbox": (
                        float(bounding_box.origin.x),
                        float(bounding_box.origin.y),
                        float(bounding_box.size.width),
                        float(bounding_box.size.height),
                    ),
                    "confidence": float(observation.confidence()),
                }
            )
        return detections

    def _numpy_to_ciimage(self, frame: np.ndarray) -> Any:
        """将 numpy RGB 帧转换为 Vision 可用的 CIImage。"""
        pil_image = Image.fromarray(frame, mode="RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        ns_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
        return CIImage.imageWithData_(ns_data)

    def _vision_bbox_to_pixel(
        self,
        bbox: tuple[float, float, float, float],
        frame_height: int,
        frame_width: int,
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        """Vision 归一化坐标转 numpy 像素坐标。"""
        x, y, w, h = bbox

        px = int(x * frame_width)
        py = int((1 - y - h) * frame_height)
        pw = int(w * frame_width)
        ph = int(h * frame_height)

        pad_w = int(pw * padding_ratio)
        pad_h = int(ph * padding_ratio)

        y1 = max(0, py - pad_h)
        y2 = min(frame_height, py + ph + pad_h)
        x1 = max(0, px - pad_w)
        x2 = min(frame_width, px + pw + pad_w)
        return y1, y2, x1, x2

    def _validate_frame(self, frame: np.ndarray) -> None:
        """校验输入帧格式是否符合约定。"""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 shape=(H, W, 3) 的 RGB 图像")
        if frame.dtype != np.uint8:
            raise ValueError("frame dtype 必须是 uint8")
