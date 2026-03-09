---
name: macos-vision-detection
description: macOS Vision framework for human body and face detection via PyObjC. Use when implementing ROI extraction, person detection, face bounding boxes, or human body detection in the preprocessing layer.
---

# macOS Vision 框架人体/人脸检测

通过 PyObjC 调用 macOS Vision 框架，实现人体矩形检测和人脸检测，用于多人场景的 ROI 裁剪。

## 依赖导入

```python
import Vision
from Foundation import NSData
from Quartz import (
    CIImage,
    CGImageSourceCreateWithData,
    CGImageSourceCreateImageAtIndex,
)
import numpy as np
```

## 核心流程

1. 将输入帧转为 Vision 可接受的图像格式（CIImage / CGImage）
2. 创建检测请求（VNDetectHumanRectanglesRequest / VNDetectFaceRectanglesRequest）
3. 创建 VNImageRequestHandler 并执行请求
4. 从 observations 中提取边界框坐标
5. 将归一化坐标转换为像素坐标并裁剪 ROI

## numpy 帧转 CIImage

```python
from PIL import Image
import io
from Foundation import NSData
from Quartz import CIImage


def numpy_to_ciimage(frame: np.ndarray) -> CIImage:
    """numpy RGB 帧 → CIImage（Vision 框架输入格式）"""
    pil_image = Image.fromarray(frame)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    ns_data = NSData.dataWithBytes_length_(buffer.getvalue(), len(buffer.getvalue()))
    ci_image = CIImage.imageWithData_(ns_data)
    return ci_image
```

## 人体矩形检测

```python
import Vision


def detect_human_rectangles(
    ci_image: CIImage,
    upper_body_only: bool = False,
) -> list[dict]:
    """检测图像中的人体矩形区域

    Returns:
        list[dict]: 每个人体的边界框信息
            - bbox: (x, y, w, h) 归一化坐标 [0, 1]
            - confidence: 检测置信度
    """
    request = Vision.VNDetectHumanRectanglesRequest.alloc().init()
    if upper_body_only:
        request.setUpperBodyOnly_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        ci_image, None
    )

    success, error = handler.performRequests_error_([request], None)
    if not success or error:
        return []

    results = []
    observations = request.results()
    if observations is None:
        return []

    for obs in observations:
        bbox = obs.boundingBox()
        results.append({
            "bbox": (
                bbox.origin.x,
                bbox.origin.y,
                bbox.size.width,
                bbox.size.height,
            ),
            "confidence": obs.confidence(),
        })

    return results
```

## 人脸检测

```python
def detect_faces(ci_image: CIImage) -> list[dict]:
    """检测图像中的人脸区域

    Returns:
        list[dict]: 每张脸的边界框
            - bbox: (x, y, w, h) 归一化坐标 [0, 1]
            - confidence: 检测置信度
    """
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()

    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        ci_image, None
    )

    success, error = handler.performRequests_error_([request], None)
    if not success or error:
        return []

    results = []
    observations = request.results()
    if observations is None:
        return []

    for obs in observations:
        bbox = obs.boundingBox()
        results.append({
            "bbox": (
                bbox.origin.x,
                bbox.origin.y,
                bbox.size.width,
                bbox.size.height,
            ),
            "confidence": obs.confidence(),
        })

    return results
```

## 坐标系转换

**Vision 框架坐标系**：
- 归一化到 `[0, 1]`
- **原点在左下角**（不是左上角！）
- `(x, y)` 是 bbox 的左下角

转换到像素坐标（原点在左上角，用于 numpy 裁剪）：

```python
def vision_bbox_to_pixel(
    bbox: tuple[float, float, float, float],
    frame_height: int,
    frame_width: int,
    padding_ratio: float = 0.2,
) -> tuple[int, int, int, int]:
    """Vision 归一化 bbox → 像素坐标 (y1, y2, x1, x2)

    Args:
        bbox: (x, y, w, h) Vision 归一化坐标
        frame_height: 帧高度
        frame_width: 帧宽度
        padding_ratio: 边界扩展比例

    Returns:
        (y1, y2, x1, x2) 像素坐标，可直接用于 frame[y1:y2, x1:x2]
    """
    x, y, w, h = bbox

    # 归一化 → 像素
    px = int(x * frame_width)
    py = int((1 - y - h) * frame_height)  # 翻转 Y 轴
    pw = int(w * frame_width)
    ph = int(h * frame_height)

    # 添加 padding
    pad_w = int(pw * padding_ratio)
    pad_h = int(ph * padding_ratio)

    y1 = max(0, py - pad_h)
    y2 = min(frame_height, py + ph + pad_h)
    x1 = max(0, px - pad_w)
    x2 = min(frame_width, px + pw + pad_w)

    return y1, y2, x1, x2
```

## ROI 裁剪完整流程

```python
def extract_person_rois(
    frame: np.ndarray,
    padding_ratio: float = 0.2,
    upper_body_only: bool = False,
) -> list[np.ndarray]:
    """从帧中检测人体并裁出各人物的 ROI 区域

    Args:
        frame: (H, W, 3) uint8 RGB
        padding_ratio: ROI 边界扩展
        upper_body_only: 是否仅检测上半身

    Returns:
        list[np.ndarray]: 每个人物的裁剪区域
    """
    h, w = frame.shape[:2]
    ci_image = numpy_to_ciimage(frame)

    detections = detect_human_rectangles(ci_image, upper_body_only)
    if not detections:
        return [frame]  # 未检测到人体时返回整帧

    rois = []
    for det in detections:
        y1, y2, x1, x2 = vision_bbox_to_pixel(
            det["bbox"], h, w, padding_ratio
        )
        roi = frame[y1:y2, x1:x2].copy()
        if roi.size > 0:
            rois.append(roi)

    return rois if rois else [frame]
```

## 性能注意事项

1. **Vision 请求在 CPU 执行** — Apple Neural Engine 会自动加速，无需手动指定
2. **CIImage 创建开销** — 对于 30fps 视频流，建议仅对采样后的帧做检测（如 2fps）
3. **复用 handler 无意义** — VNImageRequestHandler 是按图像创建的，不可复用
4. **批量请求** — 可在一次 `performRequests_error_` 中同时执行多个 request（如人体+人脸）
5. **坐标系** — 永远记住 Vision 用左下角原点，numpy 用左上角原点
