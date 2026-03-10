"""模型后端抽象契约。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BackendCapabilities:
    """描述模型后端对实时链路能力的支持情况。"""

    supports_video_frames: bool = True
    supports_audio_array: bool = False
    supports_batch_infer: bool = False
    supports_strict_json_prompting: bool = True
    supports_streaming: bool = False


@dataclass(frozen=True)
class InferenceRequest:
    """后端无关的多模态推理请求。"""

    system_prompt: str
    task_prompt: str
    frames: list[Image.Image]
    audio: np.ndarray | None = None
    use_audio: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty.")
        if not self.task_prompt.strip():
            raise ValueError("task_prompt must not be empty.")
        if not self.frames:
            raise ValueError("frames must not be empty.")

    @property
    def has_audio(self) -> bool:
        """返回该请求是否携带可用音频。"""
        return self.audio is not None and self.use_audio

    def without_audio(self) -> "InferenceRequest":
        """返回禁用音频后的请求副本。"""
        return replace(self, audio=None, use_audio=False)

    def to_conversation(self) -> list[dict[str, Any]]:
        """转换为 Qwen/transformers 风格的 conversation。"""
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt}],
        }
        user_content: list[dict[str, Any]] = [{"type": "video", "video": self.frames}]
        if self.has_audio:
            user_content.append({"type": "audio", "audio": self.audio})
        user_content.append({"type": "text", "text": self.task_prompt})
        return [
            system_message,
            {
                "role": "user",
                "content": user_content,
            },
        ]


class InferenceBackend(ABC):
    """统一的推理后端接口。"""

    name: str = "unknown"

    @abstractmethod
    def load(self) -> None:
        """加载模型权重与处理器。"""

    @abstractmethod
    def infer(self, request: InferenceRequest) -> str:
        """执行单次推理。"""

    @abstractmethod
    def batch_infer(self, requests: list[InferenceRequest]) -> list[str]:
        """执行批量推理。"""

    @abstractmethod
    def is_loaded(self) -> bool:
        """返回模型是否已加载。"""

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """返回后端能力声明。"""


__all__ = [
    "BackendCapabilities",
    "InferenceBackend",
    "InferenceRequest",
]
