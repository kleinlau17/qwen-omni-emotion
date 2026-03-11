"""模型层抽象与后端工厂。"""

from src.model.base import BackendCapabilities, InferenceBackend, InferenceRequest
from src.model.factory import BackendRouter, create_inference_backend

__all__ = [
    "BackendCapabilities",
    "BackendRouter",
    "InferenceBackend",
    "InferenceRequest",
    "create_inference_backend",
]
