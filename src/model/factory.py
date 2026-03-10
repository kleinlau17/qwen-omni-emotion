"""模型后端工厂与路由。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.model.backends import MLXQwen3OmniBackend, TransformersQwenBackend
from src.model.base import BackendCapabilities, InferenceBackend, InferenceRequest

LOGGER = logging.getLogger(__name__)


class BackendRouter(InferenceBackend):
    """在主后端与回退后端之间做能力路由。"""

    name = "router"

    def __init__(
        self,
        primary: InferenceBackend,
        fallback: InferenceBackend | None,
        policy: dict[str, str] | None = None,
    ) -> None:
        """初始化后端路由器。"""
        self.primary = primary
        self.fallback = fallback
        self.policy = policy or {}
        self._last_backend_name = primary.name

    def load(self) -> None:
        """优先加载主后端，失败时按策略回退。"""
        try:
            self.primary.load()
            self._last_backend_name = self.primary.name
        except Exception:
            if self.fallback is None or self.policy.get("on_backend_error", "fallback") != "fallback":
                raise
            LOGGER.exception("主后端加载失败，切换到回退后端: %s", self.fallback.name)
            self.fallback.load()
            self._last_backend_name = self.fallback.name

    def infer(self, request: InferenceRequest) -> str:
        """单次推理。"""
        return self.batch_infer([request])[0]

    def batch_infer(self, requests: list[InferenceRequest]) -> list[str]:
        """根据能力策略选择后端。"""
        if not requests:
            return []
        backend, adapted_requests = self._select_backend(requests)
        if not backend.is_loaded():
            backend.load()
        self._last_backend_name = backend.name
        return backend.batch_infer(adapted_requests)

    def is_loaded(self) -> bool:
        """任一已加载后端可视为已就绪。"""
        return self.primary.is_loaded() or (self.fallback is not None and self.fallback.is_loaded())

    def get_capabilities(self) -> BackendCapabilities:
        """返回主后端能力，用于诊断。"""
        return self.primary.get_capabilities()

    @property
    def active_backend_name(self) -> str:
        """返回最近一次实际使用的后端名称。"""
        return self._last_backend_name

    def _select_backend(
        self,
        requests: list[InferenceRequest],
    ) -> tuple[InferenceBackend, list[InferenceRequest]]:
        primary_caps = self.primary.get_capabilities()
        need_audio = any(request.has_audio for request in requests)
        need_batch = len(requests) > 1

        if need_audio and not primary_caps.supports_audio_array:
            action = self.policy.get("on_unsupported_audio", "fallback")
            if action == "disable":
                LOGGER.warning("主后端不支持音频数组输入，已降级为仅视频+文本模式。")
                return self.primary, [request.without_audio() for request in requests]
            if action == "fallback" and self.fallback is not None:
                LOGGER.info("主后端不支持音频数组输入，使用回退后端: %s", self.fallback.name)
                return self.fallback, requests
            raise RuntimeError("Primary backend does not support audio arrays.")

        if need_batch and not primary_caps.supports_batch_infer:
            action = self.policy.get("on_unsupported_batch", "batch1")
            if action == "batch1":
                LOGGER.warning("主后端不支持原生 batch，改为顺序逐条推理。")
                return self.primary, requests
            if action == "fallback" and self.fallback is not None:
                LOGGER.info("主后端不支持 batch 推理，使用回退后端: %s", self.fallback.name)
                return self.fallback, requests
            raise RuntimeError("Primary backend does not support batch inference.")

        return self.primary, requests


def create_inference_backend(config: dict[str, Any]) -> InferenceBackend:
    """按配置创建模型后端实例。"""
    model_cfg = config.get("model", {})
    inference_cfg = config.get("inference", {})
    mlx_cfg = config.get("mlx", {})
    transformers_cfg = config.get("transformers", {})
    policy_cfg = config.get("capability_policy", {})

    backend_name = str(model_cfg.get("backend", "transformers")).strip().lower()
    fallback_name = str(model_cfg.get("fallback_backend", "")).strip().lower()
    model_path = _resolve_model_path(model_cfg)

    primary = _build_backend(
        backend_name=backend_name,
        model_path=model_path,
        inference_cfg=inference_cfg,
        backend_cfg=mlx_cfg if backend_name == "mlx" else transformers_cfg,
    )
    fallback = None
    if fallback_name:
        fallback_path = _resolve_model_path(model_cfg, fallback=True)
        fallback = _build_backend(
            backend_name=fallback_name,
            model_path=fallback_path,
            inference_cfg=inference_cfg,
            backend_cfg=transformers_cfg if fallback_name == "transformers" else mlx_cfg,
        )
    if fallback is None:
        return primary
    return BackendRouter(primary=primary, fallback=fallback, policy=policy_cfg)


def _build_backend(
    backend_name: str,
    model_path: str,
    inference_cfg: dict[str, Any],
    backend_cfg: dict[str, Any],
) -> InferenceBackend:
    if backend_name == "transformers":
        return TransformersQwenBackend(
            model_path=model_path,
            torch_dtype=str(
                backend_cfg.get("torch_dtype", inference_cfg.get("torch_dtype", "bfloat16"))
            ),
            attn_implementation=str(
                backend_cfg.get(
                    "attn_implementation",
                    inference_cfg.get("attn_implementation", "sdpa"),
                )
            ),
            max_new_tokens=int(inference_cfg.get("max_new_tokens", 128)),
            min_pixels=int(inference_cfg.get("min_pixels", 128 * 28 * 28)),
            max_pixels=int(inference_cfg.get("max_pixels", 256 * 28 * 28)),
        )
    if backend_name == "mlx":
        return MLXQwen3OmniBackend(
            model_path=model_path,
            max_new_tokens=int(inference_cfg.get("max_new_tokens", 128)),
            trust_remote_code=bool(backend_cfg.get("trust_remote_code", True)),
            bridge_audio_via_temp_file=bool(backend_cfg.get("bridge_audio_via_temp_file", False)),
        )
    raise ValueError(f"Unsupported backend: {backend_name}")


def _resolve_model_path(model_cfg: dict[str, Any], fallback: bool = False) -> str:
    """解析主后端或回退后端的模型路径。"""
    if fallback:
        local_path = str(model_cfg.get("fallback_local_path") or "").strip()
        repo_id = str(
            model_cfg.get("fallback_repo_id")
            or model_cfg.get("repo_id")
            or ""
        ).strip()
    else:
        local_path = str(model_cfg.get("local_path") or "").strip()
        repo_id = str(model_cfg.get("repo_id") or "").strip()

    if local_path:
        expanded = Path(local_path).expanduser()
        if expanded.exists():
            return str(expanded)
    return repo_id


__all__ = ["BackendRouter", "create_inference_backend"]
