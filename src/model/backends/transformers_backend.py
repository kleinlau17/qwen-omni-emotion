"""Transformers 后端实现。"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import torch

from src.model.base import BackendCapabilities, InferenceBackend, InferenceRequest

LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_WARNING_SECONDS: float = 0.5


class TransformersQwenBackend(InferenceBackend):
    """基于 transformers 的 Qwen Omni 推理后端。"""

    name = "transformers"

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 128,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
    ) -> None:
        """初始化模型配置，不执行实际加载。"""
        self.model_path: str = model_path
        self.torch_dtype: str = torch_dtype
        self.attn_implementation: str = attn_implementation
        self.max_new_tokens: int = max_new_tokens
        self.min_pixels: int = min_pixels
        self.max_pixels: int = max_pixels
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """加载 Qwen Omni 模型与 Processor。"""
        if self.is_loaded():
            LOGGER.debug("transformers 后端已加载，跳过重复加载。")
            return

        LOGGER.info("正在导入 transformers Qwen Omni 模块 ...")
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        LOGGER.info("正在加载 transformers 权重: %s", self.model_path)
        dtype: torch.dtype = self._resolve_torch_dtype(self.torch_dtype)
        self._model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=self.attn_implementation,
        )
        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        LOGGER.info("TransformersQwenBackend 加载完成 (device=%s)", self.device)

    def infer(self, request: InferenceRequest) -> str:
        """执行单次多模态推理。"""
        results = self.batch_infer([request])
        return results[0] if results else ""

    def batch_infer(self, requests: list[InferenceRequest]) -> list[str]:
        """批量执行多模态推理。"""
        if not self.is_loaded():
            raise RuntimeError("Backend is not loaded. Please call load() before batch_infer().")
        if not requests:
            return []

        assert self._model is not None
        assert self._processor is not None

        batch_size = len(requests)
        conversations = [request.to_conversation() for request in requests]
        chat_input = conversations[0] if batch_size == 1 else conversations
        use_audio = any(request.has_audio for request in requests)

        inputs: Any = self._processor.apply_chat_template(
            chat_input,
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
        )
        inputs = inputs.to(self.device)
        input_len: int = inputs["input_ids"].shape[1]

        start_time: float = perf_counter()
        with torch.no_grad():
            text_ids: torch.Tensor = self._model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed_seconds: float = perf_counter() - start_time

        if elapsed_seconds > DEFAULT_TIMEOUT_WARNING_SECONDS:
            LOGGER.warning(
                "transformers 推理耗时超阈值：batch=%d elapsed=%.3fs threshold=%.3fs",
                batch_size,
                elapsed_seconds,
                DEFAULT_TIMEOUT_WARNING_SECONDS,
            )

        generated_ids: torch.Tensor = text_ids[:, input_len:]
        decoded: list[str] = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded

    def is_loaded(self) -> bool:
        """返回模型与 Processor 是否已加载。"""
        return self._model is not None and self._processor is not None

    def get_capabilities(self) -> BackendCapabilities:
        """返回当前后端能力。"""
        return BackendCapabilities(
            supports_video_frames=True,
            supports_audio_array=True,
            supports_batch_infer=True,
            supports_strict_json_prompting=True,
            supports_streaming=False,
        )

    @property
    def device(self) -> torch.device:
        """返回模型当前所在设备。"""
        if self._model is None:
            raise RuntimeError("Backend is not loaded. Please call load() first.")
        return self._model.device

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str) -> torch.dtype:
        """将字符串 dtype 映射为 torch.dtype。"""
        normalized: str = torch_dtype.lower().strip()
        dtype_mapping: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in dtype_mapping:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        return dtype_mapping[normalized]


__all__ = ["TransformersQwenBackend"]
