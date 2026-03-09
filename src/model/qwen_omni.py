"""Qwen2.5-Omni-3B 模型封装

职责：
- 通过 transformers 加载模型与 Processor
- 构建 conversation 格式的推理输入（视频帧 + 音频 + prompt）
- 调用 model.generate() 执行推理
- 返回文本输出（供 understanding 层解析）

不负责：
- 不决定"分析什么"（由 prompts 层决定）
- 不解析输出语义（由 understanding 层负责）
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import torch

_LOGGER: logging.Logger = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_WARNING_SECONDS: float = 0.5


class QwenOmniModel:
    """Qwen2.5-Omni-3B Thinker 模型封装。"""

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
        """加载 Qwen2.5-Omni Thinker 模型与 Processor。"""
        if self.is_loaded():
            _LOGGER.debug("模型已加载，跳过重复加载。")
            return

        _LOGGER.info("正在导入 transformers Qwen2.5-Omni 模块 ...")
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        _LOGGER.info("正在加载模型权重: %s", self.model_path)
        dtype: torch.dtype = self._resolve_torch_dtype(self.torch_dtype)
        self._model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=self.attn_implementation,
        )
        _LOGGER.info("模型权重加载完成，正在加载 Processor ...")
        _LOGGER.info(
            "Processor 像素限制: min_pixels=%d, max_pixels=%d",
            self.min_pixels,
            self.max_pixels,
        )
        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        _LOGGER.info("QwenOmniModel 加载完成 (device=%s)", self.device)

    def infer(self, conversation: list[dict], use_audio_in_video: bool = True) -> str:
        """执行一次多模态推理并返回文本结果。"""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Please call load() before infer().")

        assert self._model is not None
        assert self._processor is not None

        inputs: Any = self._processor.apply_chat_template(
            conversation,
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.device)
        input_len: int = inputs["input_ids"].shape[1]

        start_time: float = perf_counter()
        with torch.no_grad():
            text_ids: torch.Tensor = self._model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed_seconds: float = perf_counter() - start_time
        if elapsed_seconds > _DEFAULT_TIMEOUT_WARNING_SECONDS:
            _LOGGER.warning(
                "推理耗时超阈值：elapsed=%.3fs threshold=%.3fs",
                elapsed_seconds,
                _DEFAULT_TIMEOUT_WARNING_SECONDS,
            )

        generated_ids: torch.Tensor = text_ids[:, input_len:]
        decoded: list[str] = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    def is_loaded(self) -> bool:
        """返回模型与 Processor 是否已加载。"""
        return self._model is not None and self._processor is not None

    @property
    def device(self) -> torch.device:
        """返回模型当前所在设备。"""
        if self._model is None:
            raise RuntimeError("Model is not loaded. Please call load() first.")
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
