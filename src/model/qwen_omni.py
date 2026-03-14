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
from copy import deepcopy
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
        do_sample: bool = True,
        temperature: float = 0.35,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        torch_compile: bool = False,
    ) -> None:
        """初始化模型配置，不执行实际加载。"""
        self.model_path: str = model_path
        self.torch_dtype: str = torch_dtype
        self.attn_implementation: str = attn_implementation
        self.max_new_tokens: int = max_new_tokens
        self.do_sample: bool = do_sample
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.repetition_penalty: float = repetition_penalty
        self.min_pixels: int = min_pixels
        self.max_pixels: int = max_pixels
        self._torch_compile: bool = torch_compile
        self._model: Any = None
        self._processor: Any = None
        self._audio_input_disabled: bool = False
        self._audio_disable_notice_emitted: bool = False

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
        _LOGGER.info(
            "生成参数: do_sample=%s temperature=%.2f top_p=%.2f repetition_penalty=%.2f max_new_tokens=%d",
            str(self.do_sample),
            self.temperature,
            self.top_p,
            self.repetition_penalty,
            self.max_new_tokens,
        )
        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            use_fast=False,
        )

        if self._torch_compile:
            try:
                _LOGGER.info("正在 torch.compile() 模型 ...")
                self._model = torch.compile(self._model, mode="reduce-overhead")
                _LOGGER.info("torch.compile() 成功")
            except Exception:
                _LOGGER.warning("torch.compile() 失败，回退到 eager 模式", exc_info=True)

        self._model.eval()
        _LOGGER.info("QwenOmniModel 加载完成 (device=%s)", self.device)

    def infer(self, conversation: list[dict], use_audio_in_video: bool = True) -> str:
        """执行一次多模态推理并返回文本结果。"""
        results = self.batch_infer(
            conversations=[conversation],
            use_audio_in_video=use_audio_in_video,
        )
        return results[0] if results else ""

    def batch_infer(
        self,
        conversations: list[list[dict]],
        use_audio_in_video: bool = True,
    ) -> list[str]:
        """批量执行多模态推理，一次 forward pass 处理多个 conversation。

        Args:
            conversations: N 个 conversation，每个是 ``[system_msg, user_msg]`` 格式。
            use_audio_in_video: 是否将视频中的音频送入模型。

        Returns:
            长度为 N 的文本结果列表，与输入顺序对应。
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Please call load() before batch_infer().")
        if not conversations:
            return []

        assert self._model is not None
        assert self._processor is not None

        effective_use_audio = bool(use_audio_in_video and not self._audio_input_disabled)
        normalized_conversations = conversations
        if use_audio_in_video and not effective_use_audio:
            normalized_conversations = [
                self._strip_audio_content(conv) for conv in conversations
            ]
            if not self._audio_disable_notice_emitted:
                _LOGGER.warning(
                    "音频输入已禁用（先前模板构建失败），后续将使用纯视频文本推理。"
                )
                self._audio_disable_notice_emitted = True

        if effective_use_audio and len(conversations) > 1:
            audio_flags = [
                self._conversation_has_audio(conv) for conv in normalized_conversations
            ]
            if any(audio_flags) and not all(audio_flags):
                _LOGGER.warning(
                    "批次内音频输入不一致，改为逐条推理以避免 processor 音频占位符错误。"
                )
                outputs: list[str] = []
                for conv, has_audio in zip(normalized_conversations, audio_flags):
                    outputs.extend(
                        self.batch_infer(
                            conversations=[conv],
                            use_audio_in_video=bool(effective_use_audio and has_audio),
                        )
                    )
                return outputs

        batch_size = len(conversations)
        chat_input = (
            normalized_conversations[0]
            if batch_size == 1
            else normalized_conversations
        )

        t0: float = perf_counter()
        try:
            inputs: Any = self._processor.apply_chat_template(
                chat_input,
                load_audio_from_video=False,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=effective_use_audio,
            )
        except StopIteration:
            if not effective_use_audio:
                raise
            self._audio_input_disabled = True
            _LOGGER.warning(
                "Processor 音频占位符构建失败，已切换为纯视频文本推理。"
            )
            sanitized = [self._strip_audio_content(conv) for conv in normalized_conversations]
            chat_input = sanitized[0] if batch_size == 1 else sanitized
            inputs = self._processor.apply_chat_template(
                chat_input,
                load_audio_from_video=False,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            effective_use_audio = False
        inputs = inputs.to(self.device)
        input_len: int = inputs["input_ids"].shape[1]
        t_preprocess: float = perf_counter()

        with torch.inference_mode():
            generate_kwargs: dict[str, Any] = {
                "use_audio_in_video": effective_use_audio,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "use_cache": True,
                "repetition_penalty": self.repetition_penalty,
            }
            if self.do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p
            text_ids = self._model.generate(
                **inputs,
                **generate_kwargs,
            )
        t_generate: float = perf_counter()

        generated_ids: torch.Tensor = text_ids[:, input_len:]
        gen_len: int = generated_ids.shape[1]
        decoded: list[str] = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        t_decode: float = perf_counter()

        prep_ms = (t_preprocess - t0) * 1000.0
        gen_ms = (t_generate - t_preprocess) * 1000.0
        dec_ms = (t_decode - t_generate) * 1000.0
        total_ms = (t_decode - t0) * 1000.0
        tok_per_s = gen_len / max((t_generate - t_preprocess), 1e-6)

        _LOGGER.info(
            "推理明细: batch=%d input_tok=%d gen_tok=%d | "
            "tokenize=%.0fms generate=%.0fms decode=%.0fms total=%.0fms | "
            "%.1f tok/s audio=%s",
            batch_size, input_len, gen_len,
            prep_ms, gen_ms, dec_ms, total_ms,
            tok_per_s, str(effective_use_audio),
        )
        return decoded

    @staticmethod
    def _conversation_has_audio(conversation: list[dict[str, Any]]) -> bool:
        """判断会话中是否包含显式音频输入。"""
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "audio":
                    return True
        return False

    @staticmethod
    def _strip_audio_content(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """移除会话中的显式音频内容，便于纯视频文本降级推理。"""
        cloned: list[dict[str, Any]] = deepcopy(conversation)
        for message in cloned:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            message["content"] = [
                item
                for item in content
                if not (isinstance(item, dict) and item.get("type") == "audio")
            ]
        return cloned

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
