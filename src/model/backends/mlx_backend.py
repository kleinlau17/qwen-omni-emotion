"""MLX 后端实现 — 直接调用 Model.generate() 实现多帧序列推理。

调用链路：
  processor.apply_chat_template → processor(text, images) → PyTorch tensors
  → mx.array → model.generate(input_ids, return_audio=False,
      pixel_values=..., image_grid_thw=..., thinker_max_new_tokens=N)
  → thinker_result.sequences → tokenizer.decode

关键点：mlx_vlm.generate() 高级 API 对 qwen3_omni_moe 不兼容
（它把整个 Model 传给 generate_step，但 Model 没有 language_model 属性），
必须调用 Model 自身的 generate() 方法，它会正确地将 self.thinker 传给
generate_step，而 Thinker 拥有 language_model 子模块。
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from src.model.base import BackendCapabilities, InferenceBackend, InferenceRequest

LOGGER = logging.getLogger(__name__)

_THINKER_EOS_TOKEN_ID = 151645  # <|im_end|>


class MLXQwen3OmniBackend(InferenceBackend):
    """直接调用 Model.generate() 的 Qwen3 Omni 多帧推理后端。"""

    name = "mlx"

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        trust_remote_code: bool = True,
        bridge_audio_via_temp_file: bool = False,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.bridge_audio_via_temp_file = bridge_audio_via_temp_file
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        if self.is_loaded():
            LOGGER.debug("MLX 后端已加载，跳过重复加载。")
            return

        LOGGER.info("正在导入 mlx_vlm 模块 ...")
        from mlx_vlm import load

        LOGGER.info("正在加载 MLX 权重: %s", self.model_path)
        t0 = perf_counter()
        self._model, self._processor = load(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self._patch_eos_token_id()
        LOGGER.info(
            "MLXQwen3OmniBackend 加载完成 (%.1fs)",
            perf_counter() - t0,
        )

    def infer(self, request: InferenceRequest) -> str:
        results = self.batch_infer([request])
        return results[0] if results else ""

    def batch_infer(self, requests: list[InferenceRequest]) -> list[str]:
        if not self.is_loaded():
            raise RuntimeError("Backend is not loaded. Call load() first.")
        outputs: list[str] = []
        for request in requests:
            outputs.append(self._infer_one(request))
        return outputs

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_video_frames=True,
            supports_audio_array=self.bridge_audio_via_temp_file,
            supports_batch_infer=False,
            supports_strict_json_prompting=True,
            supports_streaming=False,
        )

    # ── 兼容性补丁 ────────────────────────────────────────────────

    def _patch_eos_token_id(self) -> None:
        """Qwen3-Omni-MoE config 缺少 eos_token_id，补丁供 generate_step 使用。"""
        assert self._model is not None
        cfg = self._model.config
        if hasattr(cfg, "eos_token_id") and cfg.eos_token_id is not None:
            return

        eos_id: int | None = None
        if hasattr(cfg, "im_end_token_id"):
            eos_id = cfg.im_end_token_id
        elif self._processor is not None:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            if hasattr(tokenizer, "eos_token_id"):
                eos_id = tokenizer.eos_token_id

        if eos_id is not None:
            cfg.eos_token_id = eos_id
            LOGGER.info("已补丁 model.config.eos_token_id = %d", eos_id)
        else:
            LOGGER.warning("无法确定 eos_token_id，generate 可能报错")

    # ── 核心推理 ──────────────────────────────────────────────────

    def _infer_one(self, request: InferenceRequest) -> str:
        import mlx.core as mx
        import torch

        assert self._model is not None
        assert self._processor is not None

        conversation = self._build_conversation(request)

        prompt = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if request.frames:
            processor_kwargs["images"] = request.frames

        inputs = self._processor(**processor_kwargs)

        input_ids = mx.array(inputs["input_ids"].numpy())
        input_len = input_ids.shape[1]

        generate_kwargs: dict[str, Any] = {
            "return_audio": False,
            "thinker_max_new_tokens": self.max_new_tokens,
            "thinker_eos_token_id": _THINKER_EOS_TOKEN_ID,
        }
        if "pixel_values" in inputs:
            generate_kwargs["pixel_values"] = mx.array(
                inputs["pixel_values"].to(torch.float32).numpy()
            )
        if "image_grid_thw" in inputs:
            generate_kwargs["image_grid_thw"] = mx.array(
                inputs["image_grid_thw"].numpy()
            )

        t0 = perf_counter()
        thinker_result, _ = self._model.generate(input_ids, **generate_kwargs)
        sequences = thinker_result.sequences
        generated_ids = sequences[0, input_len:].tolist()
        decoded = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        elapsed = perf_counter() - t0

        gen_tokens = len(generated_ids)
        LOGGER.info(
            "MLX 多帧推理完成: %d 帧, %d tokens in %.2fs (%.1f tok/s)",
            len(request.frames),
            gen_tokens,
            elapsed,
            gen_tokens / elapsed if elapsed > 0 else 0,
        )
        return decoded

    @staticmethod
    def _build_conversation(request: InferenceRequest) -> list[dict[str, Any]]:
        """构建 processor 兼容的 conversation，多帧作为多张 image 传入。

        每帧以 {"type": "image"} 插入 user content，processor 会为每张图
        生成独立的 pixel_values 和 image_grid_thw 条目，模型 Thinker 内部
        通过 vision_tower 编码后嵌入到文本 embedding 序列中。
        """
        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": request.system_prompt}],
        }
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": frame} for frame in request.frames
        ]
        user_content.append({"type": "text", "text": request.task_prompt})
        return [
            system_msg,
            {"role": "user", "content": user_content},
        ]


__all__ = ["MLXQwen3OmniBackend"]
