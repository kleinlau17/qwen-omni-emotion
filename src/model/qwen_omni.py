"""兼容层：保留旧的 QwenOmniModel 导入路径。"""
from __future__ import annotations

from typing import Any

from src.model.backends.transformers_backend import TransformersQwenBackend

class QwenOmniModel(TransformersQwenBackend):
    """兼容旧导入路径的 transformers 后端别名。"""

    def infer(self, conversation: list[dict[str, Any]], use_audio_in_video: bool = True) -> str:
        """兼容旧签名的单次推理方法。"""
        results = self.batch_infer(
            conversations=[conversation],
            use_audio_in_video=use_audio_in_video,
        )
        return results[0] if results else ""

    def batch_infer(
        self,
        conversations: list[list[dict[str, Any]]],
        use_audio_in_video: bool = True,
    ) -> list[str]:
        """兼容旧签名的批量推理方法。"""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Please call load() before batch_infer().")
        if not conversations:
            return []

        assert self._model is not None
        assert self._processor is not None

        batch_size = len(conversations)
        chat_input = conversations[0] if batch_size == 1 else conversations
        inputs: Any = self._processor.apply_chat_template(
            chat_input,
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
        text_ids = self._model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated_ids = text_ids[:, input_len:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


__all__ = ["QwenOmniModel"]
