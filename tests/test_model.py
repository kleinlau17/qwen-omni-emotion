"""模型层测试"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from src.model.base import InferenceRequest
from src.model.backends.mlx_backend import MLXQwen3OmniBackend
from src.model.backends.transformers_backend import TransformersQwenBackend
from src.model.factory import BackendRouter, create_inference_backend


def _make_request(num_frames: int = 1) -> InferenceRequest:
    frames = [
        Image.new("RGB", (16, 16), color=(i * 60, 100, 50))
        for i in range(num_frames)
    ]
    return InferenceRequest(
        system_prompt="你是情绪分析助手",
        task_prompt="分析情绪",
        frames=frames,
    )


def test_transformers_backend_init_does_not_load() -> None:
    """初始化时不应加载模型和 processor。"""
    model = TransformersQwenBackend(model_path="Qwen/Qwen2.5-Omni-3B")
    assert model.model_path == "Qwen/Qwen2.5-Omni-3B"
    assert model.max_new_tokens == 128
    assert model.is_loaded() is False
    capabilities = model.get_capabilities()
    assert capabilities.supports_audio_array is True
    assert capabilities.supports_batch_infer is True


def test_is_loaded_initially_false() -> None:
    """未调用 load 前 is_loaded 应为 False。"""
    model = TransformersQwenBackend(model_path="dummy/path")
    assert model.is_loaded() is False


def test_infer_with_mocked_transformers_backend() -> None:
    """infer 应按约定处理输入、调用 generate 并返回解码文本。"""
    backend = TransformersQwenBackend(model_path="dummy/path")

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ['{"emotion":"happy"}']

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 101, 102, 103]])

    backend._processor = mock_processor
    backend._model = mock_model

    request = _make_request()
    output = backend.infer(request)

    assert output == '{"emotion":"happy"}'
    mock_processor.apply_chat_template.assert_called_once_with(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是情绪分析助手"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": request.frames,
                    },
                    {"type": "text", "text": "分析情绪"},
                ],
            },
        ],
        load_audio_from_video=False,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    generate_kwargs = mock_model.generate.call_args.kwargs
    assert torch.equal(generate_kwargs["input_ids"], torch.tensor([[1, 2, 3]]))
    assert generate_kwargs["use_audio_in_video"] is False
    assert generate_kwargs["max_new_tokens"] == 128
    batch_decode_args = mock_processor.batch_decode.call_args.args
    batch_decode_kwargs = mock_processor.batch_decode.call_args.kwargs
    assert len(batch_decode_args) == 1
    assert torch.equal(batch_decode_args[0], torch.tensor([[101, 102, 103]]))
    assert batch_decode_kwargs["skip_special_tokens"] is True
    assert batch_decode_kwargs["clean_up_tokenization_spaces"] is False


def test_mlx_backend_init_does_not_load() -> None:
    """初始化时不应加载模型和 processor。"""
    backend = MLXQwen3OmniBackend(model_path="mlx-community/test-model")
    assert backend.model_path == "mlx-community/test-model"
    assert backend.max_new_tokens == 128
    assert backend.is_loaded() is False
    caps = backend.get_capabilities()
    assert caps.supports_video_frames is True
    assert caps.supports_audio_array is False
    assert caps.supports_batch_infer is False


def test_mlx_backend_build_conversation_multi_frame() -> None:
    """_build_conversation 应为每帧生成独立的 image 条目。"""
    request = _make_request(num_frames=4)
    conv = MLXQwen3OmniBackend._build_conversation(request)

    assert len(conv) == 2
    assert conv[0]["role"] == "system"
    assert conv[1]["role"] == "user"

    user_content = conv[1]["content"]
    image_items = [item for item in user_content if item["type"] == "image"]
    text_items = [item for item in user_content if item["type"] == "text"]

    assert len(image_items) == 4
    assert len(text_items) == 1
    assert text_items[0]["text"] == "分析情绪"


def test_mlx_backend_infer_calls_model_generate() -> None:
    """infer 应调用 model.generate() 而非 mlx_vlm.generate()，并正确解码。"""
    import mlx.core as mx

    backend = MLXQwen3OmniBackend(model_path="test/path")

    seq = mx.array([[1, 2, 3, 101, 102, 103]])
    thinker_result = type("obj", (object,), {"sequences": seq})()

    mock_model = MagicMock()
    mock_model.generate.return_value = (thinker_result, None)
    mock_model.config = MagicMock()
    mock_model.config.eos_token_id = 151645

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<prompt_text>"
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.randn(1, 3, 64, 64),
        "image_grid_thw": torch.tensor([[1, 8, 8]]),
    }
    mock_processor.tokenizer.decode.return_value = '{"primary_emotion":"happy"}'

    backend._model = mock_model
    backend._processor = mock_processor

    request = _make_request(num_frames=4)
    output = backend.infer(request)

    assert output == '{"primary_emotion":"happy"}'
    mock_model.generate.assert_called_once()
    gen_kwargs = mock_model.generate.call_args.kwargs
    assert gen_kwargs["return_audio"] is False
    assert gen_kwargs["thinker_max_new_tokens"] == 128
    assert gen_kwargs["thinker_eos_token_id"] == 151645
    assert "pixel_values" in gen_kwargs
    assert "image_grid_thw" in gen_kwargs


def test_mlx_backend_multi_frame_conversation_structure() -> None:
    """多帧时 conversation 的 image 条目数应等于帧数。"""
    for n in (1, 4, 8):
        request = _make_request(num_frames=n)
        conv = MLXQwen3OmniBackend._build_conversation(request)
        user_content = conv[1]["content"]
        image_count = sum(1 for item in user_content if item["type"] == "image")
        assert image_count == n, f"Expected {n} images, got {image_count}"


def test_backend_router_falls_back_when_audio_unsupported() -> None:
    """主后端不支持音频时应自动切换到 fallback。"""
    primary = MagicMock()
    primary.name = "mlx"
    primary.get_capabilities.return_value.supports_audio_array = False
    primary.get_capabilities.return_value.supports_batch_infer = False
    primary.is_loaded.return_value = True

    fallback = MagicMock()
    fallback.name = "transformers"
    fallback.is_loaded.return_value = True
    fallback.batch_infer.return_value = ['{"emotion":"happy"}']

    router = BackendRouter(
        primary=primary,
        fallback=fallback,
        policy={"on_unsupported_audio": "fallback", "on_unsupported_batch": "batch1"},
    )

    request = InferenceRequest(
        system_prompt="sys",
        task_prompt="task",
        frames=[Image.new("RGB", (8, 8))],
        audio=np.zeros(16000, dtype=np.float32),
    )
    outputs = router.batch_infer([request])

    assert outputs == ['{"emotion":"happy"}']
    fallback.batch_infer.assert_called_once()
    primary.batch_infer.assert_not_called()


def test_create_inference_backend_returns_router_for_dual_backend() -> None:
    """配置主后端和 fallback 时应返回路由器。"""
    config = {
        "model": {
            "backend": "mlx",
            "local_path": "/nonexistent/path/to/mlx-model",
            "repo_id": "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
            "fallback_backend": "transformers",
            "fallback_local_path": "/nonexistent/path/to/tf-model",
            "fallback_repo_id": "Qwen/Qwen2.5-Omni-3B",
        },
        "inference": {"max_new_tokens": 64},
        "mlx": {"trust_remote_code": True},
        "transformers": {"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    }
    backend = create_inference_backend(config)

    assert isinstance(backend, BackendRouter)
    assert backend.primary.model_path == "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
    assert backend.fallback is not None
    assert backend.fallback.model_path == "Qwen/Qwen2.5-Omni-3B"
