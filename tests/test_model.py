"""模型层测试"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from src.model.qwen_omni import QwenOmniModel


def test_qwen_omni_model_init_does_not_load() -> None:
    """初始化时不应加载模型和 processor。"""
    model = QwenOmniModel(model_path="Qwen/Qwen2.5-Omni-3B")
    assert model.model_path == "Qwen/Qwen2.5-Omni-3B"
    assert model.max_new_tokens == 128
    assert model.is_loaded() is False


def test_is_loaded_initially_false() -> None:
    """未调用 load 前 is_loaded 应为 False。"""
    model = QwenOmniModel(model_path="dummy/path")
    assert model.is_loaded() is False


def test_infer_with_mocked_model_and_processor() -> None:
    """infer 应按约定处理输入、调用 generate 并返回解码文本。"""
    qwen_model = QwenOmniModel(model_path="dummy/path")

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ['{"emotion":"happy"}']

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([[101, 102, 103]])

    qwen_model._processor = mock_processor
    qwen_model._model = mock_model

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "你是情绪分析助手"}]},
        {"role": "user", "content": [{"type": "text", "text": "分析情绪"}]},
    ]
    output = qwen_model.infer(conversation=conversation, use_audio_in_video=True)

    assert output == '{"emotion":"happy"}'
    mock_processor.apply_chat_template.assert_called_once_with(
        conversation,
        load_audio_from_video=False,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    generate_kwargs = mock_model.generate.call_args.kwargs
    assert torch.equal(generate_kwargs["input_ids"], torch.tensor([[1, 2, 3]]))
    assert generate_kwargs["use_audio_in_video"] is True
    assert generate_kwargs["max_new_tokens"] == 128
    batch_decode_args = mock_processor.batch_decode.call_args.args
    batch_decode_kwargs = mock_processor.batch_decode.call_args.kwargs
    assert len(batch_decode_args) == 1
    assert batch_decode_args[0].shape[0] == 1
    assert batch_decode_kwargs["skip_special_tokens"] is True
    assert batch_decode_kwargs["clean_up_tokenization_spaces"] is False


def test_batch_infer_audio_stop_iteration_fallback_to_no_audio() -> None:
    """音频模板构建失败时应自动降级为无音频推理。"""
    qwen_model = QwenOmniModel(model_path="dummy/path")

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.side_effect = [StopIteration(), mock_inputs]
    mock_processor.batch_decode.return_value = ['{"emotion":"neutral"}']

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([[101, 102, 103]])

    qwen_model._processor = mock_processor
    qwen_model._model = mock_model

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": []},
                {"type": "audio", "audio": [0.0]},
                {"type": "text", "text": "task"},
            ],
        },
    ]
    output = qwen_model.batch_infer(conversations=[conversation], use_audio_in_video=True)
    assert output == ['{"emotion":"neutral"}']

    calls = mock_processor.apply_chat_template.call_args_list
    assert calls[0].kwargs["use_audio_in_video"] is True
    assert calls[1].kwargs["use_audio_in_video"] is False
