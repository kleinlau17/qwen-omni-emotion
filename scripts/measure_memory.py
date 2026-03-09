"""测量 Qwen2.5-Omni-3B 在 MPS 上的实际显存占用。"""
import gc
import sys
import torch
import numpy as np
from PIL import Image


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    return f"{n / (1 << 10):.2f} KB"


def report(label: str) -> None:
    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()
    print(f"[{label}]")
    print(f"  MPS allocated (PyTorch):  {fmt_bytes(allocated)}")
    print(f"  MPS driver (系统分配):     {fmt_bytes(driver)}")
    print()


MODEL_PATH = "/Users/tonlyai/Documents/Qwen2.5/Omni3B/~/.cache/modelscope/Qwen/Qwen2.5-Omni-3B"

print("=" * 60)
print("Qwen2.5-Omni-3B MPS 显存测量")
print("=" * 60)

# 基线
torch.mps.empty_cache()
gc.collect()
report("基线 (无模型)")

# 加载模型
print("正在加载模型 (bfloat16) ...")
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
report("模型加载后 (权重 on MPS)")

# 加载 Processor
processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)
report("Processor 加载后")

# 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())
total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"模型参数量: {total_params / 1e9:.2f}B")
print(f"参数占用显存 (理论): {fmt_bytes(total_bytes)}")
print()

# 构造模拟推理输入 (1 张图片 + 文本)
print("正在构造推理输入 (模拟 1 张 640x480 视频帧) ...")
dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": "Describe the image briefly."},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    load_audio_from_video=False,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=False,
)
inputs = inputs.to(model.device)
input_len = inputs["input_ids"].shape[1]
print(f"输入 token 数: {input_len}")
report("推理输入准备后")

# 执行推理
print("正在执行推理 (max_new_tokens=128) ...")
with torch.no_grad():
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=False,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )

report("推理完成后 (含 KV cache + 中间激活)")

generated_ids = text_ids[:, input_len:]
decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f"生成 token 数: {generated_ids.shape[1]}")
print(f"输出: {decoded[0][:100]}...")
print()

# 清理
del text_ids, inputs
torch.mps.empty_cache()
gc.collect()
report("推理后清理")

print("=" * 60)
print("测量完成")
print("=" * 60)
