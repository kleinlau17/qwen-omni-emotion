---
name: qwen-omni-inference
description: Qwen2.5-Omni-3B model inference API reference for transformers. Use when implementing model loading, processor usage, conversation format, multimodal input handling, or generate() calls in the model layer.
---

# Qwen2.5-Omni-3B 推理 API 参考

## 核心类

本项目仅需文本输出，使用 Thinker 模型（节省 ~2GB 显存）：

```python
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
```

| 类名 | 用途 |
|------|------|
| `Qwen2_5OmniThinkerForConditionalGeneration` | 仅文本输出（本项目使用） |
| `Qwen2_5OmniForConditionalGeneration` | 文本 + 音频输出 |
| `Qwen2_5OmniProcessor` | 统一的多模态输入处理器 |

## 模型加载

```python
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # MPS 兼容，不要用 flash_attention_2
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
```

**MPS 注意事项**：
- `attn_implementation` 必须用 `"sdpa"`，flash_attention_2 需要 NVIDIA GPU
- `device_map="auto"` 会自动选择 MPS
- `torch_dtype=torch.bfloat16` M3 Ultra 原生支持

## Conversation 格式

模型输入为 OpenAI 风格的 conversation 列表：

```python
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "你是情绪分析专家..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "分析这段视频中人物的情绪状态"}
        ]
    }
]
```

### 支持的 content 类型

| type | 字段 | 值类型 | 说明 |
|------|------|--------|------|
| `"text"` | `text` | `str` | 文本内容 |
| `"video"` | `video` | `str` 或 `list[PIL.Image]` | 视频路径或帧列表 |
| `"audio"` | `audio` | `str` 或 `np.ndarray` | 音频路径或数组 |
| `"image"` | `image` | `str` 或 `PIL.Image` | 图片路径或对象 |

### 传入实时帧（本项目核心用法）

实时采集场景下，视频帧以 PIL.Image 列表形式传入：

```python
from PIL import Image
import numpy as np

# frames: list[np.ndarray]，每帧 shape (H, W, 3)，uint8，RGB
pil_frames = [Image.fromarray(frame) for frame in frames]

conversation = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": pil_frames},
            {"type": "text", "text": task_prompt},
        ]
    }
]
```

### 传入音频数据

音频以 numpy 数组传入时，需配合 `sampling_rate`：

```python
# audio_segment: np.ndarray, shape (samples,), float32, 16kHz
conversation = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": pil_frames},
            {"type": "audio", "audio": audio_segment},
            {"type": "text", "text": task_prompt},
        ]
    }
]
```

## 输入处理（两种 API 风格）

### 风格 A：processor.apply_chat_template 一体化（推荐）

```python
inputs = processor.apply_chat_template(
    conversation,
    load_audio_from_video=True,    # 是否从视频文件提取音频
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=True,       # 视频音轨是否送入模型
    # 可选：控制视频帧采样率
    # fps=2,
).to(model.device)
```

### 风格 B：分步处理（需要更多控制时）

```python
from qwen_omni_utils import process_mm_info

# Step 1: 生成文本 prompt
text_prompt = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)

# Step 2: 提取多模态数据
audios, images, videos = process_mm_info(
    conversation, use_audio_in_video=True
)

# Step 3: 编码为模型输入
inputs = processor(
    text=text_prompt,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=True,
)
inputs = inputs.to(model.device).to(model.dtype)
```

## 推理调用

```python
with torch.no_grad():
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=True,
        max_new_tokens=512,
    )

text_output = processor.batch_decode(
    text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
```

### generate() 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_audio_in_video` | `False` | 是否使用视频音轨（本项目传 `True`） |
| `max_new_tokens` | - | 最大生成 token 数（建议 512） |
| `return_audio` | `False` | 是否返回语音（Thinker 模型不支持） |

## 显存控制

调整 processor 的分辨率上限可降低显存占用：

```python
processor = Qwen2_5OmniProcessor.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    min_pixels=128 * 28 * 28,
    max_pixels=768 * 28 * 28,
)
```

## 本项目推理流程模板

完整推理 pipeline 示例（用于 `src/model/qwen_omni.py`）：

```python
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from PIL import Image
import numpy as np

class QwenOmniModel:
    def __init__(self, model_path: str):
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    def infer(
        self,
        conversation: list[dict],
        use_audio_in_video: bool = True,
        max_new_tokens: int = 512,
    ) -> str:
        inputs = self.processor.apply_chat_template(
            conversation,
            load_audio_from_video=False,  # 实时流不从文件加载
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        ).to(self.model.device)

        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=max_new_tokens,
            )

        return self.processor.batch_decode(
            text_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
```

## 常见陷阱

1. **MPS 不支持 flash_attention_2** → 必须用 `attn_implementation="sdpa"`
2. **Thinker 模型不能 return_audio** → 设 `return_audio=False` 或不传
3. **实时帧传入方式** → `"video"` 字段接受 `list[PIL.Image]`，不只是文件路径
4. **音频采样率** → processor 内部会自动重采样，传入 16kHz float32 即可
5. **batch_decode 的 [0]** → generate 返回的是 batch 维度，取第一个
