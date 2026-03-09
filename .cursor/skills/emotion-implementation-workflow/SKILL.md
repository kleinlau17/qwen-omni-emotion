---
name: emotion-implementation-workflow
description: Implementation order and workflow for the Qwen2.5-Omni emotion understanding system. Use when planning implementation steps, deciding module priority, or asking about what to implement next.
---

# 实时情绪理解系统 — 实现工作流

## 实现顺序（依赖关系决定）

按从零依赖到依赖最多的顺序实现，每完成一层都可独立测试：

```
Phase 1 (基础设施，零依赖):
  └── utils/logger.py

Phase 2 (纯逻辑层，无硬件/模型依赖):
  ├── prompts/output_schema.py    → 定义输出数据结构
  ├── prompts/system_prompt.py    → 系统角色设定
  ├── prompts/task_prompts.py     → 任务指令构建
  ├── understanding/response_parser.py  → JSON 解析（依赖 output_schema）
  └── understanding/state_tracker.py    → 状态追踪（依赖 response_parser 的数据结构）

Phase 3 (模型层，依赖 prompts 层的 conversation 格式):
  └── model/qwen_omni.py

Phase 4 (采集层，依赖硬件，可与 Phase 2-3 并行):
  ├── capture/video_capture.py
  ├── capture/audio_capture.py
  └── capture/stream_buffer.py

Phase 5 (预处理层，依赖 capture 数据格式 + Vision 框架):
  ├── preprocessing/frame_sampler.py
  └── preprocessing/roi_extractor.py

Phase 6 (管道编排，依赖所有层):
  └── pipeline/realtime_pipeline.py

Phase 7 (入口):
  ├── main.py
  └── scripts/download_model.py
```

## Phase 1: 基础设施

### utils/logger.py

提供统一的日志配置，所有模块通过 `get_logger(__name__)` 获取 logger。

```python
# 核心接口
def setup_logging(level: str = "INFO") -> None: ...
def get_logger(name: str) -> logging.Logger: ...
```

要点：
- 格式: `[时间] [级别] [模块名] 消息`
- 支持从配置文件读取 log_level
- 禁止用 print，全部走 logging

## Phase 2: 纯逻辑层

### prompts/output_schema.py

定义模型输出的 JSON Schema 和对应的 dataclass。这是 prompts 和 understanding 的共享契约。

```python
# 核心数据结构
@dataclass
class EmotionResult:
    person_id: str
    primary_emotion: str        # 主要情绪
    emotion_intensity: float    # 强度 0-1
    secondary_emotion: str | None
    confidence: float           # 模型信心 0-1
    description: str            # 自然语言描述

@dataclass
class AtmosphereResult:
    overall_mood: str           # 群体氛围
    tension_level: float        # 紧张度 0-1
    engagement_level: float     # 参与度 0-1
    individual_emotions: list[EmotionResult]
    description: str

# JSON Schema 字符串（嵌入 prompt 约束模型输出格式）
SINGLE_PERSON_SCHEMA: str
MULTI_PERSON_SCHEMA: str
```

### prompts/system_prompt.py

定义模型的基础角色设定。

```python
def build_system_prompt() -> dict:
    """返回 system role 消息"""
    return {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT_TEXT}]
    }
```

要点：
- 角色定义为情绪分析专家
- 约束输出语言、格式、分析维度
- 不要用 Qwen 官方的音频输出 system prompt（本项目不需要语音输出）

### prompts/task_prompts.py

根据场景构建不同的任务 prompt。

```python
def build_single_person_prompt() -> str: ...
def build_multi_person_prompt(person_count: int) -> str: ...
def build_conversation(
    system_prompt: dict,
    task_prompt: str,
    frames: list[Image],
    audio: np.ndarray | None = None,
) -> list[dict]: ...
```

要点：
- 输出为 transformers conversation 格式
- video 字段接受 `list[PIL.Image]`
- audio 字段接受 `np.ndarray`

### understanding/response_parser.py

解析模型的 JSON 文本输出。

```python
def parse_emotion_response(raw_text: str) -> EmotionResult | None: ...
def parse_atmosphere_response(raw_text: str) -> AtmosphereResult | None: ...
```

要点：
- 容错处理：模型输出可能不是严格 JSON
- 尝试提取 JSON 子串（模型可能在 JSON 前后加文字）
- 解析失败时记录原始文本并返回 None

### understanding/state_tracker.py

跨推理窗口的情绪状态追踪。

```python
class EmotionStateTracker:
    def update(self, result: EmotionResult | AtmosphereResult, timestamp: float) -> None: ...
    def get_trend(self, person_id: str, window_count: int = 5) -> list[EmotionResult]: ...
    def detect_change(self, person_id: str) -> bool: ...
```

## Phase 3: 模型层

### model/qwen_omni.py

封装模型加载和推理。参考 `qwen-omni-inference` skill 获取 API 细节。

```python
class QwenOmniModel:
    def __init__(self, config: dict) -> None: ...
    def load(self) -> None: ...
    def infer(self, conversation: list[dict], max_new_tokens: int = 512) -> str: ...
```

要点：
- 使用 `Qwen2_5OmniThinkerForConditionalGeneration`（仅文本输出）
- `attn_implementation="sdpa"`（MPS 兼容）
- 推理用 `torch.no_grad()`
- 超时机制：如推理超过延迟预算则记录警告

## Phase 4: 采集层

参考 `avfoundation-capture` skill 获取 AVFoundation API 细节。

### capture/video_capture.py

```python
class VideoCapture:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_frame_callback(self, callback: Callable) -> None: ...
```

### capture/audio_capture.py

```python
class AudioCapture:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_audio_callback(self, callback: Callable) -> None: ...
```

### capture/stream_buffer.py

核心缓冲组件，解耦采集与推理：

```python
@dataclass
class InferenceWindow:
    frames: list[tuple[np.ndarray, float]]   # (帧, 时间戳)
    audio_chunks: list[tuple[np.ndarray, float]]  # (音频块, 时间戳)
    start_ts: float
    end_ts: float

class StreamBuffer:
    def push_frame(self, frame: np.ndarray, timestamp: float) -> None: ...
    def push_audio(self, chunk: np.ndarray, timestamp: float) -> None: ...
    def get_window(self) -> InferenceWindow | None: ...
```

要点：
- 线程安全（采集回调在后台线程，推理在主线程）
- 固定窗口大小（如 1.5s）
- 限制缓冲窗口数防止内存膨胀

## Phase 5: 预处理层

### preprocessing/frame_sampler.py

```python
class FrameSampler:
    def sample(self, frames: list[tuple[np.ndarray, float]], max_frames: int = 4) -> list[np.ndarray]: ...
```

### preprocessing/roi_extractor.py

参考 `macos-vision-detection` skill 获取 Vision 框架 API 细节。

```python
class ROIExtractor:
    def extract(self, frame: np.ndarray) -> list[np.ndarray]: ...
```

## Phase 6: 管道编排

### pipeline/realtime_pipeline.py

串联所有层，实现推理循环：

```python
class RealtimePipeline:
    def __init__(self, config: dict) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def _inference_loop(self) -> None:
        # 1. buffer.get_window()
        # 2. frame_sampler.sample()
        # 3. roi_extractor.extract()  (if multi-person)
        # 4. prompts.build_conversation()
        # 5. model.infer()
        # 6. parser.parse()
        # 7. tracker.update()
```

## 测试策略

| Phase | 测试方式 | 依赖 |
|-------|---------|------|
| Phase 1 | 单元测试 | 无 |
| Phase 2 | 单元测试（mock 数据） | 无 |
| Phase 3 | 集成测试（需模型权重） | 模型下载 |
| Phase 4 | 硬件测试（需摄像头/麦克风） | 设备权限 |
| Phase 5 | 单元测试（静态图片）+ Vision 框架 | macOS |
| Phase 6 | 端到端测试 | 全部 |

## 并行开发建议

- Phase 2 和 Phase 4 可并行（互不依赖）
- Phase 3 需要 Phase 2 的 conversation 格式定义
- Phase 5 需要 Phase 4 的数据格式定义
- Phase 6 最后实现
