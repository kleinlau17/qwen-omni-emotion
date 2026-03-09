# Vibe Coding Prompt — Qwen2.5-Omni 实时情绪理解系统

> **使用方式**: 将下方 `---` 分隔的 Prompt 正文复制到 AI 编码助手中，按 Phase 逐步推进实现。
> 每个 Phase 完成后，用对应的 "Phase 完成检查" 验证质量，再进入下一 Phase。

---

## Master Prompt

```
你是一位资深 Python 工程师，正在 macOS Apple Silicon (M3 Ultra，建议 64GB+ 统一内存，开发机为 512 GB) 上实现一套基于 Qwen2.5-Omni-3B 的实时情绪理解系统。

## 项目现状

项目框架已搭建完毕 — 目录结构、配置文件、cursor rules、空壳源文件全部就绪。
你的任务是逐模块填充实现代码。所有源文件目前只有 docstring 和 `from __future__ import annotations`，没有任何业务逻辑。

## 核心链路

采集(capture) → 输入准备(preprocessing) → 提示词构建(prompts) + 模型推理(model) → 输出解析(understanding) → pipeline 编排

## 不可违背的约束

1. **端到端多模态** — 不做任何手动特征工程（不提取微表情、不分析肢体、不处理声纹），所有理解由模型完成
2. **prompt 是核心智能** — 系统的分析能力取决于 prompt 设计，代码只负责输入准备和输出解析
3. **格式转换由 Processor 完成** — 分辨率/色彩空间/音频采样率的适配由 Qwen2_5OmniProcessor 内部处理
4. **本地隐私** — 禁止网络请求上传原始音视频，所有处理在本地完成
5. **延迟预算** — 单人 <400ms，多人(4-6人) <500ms

## 技术栈

- Python 3.11+ / macOS / Apple Silicon MPS
- transformers + PyTorch (MPS 后端)
- Qwen2_5OmniThinkerForConditionalGeneration（仅文本输出，节省 ~2GB）
- attn_implementation="sdpa"（MPS 不支持 flash_attention_2）
- PyObjC + AVFoundation（视频/音频采集）
- PyObjC + macOS Vision 框架（人体/人脸检测，用于 ROI 裁剪）
- qwen-omni-utils（官方多模态输入处理）

## 编码规范

- `from __future__ import annotations` 在每个文件开头
- 函数参数和返回值必须有类型注解
- 中文注释说明业务逻辑，英文命名变量和函数
- 使用 logging 模块，禁止 print
- 导入顺序：标准库 → 第三方 → 项目内部，各组之间空一行
- PEP 8，行宽 100
- 所有公开 API 需要 docstring

## 数据格式约定

- 视频帧: numpy.ndarray, shape (H, W, 3), dtype uint8, RGB
- 音频段: numpy.ndarray, shape (samples,), dtype float32, 16kHz
- 时间戳: float, Unix epoch seconds
- 推理窗口: dataclass with fields: frames, audio, start_ts, end_ts
- 模型输入: transformers conversation 格式 (list[dict])
- 模型输出: str (JSON 格式文本)
- 解析结果: dataclass

## 并发模型

- 音视频采集在后台线程（AVFoundation dispatch queue 回调）
- 模型推理在主线程（MPS 设备要求）
- 采集与推理通过 StreamBuffer 解耦
- 共享数据需加锁（threading.Lock）

## 错误处理策略

- 采集层异常：graceful degradation，不崩溃管道
- 模型推理超时：跳过当前窗口，记录警告
- JSON 解析失败：记录原始输出，返回 None/默认值

## 配置系统

项目使用 YAML 配置文件（configs/ 目录），实现时通过参数传入配置值，不要硬编码：
- configs/default.yaml — 全局配置（日志级别、延迟预算、窗口参数）
- configs/model.yaml — 模型路径、推理参数
- configs/pipeline.yaml — 采集参数、预处理参数、缓冲参数

## 目录结构

src/
├── capture/          # video_capture.py, audio_capture.py, stream_buffer.py
├── preprocessing/    # frame_sampler.py, roi_extractor.py
├── model/            # qwen_omni.py
├── prompts/          # system_prompt.py, task_prompts.py, output_schema.py
├── understanding/    # response_parser.py, state_tracker.py
├── pipeline/         # realtime_pipeline.py
└── utils/            # logger.py
tests/                # test_*.py 对应每个模块
```

---

## Phase 1 Prompt（基础设施）

```
实现 src/utils/logger.py — 统一日志工具。

要求:
1. setup_logging(level: str = "INFO") → None
   - 配置 root logger 格式为: [%(asctime)s] [%(levelname)s] [%(name)s] %(message)s
   - 时间格式: %H:%M:%S
   - 支持从 configs/default.yaml 读取 log_level

2. get_logger(name: str) → logging.Logger
   - 返回指定名称的 logger 实例

3. 同时实现 tests/test_utils.py 的测试用例：
   - 验证 setup_logging 设置日志级别
   - 验证 get_logger 返回正确名称的 logger
   - 验证日志格式包含时间、级别、模块名

注意: 不要用 print，这个模块是所有其他模块的日志基础。
```

---

## Phase 2 Prompt（提示词与理解层 — 纯逻辑，零硬件依赖）

```
按以下顺序实现 prompts/ 和 understanding/ 层。这些模块是纯 Python 逻辑，不依赖硬件或模型。

### Step 1: src/prompts/output_schema.py

定义模型输出的结构化数据类型和 JSON Schema（这是 prompts 和 understanding 的共享契约）:

@dataclass
class EmotionResult:
    person_id: str               # 人物标识
    primary_emotion: str         # 主要情绪 (happy/sad/angry/fearful/surprised/disgusted/neutral/contemptuous)
    emotion_intensity: float     # 强度 0.0-1.0
    secondary_emotion: str | None  # 次要情绪
    confidence: float            # 模型信心 0.0-1.0
    description: str             # 自然语言描述

@dataclass
class AtmosphereResult:
    overall_mood: str            # 群体氛围
    tension_level: float         # 紧张度 0.0-1.0
    engagement_level: float      # 参与度 0.0-1.0
    individual_emotions: list[EmotionResult]
    description: str

同时提供:
- SINGLE_PERSON_SCHEMA: str — 单人分析的 JSON Schema 文本（嵌入 prompt）
- MULTI_PERSON_SCHEMA: str — 多人分析的 JSON Schema 文本
- VALID_EMOTIONS: list[str] — 合法情绪标签列表

### Step 2: src/prompts/system_prompt.py

定义模型的系统角色设定:
- build_system_prompt() → dict
- 返回 {"role": "system", "content": [{"type": "text", "text": ...}]}
- 角色: 多模态情绪分析专家
- 约束: 只输出 JSON，不输出其他文字；分析维度包括面部表情、肢体语言、语音语调
- 不要用 Qwen 官方音频输出的 system prompt，本项目只需文本输出

### Step 3: src/prompts/task_prompts.py

构建不同分析任务的 prompt 和完整 conversation:
- build_single_person_prompt() → str — 单人情绪分析指令
- build_multi_person_prompt(person_count: int) → str — 多人氛围分析指令
- build_conversation(system_prompt: dict, task_prompt: str, frames: list[Image.Image], audio: np.ndarray | None = None) → list[dict]
  - 输出为 transformers conversation 格式
  - video 字段接受 list[PIL.Image]
  - audio 字段接受 np.ndarray (可选)

### Step 4: src/understanding/response_parser.py

解析模型 JSON 文本输出:
- parse_emotion_response(raw_text: str) → EmotionResult | None
- parse_atmosphere_response(raw_text: str) → AtmosphereResult | None
- 容错: 模型输出可能不是严格 JSON，需要尝试提取 JSON 子串（用正则找 {...}）
- 解析失败: 记录原始文本 (logging.warning)，返回 None

### Step 5: src/understanding/state_tracker.py

跨推理窗口的情绪状态追踪:

class EmotionStateTracker:
    def update(self, result: EmotionResult | AtmosphereResult, timestamp: float) → None
    def get_trend(self, person_id: str, window_count: int = 5) → list[EmotionResult]
    def detect_change(self, person_id: str, threshold: float = 0.3) → bool
    def get_current_state(self, person_id: str) → EmotionResult | None

- 内部用 dict[str, deque[tuple[EmotionResult, float]]] 存储历史
- deque 限制长度（如 maxlen=20）防止内存膨胀
- detect_change: 比较最近两次情绪的 intensity 差异是否超过阈值

### Step 6: 测试

同时实现 tests/test_prompts.py 和 tests/test_understanding.py:
- 测试 output_schema 的 dataclass 创建和字段验证
- 测试 system_prompt 返回格式
- 测试 task_prompts 的 conversation 构建格式
- 测试 response_parser 对合法 JSON、残缺 JSON、非 JSON 的处理
- 测试 state_tracker 的 update/get_trend/detect_change
```

---

## Phase 3 Prompt（模型层）

```
实现 src/model/qwen_omni.py — Qwen2.5-Omni-3B 模型封装。

### 核心类: QwenOmniModel

class QwenOmniModel:
    def __init__(self, model_path: str, torch_dtype: str = "bfloat16",
                 attn_implementation: str = "sdpa", max_new_tokens: int = 512):
        # 保存配置，不在 __init__ 中加载模型

    def load(self) → None:
        # 加载模型和 processor
        # 使用 Qwen2_5OmniThinkerForConditionalGeneration（仅文本输出）
        # attn_implementation="sdpa"（MPS 兼容）
        # device_map="auto"

    def infer(self, conversation: list[dict],
              use_audio_in_video: bool = True) → str:
        # 1. processor.apply_chat_template() 处理输入
        # 2. torch.no_grad() 下调用 model.generate()
        # 3. processor.batch_decode() 解码输出
        # 4. 返回文本字符串

    def is_loaded(self) → bool:
        # 检查模型是否已加载

    @property
    def device(self) → torch.device:
        # 返回模型所在设备

### 技术要点

- 导入: from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
- 加载: from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
- 推理输入处理用 processor.apply_chat_template():
  processor.apply_chat_template(conversation, load_audio_from_video=False,
      add_generation_prompt=True, tokenize=True, return_dict=True,
      return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
- generate() 调用: model.generate(**inputs, use_audio_in_video=use_audio_in_video, max_new_tokens=self.max_new_tokens)
- 解码: processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
- 推理超时记录警告但不抛异常

### 同时实现 scripts/download_model.py

from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-Omni-3B", cache_dir="~/.cache/modelscope")

### 测试 tests/test_model.py

- 测试 QwenOmniModel 初始化（不加载模型）
- 测试 is_loaded() 初始状态为 False
- Mock 测试 infer() 的输入输出格式
```

---

## Phase 4 Prompt（采集层）

```
实现 src/capture/ 层 — 通过 PyObjC + AVFoundation 实时采集音视频。

### Step 1: src/capture/video_capture.py

class VideoCapture:
    def __init__(self, resolution: tuple[int, int] = (1920, 1080), fps: int = 30):
        # 不在 __init__ 中启动采集

    def start(self) → None:
        # 创建 AVCaptureSession
        # 获取默认摄像头 (AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeVideo))
        # 创建 AVCaptureDeviceInput
        # 创建 AVCaptureVideoDataOutput，设置 BGRA 像素格式
        # 设置 delegate (VideoCaptureDelegate)
        # 启动 session

    def stop(self) → None:
        # 停止 session

    def set_frame_callback(self, callback: Callable[[np.ndarray, float], None]) → None:
        # 设置帧回调函数 (frame_rgb, timestamp)

    @property
    def is_running(self) → bool

技术要点:
- VideoCaptureDelegate 是 NSObject 子类，实现 captureOutput_didOutputSampleBuffer_fromConnection_
- CVPixelBuffer → numpy: 锁定 → 读取 BGRA → 转 RGB → copy → 解锁
- 回调在 dispatch queue 后台线程，写共享数据需加锁
- AlwaysDiscardsLateVideoFrames = True

### Step 2: src/capture/audio_capture.py

class AudioCapture:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        ...

    def start(self) → None:
        # 类似视频采集，使用 AVMediaTypeAudio
        # AVCaptureAudioDataOutput + AudioCaptureDelegate

    def stop(self) → None
    def set_audio_callback(self, callback: Callable[[np.ndarray, float], None]) → None

技术要点:
- CMSampleBuffer → numpy: 从 CMBlockBuffer 提取 PCM 数据
- 16-bit PCM → float32: / 32768.0

### Step 3: src/capture/stream_buffer.py

解耦采集与推理的核心缓冲组件:

@dataclass
class InferenceWindow:
    frames: list[tuple[np.ndarray, float]]     # [(帧RGB, 时间戳), ...]
    audio_chunks: list[tuple[np.ndarray, float]]  # [(音频块, 时间戳), ...]
    start_ts: float
    end_ts: float

    def get_audio_array(self) → np.ndarray:
        # 拼接所有音频块为单一数组

class StreamBuffer:
    def __init__(self, window_duration: float = 1.5, max_windows: int = 3):
        ...

    def push_frame(self, frame: np.ndarray, timestamp: float) → None:
        # 线程安全地存入帧

    def push_audio(self, chunk: np.ndarray, timestamp: float) → None:
        # 线程安全地存入音频块

    def get_window(self) → InferenceWindow | None:
        # 如果当前窗口已满（时长 >= window_duration），返回并切换到新窗口
        # 否则返回 None

    def reset(self) → None

技术要点:
- 所有 push/get 方法必须 threading.Lock 保护
- 窗口数超过 max_windows 时丢弃最旧窗口
- get_window 是非阻塞的

### 测试 tests/test_capture.py

- 测试 StreamBuffer 的 push/get 逻辑（不依赖硬件）
- 测试 InferenceWindow 的 audio 拼接
- 测试窗口超限丢弃
- VideoCapture/AudioCapture 的测试标记为 @pytest.mark.skipif (无摄像头环境跳过)
```

---

## Phase 5 Prompt（预处理层）

```
实现 src/preprocessing/ 层。

### Step 1: src/preprocessing/frame_sampler.py

class FrameSampler:
    def __init__(self, strategy: str = "uniform", max_frames: int = 4):
        ...

    def sample(self, frames: list[tuple[np.ndarray, float]]) → list[np.ndarray]:
        # 从带时间戳的帧列表中采样
        # strategy="uniform": 均匀间隔选取
        # 返回最多 max_frames 个帧

策略:
- uniform: 等间隔选帧（如 45 帧选 4 帧 → 取第 0, 15, 30, 44 帧）
- 帧数不足 max_frames 时全部返回

### Step 2: src/preprocessing/roi_extractor.py

class ROIExtractor:
    def __init__(self, padding_ratio: float = 0.2, detection_backend: str = "vision"):
        ...

    def extract(self, frame: np.ndarray) → list[np.ndarray]:
        # 1. numpy → CIImage
        # 2. VNDetectHumanRectanglesRequest 检测人体
        # 3. 坐标转换（Vision 左下角原点 → numpy 左上角原点）
        # 4. 添加 padding 后裁剪 ROI
        # 5. 未检测到人体时返回 [frame]（整帧兜底）

    def detect_persons(self, frame: np.ndarray) → list[dict]:
        # 返回检测结果 [{"bbox": (x,y,w,h), "confidence": float}, ...]

技术要点:
- Vision 坐标系: 归一化 [0,1]，原点在左下角
- 转换: py = int((1 - y - h) * frame_height) 翻转 Y 轴
- padding: 每边扩展 padding_ratio * bbox尺寸
- numpy → CIImage: 通过 PIL.Image → PNG bytes → NSData → CIImage

### 测试 tests/test_preprocessing.py

- 测试 FrameSampler 的均匀采样（给定 10 帧选 4 帧的结果）
- 测试帧数不足时全部返回
- ROIExtractor 的测试标记为 macOS only（需要 Vision 框架）
```

---

## Phase 6 Prompt（管道编排 + 主入口）

```
实现 src/pipeline/realtime_pipeline.py 和 main.py。

### src/pipeline/realtime_pipeline.py

class RealtimePipeline:
    def __init__(self, config: dict):
        # 从 config 初始化所有组件:
        # - VideoCapture, AudioCapture, StreamBuffer
        # - FrameSampler, ROIExtractor
        # - QwenOmniModel
        # - EmotionStateTracker

    def start(self) → None:
        # 1. 加载模型 (model.load())
        # 2. 启动采集 (video_capture.start(), audio_capture.start())
        # 3. 设置帧/音频回调 → push 到 stream_buffer
        # 4. 启动推理循环

    def stop(self) → None:
        # 停止采集和推理循环

    def _inference_loop(self) → None:
        # 主循环:
        # while self._running:
        #   1. window = buffer.get_window()  — 非阻塞
        #   2. if window is None: sleep(interval); continue
        #   3. sampled_frames = frame_sampler.sample(window.frames)
        #   4. 判断单人/多人 → 构建 conversation
        #      - 多人: roi_extractor.extract() 裁剪各人物，逐人推理
        #      - 单人: 直接用采样帧
        #   5. conversation = build_conversation(...)
        #   6. response = model.infer(conversation)
        #   7. result = parser.parse(response)
        #   8. tracker.update(result, timestamp)
        #   9. 记录延迟 (logging.info)

    def get_current_state(self) → dict:
        # 返回当前所有人物的情绪状态

### main.py

入口:
1. 解析命令行参数（--config 指定配置目录）
2. 加载并合并 YAML 配置
3. setup_logging()
4. 创建 RealtimePipeline(config)
5. pipeline.start()
6. 主线程等待（Ctrl+C 优雅退出）
7. pipeline.stop()

### 测试 tests/test_pipeline.py

- Mock 所有组件，测试 pipeline 的编排逻辑
- 测试 start/stop 生命周期
- 测试配置加载与合并
```

---

## 每个 Phase 完成后的检查清单

```
完成当前 Phase 后，请执行以下检查:

1. [ ] 所有新文件都有 `from __future__ import annotations`
2. [ ] 所有公开函数/类都有 docstring
3. [ ] 所有函数参数和返回值都有类型注解
4. [ ] 使用 logging 而非 print
5. [ ] 导入顺序: 标准库 → 第三方 → 项目内部
6. [ ] 运行 ruff check src/ 无错误
7. [ ] 运行 pytest tests/test_xxx.py 通过
8. [ ] 没有硬编码的配置值（从参数或配置传入）
9. [ ] 没有网络请求（隐私保护）
10. [ ] 模块只依赖其职责范围内的模块（不违反数据流方向）
```

---

## 快速启动（复制此段到对话开头）

```
我要开始实现 Qwen2.5-Omni 实时情绪理解系统。项目框架已就绪（目录/配置/规则全部到位），所有 src/ 下的 .py 文件目前只有 docstring 和 future import，需要填充实现代码。

请先阅读以下文件了解项目上下文:
- Doc/PRD_V1.md (产品需求)
- configs/*.yaml (所有配置文件)
- .cursor/rules/*.mdc (架构规则和编码规范)

然后从 Phase 1 开始，逐 Phase 实现。每个 Phase 完成后运行测试确认通过，再进入下一 Phase。

现在请开始 Phase 1: 实现 src/utils/logger.py 和 tests/test_utils.py。
```
