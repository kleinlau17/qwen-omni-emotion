# BDX 机器人动画模块

动画命令测试与 Pipeline 集成：模型输出 → 调度层 → 机器人发送。

## 项目结构

```
robot/
├── action_scheduler.py            # 动作调度层（模型输出与发送之间的策略）
├── animation_action/              # 动画与发送模块
│   ├── __init__.py                # 模块导出
│   ├── animation_mapper.py        # 动画映射模块
│   └── animation_sender.py        # UDP 发送器
│
├── config.py                      # 配置文件
└── README.md                      # 本文件
```

## Pipeline 集成配置

在 `configs/default.yaml` 或 `configs/pipeline.yaml` 中配置：

```yaml
robot:
  enabled: true           # 启用时，推理结果经调度后发送给机器人
  host: "192.168.1.102"
  port: 5205
  debug_print: false

action_scheduler:
  pass_through: true      # 透传模式（当前实现）
```

数据流：`模型输出 (EmotionResult)` → `ActionScheduler.submit()` → `send_animation(action)`。

## 独立测试配置（config.py）

```python
# 目标设备配置
ANIMATION_TARGET_HOST = "192.168.1.102"
ANIMATION_TARGET_PORT = 5205

# 模拟推理的循环次数
TEST_LOOP_COUNT = 5

# 模拟推理的时间间隔(秒)
TEST_INTERVAL_SEC = 10.0

# 是否实际发送（False 为 dry-run 模式）
TEST_SEND_ENABLED = False

# 是否打印调试信息（JSON 命令）
DEBUG_PRINT = True
```

## 使用方法

### 运行主程序（模拟推理测试）

```bash
cd bdx_emotion_controller
python main.py
```

主程序会：
1. 循环模拟推理输出
2. 将动画映射为命令 JSON
3. 显示生成的命令
4. 发送到机器人

## API

### 主要 API (发送函数)

大多数用户只需要这些函数：

| 函数 | 参数 | 说明 |
| :--- | :--- | :--- |
| `init_sender(host, port)` | `host`: IP地址, `port`: 端口 | 初始化发送器 |
| `close_sender()` | 无 | 关闭发送器 |
| `set_debug_print(enabled)` | `enabled`: True/False | 设置调试打印开关 |
| `send_animation(animation)` | `animation`: 动画名称 | 播放动画 |
| `send_emotion(emotion)` | `emotion`: 情绪名称 | 切换情绪状态 |
| `send_light_turn(on)` | `on`: True/False | 开灯/关灯 |
| `send_standing(emotion)` | `emotion`: 初始情绪 | 进入 Standing 状态 |
| `send_homing()` | 无 | 归位 |
| `send_walking(start, foot_id)` | `start`: True/False, `foot_id`: 足部ID | 开始/停止行走 |
| `send_stop()` | 无 | 停止所有动作 |

### 情绪列表 (`send_emotion` / `send_standing`)

- `neutral` - 平静
- `shy` - 害羞
- `sad` - 悲伤
- `angry` - 愤怒
- `excited` - 兴奋

### 动画列表 (`send_animation`)

- **空闲动画**: `idle_neutral`, `idle_shy`, `idle_sad`, `idle_angry`, `idle_excited`
- **英雄动画**: `hurry_up`, `happy_dance`, `jump`, `laugh_big`, ...
- **前景动画**: `angry_no`, `sad_yes`, `shy_no`, `bo_beep`, ...

### 使用示例

```python
from animation_action import init_sender, send_animation, send_emotion, close_sender

# 初始化发送器（只需一次）
init_sender("192.168.1.102", 5205)

# 直接发送命令
send_animation("angry_no")
send_emotion("sad")
send_light_turn(True)

# 关闭发送器
close_sender()
```

### 动作调度层 (ActionScheduler)

`action_scheduler.py` 在模型输出与机器人发送之间提供可扩展的调度逻辑。当前为透传实现，可后续扩展节流、去重、优先级等策略。调度层会将模型侧的新动作名映射回机器人旧动作名后再发送。

```python
from src.robot.action_scheduler import ActionScheduler
from src.prompts.output_schema import EmotionResult

scheduler = ActionScheduler(config={})
result = EmotionResult(
    person_id="person_0",
    detected_emotion="happy",
    self_emotion="neutral",
    action="neutral.affirm.default.mid",
)
actions = scheduler.submit(result, timestamp=0.0)  # -> ["standard_yes"]
```
