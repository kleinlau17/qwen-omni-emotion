"""机器人动画与动作调度模块。

- animation_action: 动画映射与 UDP 发送
- action_scheduler: 模型输出与发送之间的调度层
"""

from .action_scheduler import ActionScheduler

__all__ = ["ActionScheduler"]
