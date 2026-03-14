#!/usr/bin/env python3
"""测试机器人动作发送：模拟模型输出，经调度层转发给机器人。

用法:
    python scripts/test_robot_send.py
    python scripts/test_robot_send.py --host 192.168.1.102 --port 5205
    python scripts/test_robot_send.py --dry-run   # 仅打印 JSON，不实际发送
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts.output_schema import EmotionResult
from src.robot.action_scheduler import ActionScheduler
from src.robot.animation_action import (
    close_sender,
    init_sender,
    send_animation,
    set_debug_print,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模拟模型输出并转发给机器人，测试 sad.affirm.default.low 等动作是否执行"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="机器人 IP，未指定时使用 src/robot/config.py",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="机器人端口，未指定时使用 src/robot/config.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印 JSON 命令，不实际发送",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="sad.affirm.default.low",
        help="模拟的 action，默认 sad.affirm.default.low",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # 模拟模型输出
    model_output = {
        "detected_emotion": "neutral",
        "action": args.action,
    }
    print(f"模拟模型输出: {model_output}")

    result = EmotionResult(
        person_id="person_0",
        detected_emotion=model_output["detected_emotion"],
        action=model_output["action"],
    )

    # 经调度层
    scheduler = ActionScheduler()
    actions = scheduler.submit(result, timestamp=0.0)
    print(f"调度输出: {actions}")

    if not actions:
        print("调度层未产出动作，退出")
        return

    # 初始化发送器（dry-run 不初始化，send_animation 在 _sender=None 时只打印）
    if not args.dry_run:
        try:
            if args.host is not None and args.port is not None:
                host, port = args.host, args.port
            else:
                from src.robot.config import ANIMATION_TARGET_HOST, ANIMATION_TARGET_PORT
                host, port = ANIMATION_TARGET_HOST, ANIMATION_TARGET_PORT
            init_sender(host, port)
            set_debug_print(True)
            print(f"已连接机器人: {host}:{port}")
        except Exception as e:
            print(f"初始化发送器失败: {e}")
            sys.exit(1)
    else:
        print("(dry-run 模式，不实际发送)")
        set_debug_print(True)

    # 发送每个动作
    for action in actions:
        ok = send_animation(action, print_json=True)
        status = "成功" if ok else "失败"
        print(f"  send_animation({action!r}) -> {status}")

    if not args.dry_run:
        close_sender()

    print("测试完成，请观察机器人是否执行对应动作。")


if __name__ == "__main__":
    main()
