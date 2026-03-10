"""验证后端能力声明与实时链路关键场景。"""
from __future__ import annotations

import argparse
import sys
from typing import Any
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import main as app_main
from src.model import create_inference_backend
from src.prompts.system_prompt import build_system_prompt_text
from src.prompts.task_prompts import build_inference_request, build_single_person_prompt


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="验证 MLX/transformers 后端能力")
    parser.add_argument("--config", type=str, default="configs", help="配置目录")
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="执行轻量 smoke test（会真实加载模型）",
    )
    return parser.parse_args()


def _make_request(with_audio: bool) -> Any:
    frame = Image.new("RGB", (64, 64), color=(128, 128, 128))
    audio = np.zeros(16000, dtype=np.float32) if with_audio else None
    return build_inference_request(
        system_prompt=build_system_prompt_text(),
        task_prompt=build_single_person_prompt(),
        frames=[frame, frame.copy()],
        audio=audio,
        use_audio=with_audio,
    )


def print_capabilities(config: dict[str, Any]) -> None:
    """打印后端能力与策略。"""
    backend = create_inference_backend(config)
    capabilities = backend.get_capabilities()
    print("=== Backend Capabilities ===")
    print(f"backend: {config.get('model', {}).get('backend')}")
    print(f"supports_video_frames: {capabilities.supports_video_frames}")
    print(f"supports_audio_array: {capabilities.supports_audio_array}")
    print(f"supports_batch_infer: {capabilities.supports_batch_infer}")
    print(f"supports_strict_json_prompting: {capabilities.supports_strict_json_prompting}")
    print(f"supports_streaming: {capabilities.supports_streaming}")
    print()
    print("=== Policy ===")
    for key, value in config.get("capability_policy", {}).items():
        print(f"{key}: {value}")
    print()


def run_smoke_test(config: dict[str, Any]) -> None:
    """执行关键场景的轻量 smoke test。"""
    backend = create_inference_backend(config)
    scenarios = [
        ("single_no_audio", [_make_request(with_audio=False)]),
        ("batch_no_audio", [_make_request(with_audio=False), _make_request(with_audio=False)]),
        ("single_with_audio", [_make_request(with_audio=True)]),
    ]

    print("=== Smoke Test ===")
    backend.load()
    for name, requests in scenarios:
        try:
            outputs = backend.batch_infer(requests)
            active_backend = getattr(backend, "active_backend_name", getattr(backend, "name", "unknown"))
            print(f"{name}: ok outputs={len(outputs)} active_backend={active_backend}")
        except Exception as exc:
            print(f"{name}: failed error={exc}")


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = app_main.load_merged_config(args.config)
    print_capabilities(config)
    if args.run_smoke:
        run_smoke_test(config)


if __name__ == "__main__":
    main()
