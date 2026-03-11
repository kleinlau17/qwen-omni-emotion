"""测量不同后端加载与一次推理后的内存占用。"""
from __future__ import annotations

import argparse
import copy
import resource
import sys
from pathlib import Path
from time import perf_counter

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
    parser = argparse.ArgumentParser(description="测量后端加载与单次推理内存")
    parser.add_argument("--config", type=str, default="configs", help="配置目录")
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="要测试的后端，可重复传入，如 --backend mlx --backend transformers",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允许在本地缺少权重时自动从远端下载模型",
    )
    return parser.parse_args()


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    return f"{n / (1 << 10):.2f} KB"


def process_rss_bytes() -> int:
    """返回当前进程 RSS。macOS 上 ru_maxrss 单位为字节。"""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def build_backend_config(config: dict, backend_name: str, allow_download: bool) -> dict:
    """生成只启用某一个后端的配置副本。"""
    runtime_config = copy.deepcopy(config)
    model_cfg = runtime_config.setdefault("model", {})
    model_cfg["backend"] = backend_name
    model_cfg["fallback_backend"] = ""
    if backend_name == "transformers":
        model_cfg["local_path"] = model_cfg.get("fallback_local_path") or model_cfg.get("local_path")
        model_cfg["repo_id"] = model_cfg.get("fallback_repo_id") or model_cfg.get("repo_id")
    local_path = str(model_cfg.get("local_path", "")).strip()
    if not allow_download and (not local_path or not Path(local_path).expanduser().exists()):
        raise FileNotFoundError(f"{backend_name} 本地模型不存在，且未启用 --allow-download。")
    return runtime_config


def make_request() -> object:
    """构造单条模拟推理请求。"""
    frame = Image.fromarray(np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8))
    return build_inference_request(
        system_prompt=build_system_prompt_text(),
        task_prompt=build_single_person_prompt(),
        frames=[frame],
        use_audio=False,
    )


def measure_backend(config: dict, backend_name: str, allow_download: bool) -> None:
    """测量单个后端的内存占用。"""
    runtime_config = build_backend_config(config, backend_name, allow_download)
    backend = create_inference_backend(runtime_config)

    rss_baseline = process_rss_bytes()
    print(f"[{backend_name}] 基线 RSS: {fmt_bytes(rss_baseline)}")

    t0 = perf_counter()
    backend.load()
    load_elapsed = perf_counter() - t0
    rss_after_load = process_rss_bytes()
    print(
        f"[{backend_name}] 加载完成: load_time={load_elapsed:.2f}s "
        f"rss={fmt_bytes(rss_after_load)} extra={fmt_bytes(rss_after_load - rss_baseline)}"
    )

    request = make_request()
    t1 = perf_counter()
    outputs = backend.batch_infer([request])
    infer_elapsed = perf_counter() - t1
    rss_after_infer = process_rss_bytes()
    print(
        f"[{backend_name}] 单次推理: infer_time={infer_elapsed:.2f}s "
        f"rss={fmt_bytes(rss_after_infer)} extra={fmt_bytes(rss_after_infer - rss_baseline)} "
        f"outputs={len(outputs)}"
    )
    print()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = app_main.load_merged_config(args.config)
    backends = args.backends or [
        str(config.get("model", {}).get("backend", "mlx")),
        str(config.get("model", {}).get("fallback_backend", "transformers")),
    ]

    print("=" * 60)
    print("后端内存测量")
    print("=" * 60)
    for backend_name in dict.fromkeys(backends):
        if not backend_name:
            continue
        try:
            measure_backend(config, backend_name, args.allow_download)
        except Exception as exc:
            print(f"[{backend_name}] measure failed: {exc}")


if __name__ == "__main__":
    main()
