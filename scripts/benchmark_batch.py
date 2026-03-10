"""对比不同后端在多批次输入下的推理延迟与吞吐。"""
from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import main as app_main
from src.model import create_inference_backend
from src.prompts.system_prompt import build_system_prompt_text
from src.prompts.task_prompts import build_inference_request, build_single_person_prompt

FRAME_SIZE = (480, 270)
DEFAULT_NUM_FRAMES = 2
DEFAULT_WARMUP_RUNS = 1
DEFAULT_BENCH_RUNS = 2
DEFAULT_BATCH_SIZES = [1, 2, 4]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="比较不同后端的批量推理性能")
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
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / (1 << 10):.1f} KB"


def process_rss_bytes() -> int:
    """返回当前进程 RSS。macOS 上 ru_maxrss 单位为字节。"""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def make_dummy_frames(n: int = DEFAULT_NUM_FRAMES) -> list[Image.Image]:
    """构造模拟视频帧。"""
    return [
        Image.fromarray(np.random.randint(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
        for _ in range(n)
    ]


def make_requests(batch_size: int) -> list[Any]:
    """构造批量推理请求。"""
    return [
        build_inference_request(
            system_prompt=build_system_prompt_text(),
            task_prompt=build_single_person_prompt(),
            frames=make_dummy_frames(),
            use_audio=False,
            metadata={"sample_id": idx},
        )
        for idx in range(batch_size)
    ]


def build_backend_config(
    config: dict[str, Any],
    backend_name: str,
    allow_download: bool,
) -> dict[str, Any]:
    """生成只启用某一个后端的 benchmark 配置。"""
    bench_config = copy.deepcopy(config)
    model_cfg = bench_config.setdefault("model", {})
    if backend_name == "mlx":
        model_cfg["backend"] = "mlx"
        model_cfg["fallback_backend"] = ""
        if not allow_download and not _local_path_exists(str(model_cfg.get("local_path", ""))):
            raise FileNotFoundError("MLX 本地模型不存在，且未启用 --allow-download。")
    elif backend_name == "transformers":
        model_cfg["backend"] = "transformers"
        model_cfg["fallback_backend"] = ""
        model_cfg["local_path"] = model_cfg.get("fallback_local_path") or model_cfg.get("local_path")
        model_cfg["repo_id"] = model_cfg.get("fallback_repo_id") or model_cfg.get("repo_id")
        if not allow_download and not _local_path_exists(str(model_cfg.get("local_path", ""))):
            raise FileNotFoundError("transformers 本地模型不存在，且未启用 --allow-download。")
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")
    return bench_config


def benchmark_backend(
    config: dict[str, Any],
    backend_name: str,
    allow_download: bool,
) -> list[dict[str, Any]]:
    """对指定后端执行多组 batch benchmark。"""
    backend = create_inference_backend(build_backend_config(config, backend_name, allow_download))
    baseline_mem = process_rss_bytes()
    backend.load()

    results: list[dict[str, Any]] = []
    for batch_size in DEFAULT_BATCH_SIZES:
        print(f"[{backend_name}] batch_size={batch_size}")

        for warmup_idx in range(DEFAULT_WARMUP_RUNS):
            requests = make_requests(batch_size)
            backend.batch_infer(requests)
            print(f"  warmup {warmup_idx + 1}: ok")

        run_latencies: list[float] = []
        for run_idx in range(DEFAULT_BENCH_RUNS):
            requests = make_requests(batch_size)
            start = perf_counter()
            outputs = backend.batch_infer(requests)
            elapsed = perf_counter() - start
            run_latencies.append(elapsed)
            print(
                f"  run {run_idx + 1}: total={elapsed:.2f}s "
                f"per_item={elapsed / batch_size:.2f}s outputs={len(outputs)}"
            )

        avg_total = sum(run_latencies) / len(run_latencies)
        results.append(
            {
                "backend": backend_name,
                "batch_size": batch_size,
                "avg_total_s": avg_total,
                "avg_per_item_s": avg_total / batch_size,
                "avg_throughput": batch_size / avg_total if avg_total else 0.0,
                "rss_bytes": process_rss_bytes(),
                "extra_mem": max(0, process_rss_bytes() - baseline_mem),
            }
        )
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """打印汇总报告。"""
    print("=" * 80)
    print("汇总报告")
    print("=" * 80)
    if not results:
        print("无有效结果")
        return

    header = (
        f"{'backend':>12} | {'batch':>5} | {'总耗时(s)':>10} | "
        f"{'单窗口(s)':>10} | {'吞吐(/s)':>9} | {'额外内存':>10}"
    )
    print(header)
    print("─" * len(header))
    for item in results:
        print(
            f"{item['backend']:>12} | {item['batch_size']:>5} | "
            f"{item['avg_total_s']:>10.2f} | {item['avg_per_item_s']:>10.2f} | "
            f"{item['avg_throughput']:>9.3f} | {fmt_bytes(item['extra_mem']):>10}"
        )

    print()
    best_per_item = min(results, key=lambda item: item["avg_per_item_s"])
    best_throughput = max(results, key=lambda item: item["avg_throughput"])
    print("推荐观察点：")
    print(
        f"- 单窗口最低延迟: backend={best_per_item['backend']} "
        f"batch={best_per_item['batch_size']} per_item={best_per_item['avg_per_item_s']:.2f}s"
    )
    print(
        f"- 最高吞吐: backend={best_throughput['backend']} "
        f"batch={best_throughput['batch_size']} throughput={best_throughput['avg_throughput']:.3f}/s"
    )


def _local_path_exists(path: str) -> bool:
    """检查本地模型路径是否存在。"""
    return bool(path) and Path(path).expanduser().exists()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = app_main.load_merged_config(args.config)
    backend_names = args.backends or [
        str(config.get("model", {}).get("backend", "mlx")),
        str(config.get("model", {}).get("fallback_backend", "transformers")),
    ]
    backend_names = [name for name in dict.fromkeys(backend_names) if name]

    all_results: list[dict[str, Any]] = []
    for backend_name in backend_names:
        try:
            all_results.extend(benchmark_backend(config, backend_name, args.allow_download))
        except Exception as exc:
            print(f"[{backend_name}] benchmark failed: {exc}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
