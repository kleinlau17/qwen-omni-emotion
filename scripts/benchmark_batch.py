"""Benchmark: 不同 batch size 下 Qwen2.5-Omni-3B 的推理延迟与吞吐量。

在 M3 Ultra MPS 上测量，使用与实际 pipeline 相同的输入规格：
- 2 帧 480x270 视频帧 (PIL Image)
- bfloat16 权重
- max_new_tokens=128
- SDPA attention
"""
from __future__ import annotations

import gc
import json
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "/Users/tonlyai/Documents/Qwen2.5/Omni3B/~/.cache/modelscope/Qwen/Qwen2.5-Omni-3B"
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
MAX_NEW_TOKENS = 128
FRAME_SIZE = (480, 270)
NUM_FRAMES = 2
WARMUP_RUNS = 2
BENCH_RUNS = 3
BATCH_SIZES = [1, 2, 3, 4, 6, 8]


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / (1 << 20):.1f} MB"


def make_dummy_frames(n: int = NUM_FRAMES) -> list[Image.Image]:
    return [
        Image.fromarray(np.random.randint(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
        for _ in range(n)
    ]


def build_conversation(frames: list[Image.Image]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "你是一位多模态情绪理解与互动决策专家。"
                "请严格遵循任务指令中的 JSON Schema 输出，仅返回一个合法 JSON 对象。"
            )}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": (
                    "请对当前单人场景进行情绪识别与互动决策，并返回严格 JSON。"
                    '{"detected_emotion":"happy","self_emotion":"neutral","action":"scan_01"}'
                )},
            ],
        },
    ]


def benchmark_single(model: Any, processor: Any, device: torch.device, batch_size: int) -> dict:
    """对指定 batch_size 执行一次推理并返回结果。"""
    conversations = [build_conversation(make_dummy_frames()) for _ in range(batch_size)]

    # 逐个 tokenize 后手动 batch (对齐 padding)
    if batch_size == 1:
        inputs = processor.apply_chat_template(
            conversations[0],
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
    else:
        inputs = processor.apply_chat_template(
            conversations,
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

    inputs = inputs.to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    actual_batch = input_ids.shape[0]

    torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            use_audio_in_video=False,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    gen_len = output_ids.shape[1] - input_len
    mem_alloc = torch.mps.current_allocated_memory()
    mem_driver = torch.mps.driver_allocated_memory()

    del output_ids, inputs
    torch.mps.empty_cache()
    gc.collect()

    return {
        "batch_size": actual_batch,
        "input_tokens": input_len,
        "gen_tokens": gen_len,
        "total_seconds": elapsed,
        "per_item_seconds": elapsed / actual_batch,
        "throughput_items_per_sec": actual_batch / elapsed,
        "mem_allocated": mem_alloc,
        "mem_driver": mem_driver,
    }


def main() -> None:
    print("=" * 80)
    print("Qwen2.5-Omni-3B Batch Inference Benchmark (MPS)")
    print(f"帧规格: {NUM_FRAMES} × {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"batch sizes: {BATCH_SIZES}")
    print(f"每个 batch size: warmup={WARMUP_RUNS} + bench={BENCH_RUNS} 次")
    print("=" * 80)

    print("\n正在加载模型 ...")
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
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    device = model.device
    print(f"模型已加载 (device={device})\n")

    baseline_mem = torch.mps.current_allocated_memory()
    results: list[dict] = []

    for bs in BATCH_SIZES:
        print(f"─── batch_size={bs} ───")

        # warmup
        for i in range(WARMUP_RUNS):
            try:
                r = benchmark_single(model, processor, device, bs)
                print(f"  warmup {i+1}: {r['total_seconds']:.2f}s")
            except Exception as e:
                print(f"  warmup {i+1}: FAILED - {e}")
                break
        else:
            # bench runs
            run_results = []
            for i in range(BENCH_RUNS):
                try:
                    r = benchmark_single(model, processor, device, bs)
                    run_results.append(r)
                    print(f"  run {i+1}: total={r['total_seconds']:.2f}s "
                          f"per_item={r['per_item_seconds']:.2f}s "
                          f"throughput={r['throughput_items_per_sec']:.2f}/s "
                          f"gen_tokens={r['gen_tokens']} "
                          f"mem={fmt_bytes(r['mem_driver'])}")
                except Exception as e:
                    print(f"  run {i+1}: FAILED - {e}")

            if run_results:
                avg = {
                    "batch_size": bs,
                    "avg_total_s": sum(r["total_seconds"] for r in run_results) / len(run_results),
                    "avg_per_item_s": sum(r["per_item_seconds"] for r in run_results) / len(run_results),
                    "avg_throughput": sum(r["throughput_items_per_sec"] for r in run_results) / len(run_results),
                    "avg_gen_tokens": sum(r["gen_tokens"] for r in run_results) / len(run_results),
                    "max_mem_driver": max(r["mem_driver"] for r in run_results),
                    "extra_mem": max(r["mem_driver"] for r in run_results) - baseline_mem if baseline_mem else 0,
                }
                results.append(avg)

        print()

    # ── 汇总报告 ──
    print("=" * 80)
    print("汇总报告")
    print("=" * 80)
    if not results:
        print("无有效结果")
        return

    b1 = next((r for r in results if r["batch_size"] == 1), None)
    b1_total = b1["avg_total_s"] if b1 else 1.0

    header = (f"{'batch':>5} | {'总耗时(s)':>10} | {'单窗口(s)':>10} | {'吞吐(/s)':>9} | "
              f"{'gen_tok':>8} | {'额外显存':>10} | {'加速比':>7} | {'效率':>6}")
    print(header)
    print("─" * len(header))
    for r in results:
        speedup = b1_total / r["avg_per_item_s"] if b1 else 0
        efficiency = (r["avg_throughput"] / r["batch_size"]) / (b1["avg_throughput"] if b1 else 1) * 100
        print(f"{r['batch_size']:>5} | {r['avg_total_s']:>10.2f} | {r['avg_per_item_s']:>10.2f} | "
              f"{r['avg_throughput']:>9.3f} | {r['avg_gen_tokens']:>8.0f} | "
              f"{fmt_bytes(r['extra_mem']):>10} | {speedup:>6.2f}x | {efficiency:>5.1f}%")

    print()
    print("─── 窗口丢弃率模拟 (窗口间隔=1.5s) ───")
    window_interval = 1.5
    for r in results:
        windows_per_cycle = r["batch_size"]
        wait_time = window_interval * windows_per_cycle
        total_cycle = wait_time + r["avg_total_s"]
        produced = windows_per_cycle / total_cycle
        incoming = 1.0 / window_interval
        drop_rate = max(0, 1 - produced / incoming) * 100
        print(f"  batch={r['batch_size']}: cycle={total_cycle:.1f}s "
              f"产出={produced:.2f}窗口/s vs 输入={incoming:.2f}窗口/s "
              f"丢弃率={drop_rate:.0f}%")


if __name__ == "__main__":
    main()
