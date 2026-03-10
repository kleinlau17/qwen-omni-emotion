"""按配置下载主后端与回退后端模型。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modelscope import snapshot_download as modelscope_snapshot_download

import main as app_main


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="下载主模型与回退模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="配置目录路径",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=("primary", "fallback", "all"),
        default="all",
        help="下载主模型、回退模型或两者都下载",
    )
    return parser.parse_args()


def download_backend_model(model_id: str, backend: str, cache_dir: str | None = None) -> str:
    """按后端类型下载模型（统一通过 ModelScope）。"""
    normalized_backend = backend.lower().strip()
    expanded_cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else str(
        Path("~/.cache/modelscope").expanduser()
    )
    # 目前无论是 MLX 版还是 transformers 版，都从 ModelScope 拉取。
    return str(
        modelscope_snapshot_download(
            model_id,
            cache_dir=expanded_cache_dir,
        )
    )
    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    """按配置下载主模型与回退模型。"""
    args = parse_args()
    config = app_main.load_merged_config(args.config)

    model_cfg = config.get("model", {})
    targets: list[tuple[str, str]] = []

    if args.target in {"primary", "all"}:
        targets.append(
            (
                str(model_cfg.get("backend", "mlx")),
                str(model_cfg.get("repo_id") or model_cfg.get("local_path") or "").strip(),
            )
        )
    if args.target in {"fallback", "all"} and model_cfg.get("fallback_backend"):
        targets.append(
            (
                str(model_cfg.get("fallback_backend", "transformers")),
                str(
                    model_cfg.get("fallback_repo_id")
                    or model_cfg.get("fallback_local_path")
                    or ""
                ).strip(),
            )
        )

    for backend, model_id in targets:
        if not model_id:
            continue
        path = download_backend_model(model_id=model_id, backend=backend)
        print(f"[{backend}] downloaded: {path}")


if __name__ == "__main__":
    main()
