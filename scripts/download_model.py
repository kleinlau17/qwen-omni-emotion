"""从 ModelScope 下载 Qwen2.5-Omni-3B 模型"""
from __future__ import annotations

from modelscope import snapshot_download


def main() -> None:
    """下载 Qwen2.5-Omni-3B 到本地缓存目录。"""
    snapshot_download("Qwen/Qwen2.5-Omni-3B", cache_dir="~/.cache/modelscope")


if __name__ == "__main__":
    main()
