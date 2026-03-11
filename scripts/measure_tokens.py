"""测量实际 JSON 输出所需的 token 数量。"""
import json
import sys

MODEL_PATH = "/Users/tonlyai/Documents/Qwen2.5/Omni3B/~/.cache/modelscope/Qwen/Qwen2.5-Omni-3B"

print("正在加载 tokenizer ...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ── 模拟各种长度的真实输出 ──

samples = {
    "最短输出 (null secondary)": json.dumps({
        "primary_emotion": "neutral",
        "secondary_emotion": None,
    }, ensure_ascii=False),

    "典型输出 (有 secondary)": json.dumps({
        "primary_emotion": "happy",
        "secondary_emotion": "surprised",
    }, ensure_ascii=False),

    "模型可能带前缀的情况": '```json\n' + json.dumps({
        "primary_emotion": "happy",
        "secondary_emotion": None,
    }, ensure_ascii=False) + '\n```',

    "带缩进的 JSON (模型有时会格式化)": json.dumps({
        "primary_emotion": "surprised",
        "secondary_emotion": "happy",
    }, indent=2, ensure_ascii=False),
}

print("\n" + "=" * 70)
print(f"{'场景':<40} {'token 数':>8}  {'字符数':>6}")
print("=" * 70)

max_tokens = 0
for label, text in samples.items():
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    count = len(token_ids)
    max_tokens = max(max_tokens, count)
    print(f"{label:<40} {count:>8}  {len(text):>6}")

print("=" * 70)
print(f"{'最大 token 数':<40} {max_tokens:>8}")
print()

# ── 安全边际分析 ──
print("─" * 70)
print("安全边际分析:")
print("─" * 70)
for limit in [48, 56, 64, 80, 96, 128]:
    safe = "OK" if limit > max_tokens * 1.2 else ("RISKY" if limit > max_tokens else "DANGER")
    margin = limit - max_tokens
    print(f"  max_new_tokens={limit:>3}  余量={margin:>+4} tokens  状态={safe}")
print()

# ── 截断测试: 看 JSON 在不同 token 限制下是否完整 ──
print("─" * 70)
print("截断风险测试 (典型输出在不同 token 限制下是否仍为合法 JSON):")
print("─" * 70)
typical = samples["典型输出 (有 secondary)"]
token_ids = tokenizer.encode(typical, add_special_tokens=False)
full_len = len(token_ids)

for limit in [32, 40, 48, 56, 64, 80]:
    truncated_ids = token_ids[:limit]
    truncated_text = tokenizer.decode(truncated_ids)
    try:
        parsed = json.loads(truncated_text)
        fields = list(parsed.keys())
        print(f"  limit={limit:>3}: 合法 JSON, 字段={fields}")
    except json.JSONDecodeError as e:
        last_20 = truncated_text[-30:].replace('\n', '\\n')
        print(f"  limit={limit:>3}: 截断! 末尾=...{last_20}")

print(f"  limit={full_len:>3}: 完整 (原始长度)")
