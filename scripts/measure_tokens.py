"""测量实际 JSON 输出所需的 token 数量。"""
import json
import sys

MODEL_PATH = "/Users/tonlyai/Documents/Qwen2.5/Omni3B/~/.cache/modelscope/Qwen/Qwen2.5-Omni-3B"

print("正在加载 tokenizer ...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ── 模拟各种长度的真实输出 ──

samples = {
    "最短输出 (英文 description, null secondary)": json.dumps({
        "person_id": "person_0",
        "primary_emotion": "neutral",
        "emotion_intensity": 0.3,
        "secondary_emotion": None,
        "confidence": 0.8,
        "description": "The person appears calm with a relaxed posture."
    }, ensure_ascii=False),

    "典型输出 (英文 description, 有 secondary)": json.dumps({
        "person_id": "person_0",
        "primary_emotion": "happy",
        "emotion_intensity": 0.75,
        "secondary_emotion": "surprised",
        "confidence": 0.85,
        "description": "The person is smiling broadly with raised eyebrows, suggesting genuine happiness mixed with pleasant surprise."
    }, ensure_ascii=False),

    "中文 description (较长)": json.dumps({
        "person_id": "person_0",
        "primary_emotion": "sad",
        "emotion_intensity": 0.6,
        "secondary_emotion": "fearful",
        "confidence": 0.7,
        "description": "该人物眉头紧锁，嘴角下垂，肩膀微微内收，整体表现出悲伤情绪，同时眼神中流露出一丝不安。"
    }, ensure_ascii=False),

    "最长合理输出 (详细中文 description)": json.dumps({
        "person_id": "person_0",
        "primary_emotion": "angry",
        "emotion_intensity": 0.9,
        "secondary_emotion": "contemptuous",
        "confidence": 0.92,
        "description": "该人物面部表情紧绷，眉毛压低且内聚，嘴唇紧闭呈一字形，下颌肌肉明显紧张。肢体语言上双手握拳，身体前倾，整体姿态传达出强烈的愤怒情绪，同时嘴角一侧微微上扬带有轻蔑意味。"
    }, ensure_ascii=False),

    "模型可能带前缀的情况": '```json\n' + json.dumps({
        "person_id": "person_0",
        "primary_emotion": "happy",
        "emotion_intensity": 0.8,
        "secondary_emotion": None,
        "confidence": 0.9,
        "description": "The person shows a warm smile with relaxed body language."
    }, ensure_ascii=False) + '\n```',

    "带缩进的 JSON (模型有时会格式化)": json.dumps({
        "person_id": "person_0",
        "primary_emotion": "surprised",
        "emotion_intensity": 0.7,
        "secondary_emotion": "happy",
        "confidence": 0.82,
        "description": "Wide eyes and open mouth indicate surprise, with subtle smile suggesting positive reaction."
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
typical = samples["典型输出 (英文 description, 有 secondary)"]
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
