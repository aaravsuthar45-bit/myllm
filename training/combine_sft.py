import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMINI_FILE = ROOT / "data" / "train.jsonl"
CLAUDE_FILE = ROOT / "data" / "processed" / "claude_sft_clean.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "combined_sft.jsonl"

combined = []

# Convert Gemini question-answer examples into conversations.
with GEMINI_FILE.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue

        item = json.loads(line)
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question or not answer:
            continue

        combined.append({
            "conversation": f"gemini_{i + 1}",
            "source": "gemini",
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })

# Load cleaned Claude multi-turn conversations.
with CLAUDE_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        messages = item.get("messages", [])

        if not any(m["role"] == "user" for m in messages):
            continue
        if not any(m["role"] == "assistant" for m in messages):
            continue

        combined.append({
            "conversation": item.get("conversation", "claude_unknown"),
            "source": "claude",
            "messages": messages
        })

# Shuffle conversation order, keeping each conversation intact.
random.seed(42)
random.shuffle(combined)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for item in combined:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

gemini_count = sum(x["source"] == "gemini" for x in combined)
claude_count = sum(x["source"] == "claude" for x in combined)

print("Combined SFT dataset created!")
print("Gemini conversations:", gemini_count)
print("Claude conversations:", claude_count)
print("Total conversations:", len(combined))
print("Saved to:", OUTPUT_FILE)