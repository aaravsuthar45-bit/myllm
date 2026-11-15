import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "claude_sft.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "claude_sft_clean.jsonl"


def signature(message):
    return (
        message.get("role", "").strip().lower(),
        message.get("content", "").strip()
    )


def merge_messages(existing, incoming):
    """Merge cumulative snapshots without repeating their overlapping messages."""
    old = [m for m in existing]
    new = [m for m in incoming]

    old_sigs = [signature(m) for m in old]
    new_sigs = [signature(m) for m in new]

    # Incoming snapshot contains the entire existing conversation.
    if len(new_sigs) >= len(old_sigs) and new_sigs[:len(old_sigs)] == old_sigs:
        return new

    # Existing conversation already contains the incoming snapshot.
    if len(old_sigs) >= len(new_sigs) and old_sigs[:len(new_sigs)] == new_sigs:
        return old

    # Find the largest overlap between the end of old and start of new.
    overlap = 0
    for size in range(1, min(len(old_sigs), len(new_sigs)) + 1):
        if old_sigs[-size:] == new_sigs[:size]:
            overlap = size

    return old + new[overlap:]


conversations = defaultdict(list)
records_read = 0

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        name = record.get("conversation", "unknown")
        messages = record.get("messages", [])

        # Keep only useful user/assistant messages.
        messages = [
            {
                "role": m.get("role", "").strip().lower(),
                "content": m.get("content", "").strip()
            }
            for m in messages
            if m.get("role", "").strip().lower() in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and m.get("content", "").strip()
        ]

        conversations[name] = merge_messages(
            conversations[name], messages
        )
        records_read += 1


cleaned = []
total_messages = 0
total_pairs = 0

for name, messages in conversations.items():
    # Remove consecutive duplicate messages.
    deduplicated = []
    for message in messages:
        if not deduplicated or signature(message) != signature(deduplicated[-1]):
            deduplicated.append(message)

    if not deduplicated:
        continue

    cleaned.append({
        "conversation": name,
        "messages": deduplicated
    })

    total_messages += len(deduplicated)

    for i in range(len(deduplicated) - 1):
        if (
            deduplicated[i]["role"] == "user"
            and deduplicated[i + 1]["role"] == "assistant"
        ):
            total_pairs += 1


with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for conversation in cleaned:
        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


print("Claude cleanup complete!")
print(f"Original records: {records_read}")
print(f"Unique conversations: {len(cleaned)}")
print(f"Messages after cleanup: {total_messages}")
print(f"Adjacent user/assistant pairs: {total_pairs}")
print(f"Saved to: {OUTPUT_FILE}")