import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Put your Claude conversation chunk here:
INPUT_PATH = ROOT / "data" / "raw" / "combined_conversations.json"

TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
OUTPUT_JSONL = ROOT / "data" / "processed" / "claude_sft.jsonl"
TRAIN_PT = ROOT / "data" / "processed" / "claude_train.pt"
VALIDATION_PT = ROOT / "data" / "processed" / "claude_validation.pt"

MAX_SEQ_LEN = 256
MAX_TARGET_TOKENS = 192
VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42

# Exclude conversations that contain private server/connector details.
# Add more terms here if you want to exclude other topics.
EXCLUDE_KEYWORDS = (
    "modlink",
    "authorized servers",
    "connector status",
    "guild id",
)


# ------------------------------------------------------------
# Read Claude conversation chunk
# ------------------------------------------------------------

def load_conversations(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Some pasted chunks end with a trailing comma and omit the
        # final closing bracket. Repair only that specific case.
        repaired = re.sub(r",\s*$", "", text)
        if repaired.startswith("[") and not repaired.endswith("]"):
            data = json.loads(repaired + "]")
        else:
            raise

    if isinstance(data, dict):
        # Support exports that wrap the conversation list.
        for key in ("conversations", "chats", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of conversation records.")

    return data


# ------------------------------------------------------------
# Parse the Markdown stored inside each conversation record
# ------------------------------------------------------------

MESSAGE_PATTERN = re.compile(
    r"^## (You|Claude)\s*\n"
    r"(?:\*[^*\n]+\*\s*\n)?"
    r"\s*\n?(.*?)"
    r"(?=^## (?:You|Claude)\s*$|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)

UNSUPPORTED_BLOCK = re.compile(
    r"```[ \t]*\n[ \t]*This block is not supported on your current device yet\.[ \t]*\n```",
    flags=re.IGNORECASE,
)


def clean_message(text: str) -> str:
    text = UNSUPPORTED_BLOCK.sub("", text)

    # Remove artifact links and other raw URLs. The surrounding
    # explanation remains useful without a one-off personal link.
    text = re.sub(
        r"(?im)^\s*Link:\s*https?://\S+\s*$",
        "",
        text,
    )
    text = re.sub(r"https?://\S+", "[external link]", text)

    # Remove conversation separators left by the export format.
    text = re.sub(r"(?m)^\s*---\s*$", "", text)

    return text.strip()


def parse_messages(conversation):
    content = conversation.get("content", "")
    messages = []

    for role, body in MESSAGE_PATTERN.findall(content):
        body = clean_message(body)
        if not body:
            continue

        mapped_role = "user" if role == "You" else "assistant"
        messages.append({"role": mapped_role, "content": body})

    return messages


def should_exclude(conversation) -> bool:
    searchable = " ".join(
        str(conversation.get(key, ""))
        for key in ("filename", "path", "content")
    ).lower()

    return any(keyword in searchable for keyword in EXCLUDE_KEYWORDS)


# ------------------------------------------------------------
# Convert each assistant turn into a masked causal-LM example
# ------------------------------------------------------------

def build_examples(conversations, tokenizer):
    special_ids = {
        token: tokenizer.token_to_id(token)
        for token in ("[BOS]", "[EOS]", "[USER]", "[ASSISTANT]")
    }

    missing = [name for name, token_id in special_ids.items() if token_id is None]
    if missing:
        raise ValueError(
            "Tokenizer is missing required special tokens: " + ", ".join(missing)
        )

    bos_id = special_ids["[BOS]"]
    eos_id = special_ids["[EOS]"]
    user_id = special_ids["[USER]"]
    assistant_id = special_ids["[ASSISTANT]"]

    conversation_groups = []
    jsonl_records = []
    skipped_conversations = []
    skipped_turns = 0

    for conversation in conversations:
        filename = conversation.get("filename", "unnamed conversation")

        if should_exclude(conversation):
            skipped_conversations.append(filename)
            continue

        messages = parse_messages(conversation)
        history = []
        group_examples = []

        for message in messages:
            if message["role"] == "user":
                history.append(message)
                continue

            # Only learn from assistant replies that have user context.
            if not any(item["role"] == "user" for item in history):
                skipped_turns += 1
                history.append(message)
                continue

            # Build context from all earlier turns.
            context_ids = [bos_id]

            for old_message in history:
                marker = user_id if old_message["role"] == "user" else assistant_id
                content_ids = tokenizer.encode(
                    old_message["content"],
                    add_special_tokens=False,
                ).ids
                context_ids.append(marker)
                context_ids.extend(content_ids)

            answer_ids = tokenizer.encode(
                message["content"],
                add_special_tokens=False,
            ).ids

            # The target includes the assistant marker and EOS.
            target_ids = [assistant_id] + answer_ids + [eos_id]

            if len(target_ids) > MAX_TARGET_TOKENS:
                target_ids = target_ids[: MAX_TARGET_TOKENS - 1] + [eos_id]

            context_budget = MAX_SEQ_LEN - len(target_ids)
            if context_budget < 1:
                skipped_turns += 1
                history.append(message)
                continue

            # Keep the most recent context when a conversation is long.
            if len(context_ids) > context_budget:
                context_ids = context_ids[-context_budget:]

            input_ids = context_ids + target_ids
            loss_mask = [0] * len(context_ids) + [1] * len(target_ids)

            if len(input_ids) > MAX_SEQ_LEN:
                raise RuntimeError("Internal error: sequence exceeds MAX_SEQ_LEN.")

            group_examples.append({
                "input_ids": input_ids,
                "loss_mask": loss_mask,
            })

            jsonl_records.append({
                "conversation": filename,
                "messages": history + [message],
            })

            history.append(message)

        if group_examples:
            conversation_groups.append(group_examples)

    return (
        conversation_groups,
        jsonl_records,
        skipped_conversations,
        skipped_turns,
    )


# ------------------------------------------------------------
# Split by conversation, not by individual message
# ------------------------------------------------------------

def split_groups(groups):
    if len(groups) < 2:
        # A single conversation cannot provide a clean held-out split.
        # Keep it all for training and leave validation empty.
        return groups, []

    groups = list(groups)
    random.Random(RANDOM_SEED).shuffle(groups)

    val_count = max(1, round(len(groups) * VALIDATION_FRACTION))
    val_count = min(val_count, len(groups) - 1)

    validation_groups = groups[:val_count]
    train_groups = groups[val_count:]

    return train_groups, validation_groups


def flatten(groups):
    return [example for group in groups for example in group]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}\n"
            "Create the data/raw folder and save your Claude chunk as "
            "claude_chunk.json."
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    print("Reading:", INPUT_PATH)
    conversations = load_conversations(INPUT_PATH)
    print("Conversation records found:", len(conversations))

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    groups, jsonl_records, skipped_conversations, skipped_turns = build_examples(
        conversations,
        tokenizer,
    )

    train_groups, validation_groups = split_groups(groups)
    train_examples = flatten(train_groups)
    validation_examples = flatten(validation_groups)

    # Save readable conversation examples for inspection.
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for record in jsonl_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save tokenized examples in the same format as our training pipeline.
    torch.save(train_examples, TRAIN_PT)
    torch.save(validation_examples, VALIDATION_PT)

    print("\nFinished importing Claude conversations.")
    print("Included conversations:", len(groups))
    print("Excluded conversations:", len(skipped_conversations))
    for name in skipped_conversations:
        print("  Excluded:", name)

    print("Assistant training examples:", len(jsonl_records))
    print("Skipped assistant turns:", skipped_turns)
    print("Training examples:", len(train_examples))
    print("Validation examples:", len(validation_examples))

    print("\nSaved files:")
    print(" ", OUTPUT_JSONL)
    print(" ", TRAIN_PT)
    print(" ", VALIDATION_PT)

    if len(groups) < 2:
        print(
            "\nWARNING: Only one eligible conversation was found. "
            "No validation split was made. Do not use this tiny chunk "
            "for a meaningful fine-tuning run; use it to verify the converter."
        )


if __name__ == "__main__":
    main()
