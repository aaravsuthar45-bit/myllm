
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "train.jsonl"
TOKENIZER_FILE = ROOT / "data" / "tokenizer" / "tokenizer.json"
OUTPUT_DIR = ROOT / "data" / "processed"

MAX_SEQ_LEN = 256
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")
user_id = tokenizer.token_to_id("[USER]")
assistant_id = tokenizer.token_to_id("[ASSISTANT]")

required = {
    "BOS": bos_id,
    "EOS": eos_id,
    "USER": user_id,
    "ASSISTANT": assistant_id,
}

for name, token_id in required.items():
    if token_id is None:
        raise ValueError(f"Missing special token: {name}")

print("Vocabulary size:", tokenizer.get_vocab_size())


# --------------------------------------------------
# ENCODE EXAMPLES
# --------------------------------------------------

def encode_example(question, answer):

    question_ids = tokenizer.encode(
        question,
        add_special_tokens=False
    ).ids

    answer_ids = tokenizer.encode(
        answer,
        add_special_tokens=False
    ).ids

    # Sequence structure:
    # [BOS] [USER] question [ASSISTANT] answer [EOS]

    prefix = [
        bos_id,
        user_id,
        *question_ids,
        assistant_id
    ]

    # Reserve space for at least one answer token and EOS.
    available_answer_length = MAX_SEQ_LEN - len(prefix) - 1

    if available_answer_length < 1:
        prefix = prefix[:MAX_SEQ_LEN - 2]
        available_answer_length = 1

    answer_ids = answer_ids[:available_answer_length]

    token_ids = prefix + answer_ids + [eos_id]

    # Loss mask: 0 means ignore, 1 means learn.
    loss_mask = (
        [0] * len(prefix)
        + [1] * len(answer_ids)
        + [1]
    )

    return {
        "input_ids": token_ids,
        "loss_mask": loss_mask
    }


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nLoading examples...")

examples = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)

        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question or not answer:
            continue

        encoded = encode_example(question, answer)

        examples.append(encoded)


if len(examples) < 2:
    raise ValueError("Need at least two valid examples.")

print("Encoded examples:", len(examples))


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

random.seed(RANDOM_SEED)
random.shuffle(examples)

validation_count = max(
    1,
    int(len(examples) * VALIDATION_RATIO)
)

validation_examples = examples[:validation_count]
training_examples = examples[validation_count:]


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    training_examples,
    OUTPUT_DIR / "train.pt"
)

torch.save(
    validation_examples,
    OUTPUT_DIR / "validation.pt"
)


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

total_tokens = sum(
    len(item["input_ids"])
    for item in training_examples
)

answer_tokens = sum(
    sum(item["loss_mask"])
    for item in training_examples
)

print("\nDATA PREPARATION COMPLETE")

print("Training examples:", len(training_examples))
print("Validation examples:", len(validation_examples))
print("Training tokens:", total_tokens)
print("Assistant target tokens:", answer_tokens)

print(
    "Average sequence length:",
    round(total_tokens / len(training_examples), 2)
)

print("\nSaved files:")
print(OUTPUT_DIR / "train.pt")
print(OUTPUT_DIR / "validation.pt")
