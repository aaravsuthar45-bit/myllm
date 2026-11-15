"""
Unified SFT Data Preparer:
Takes Q&A / instruction conversation pairs, formats them into standard [USER] ... [ASSISTANT] sequences,
generates strict loss masks (calculating loss ONLY on the assistant's answer),
and packages them into PyTorch tensors for lightning-fast fine-tuning.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def prepare_sft(
    input_file: Path,
    tokenizer_file: Path,
    output_dir: Path,
    max_seq_len: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42
):
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading tokenizer from {tokenizer_file}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    user_id = tokenizer.token_to_id("[USER]")
    assistant_id = tokenizer.token_to_id("[ASSISTANT]")
    pad_id = tokenizer.token_to_id("[PAD]")

    assert None not in (bos_id, eos_id, user_id, assistant_id), "Special tokens missing in tokenizer!"
    print(f"Tokenizer loaded (Vocab size: {tokenizer.get_vocab_size()})")

    records = []
    print(f"Reading instructions from {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            # Support both format types:
            # 1. {"question": "...", "answer": "..."}
            # 2. {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            q, a = "", ""
            if "question" in data and "answer" in data:
                q = data["question"].strip()
                a = data["answer"].strip()
            elif "messages" in data:
                msgs = data["messages"]
                for i in range(len(msgs) - 1):
                    if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                        q = msgs[i].get("content", "").strip()
                        a = msgs[i + 1].get("content", "").strip()
                        break
            elif "instruction" in data and "output" in data:
                # Alpaca format
                inst = data["instruction"].strip()
                inp = data.get("input", "").strip()
                q = f"{inst}\n{inp}".strip() if inp else inst
                a = data["output"].strip()

            if q and a:
                records.append((q, a))

    print(f"Found {len(records)} total instruction pairs.")

    processed_examples = []
    skipped_count = 0

    for q, a in records:
        q_ids = tokenizer.encode(q, add_special_tokens=False).ids
        a_ids = tokenizer.encode(a, add_special_tokens=False).ids

        # Build prompt: [BOS] [USER] <question> [ASSISTANT] <answer> [EOS]
        prompt_tokens = [bos_id, user_id] + q_ids + [assistant_id]
        response_tokens = a_ids + [eos_id]

        full_tokens = prompt_tokens + response_tokens

        # If too long, truncate from the answer
        if len(full_tokens) > max_seq_len:
            available_for_ans = max_seq_len - len(prompt_tokens) - 1
            if available_for_ans < 5:
                # If prompt is overwhelmingly long, skip
                skipped_count += 1
                continue
            response_tokens = a_ids[:available_for_ans] + [eos_id]
            full_tokens = prompt_tokens + response_tokens

        # Loss mask: 0 on prompt/user tokens, 1 strictly on assistant tokens
        loss_mask = ([0] * len(prompt_tokens)) + ([1] * len(response_tokens))

        processed_examples.append({
            "input_ids": full_tokens,
            "loss_mask": loss_mask
        })

    print(f"Successfully processed {len(processed_examples)} examples (skipped {skipped_count} overlong).")

    # Split train/validation
    random.shuffle(processed_examples)
    val_size = max(1, int(len(processed_examples) * val_ratio))
    val_set = processed_examples[:val_size]
    train_set = processed_examples[val_size:]

    train_out = output_dir / "sft_train.pt"
    val_out = output_dir / "sft_val.pt"

    torch.save(train_set, train_out)
    torch.save(val_set, val_out)

    print(f"Saved {len(train_set)} train samples -> {train_out}")
    print(f"Saved {len(val_set)} val samples   -> {val_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/train.jsonl")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    prepare_sft(
        input_file=Path(args.input),
        tokenizer_file=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio
    )
