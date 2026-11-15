"""
Tokenize FineWeb / raw text documents into binary memory-mapped format for pretraining.
Produces uint16 binary files (.bin) ready for fast streaming in pretrain.py.
"""

import argparse
import json
from pathlib import Path
from array import array
from tokenizers import Tokenizer
from tqdm import tqdm


def prepare_pretraining_bin(
    input_file: Path,
    tokenizer_file: Path,
    output_dir: Path,
    val_fraction: float = 0.05,
    flush_size: int = 200_000,
    text_key: str = "text"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    eos_id = tokenizer.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("Tokenizer must contain an [EOS] token.")

    train_path = output_dir / "fineweb_train.bin"
    val_path = output_dir / "fineweb_val.bin"

    train_path.write_bytes(b"")
    val_path.write_bytes(b"")

    # Count lines first (or estimate)
    print(f"Counting lines in {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    split_line = int(total_lines * (1.0 - val_fraction))
    print(f"Total lines: {total_lines} | Train: {split_line} | Val: {total_lines - split_line}")

    train_tokens = 0
    val_tokens = 0
    train_buffer = array("H")
    val_buffer = array("H")

    def flush(buf, target_file):
        if buf:
            with open(target_file, "ab") as f:
                buf.tofile(f)
            buf = array("H")
        return buf

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc="Tokenizing")):
            line = line.strip()
            if not line:
                continue

            # Support both raw line text and JSON lines
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    text = data.get(text_key, "")
                except Exception:
                    text = line
            else:
                text = line

            if not text:
                continue

            ids = tokenizer.encode(text).ids
            if not ids:
                continue

            # Append EOS
            ids.append(eos_id)

            if idx < split_line:
                train_buffer.extend(ids)
                train_tokens += len(ids)
                if len(train_buffer) >= flush_size:
                    train_buffer = flush(train_buffer, train_path)
            else:
                val_buffer.extend(ids)
                val_tokens += len(ids)
                if len(val_buffer) >= flush_size:
                    val_buffer = flush(val_buffer, val_path)

    flush(train_buffer, train_path)
    flush(val_buffer, val_path)

    print("Complete!")
    print(f"Train Tokens: {train_tokens:,} -> {train_path}")
    print(f"Val Tokens:   {val_tokens:,} -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/training_data.jsonl")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--output_dir", type=str, default="data/pretraining")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    args = parser.parse_args()

    prepare_pretraining_bin(
        input_file=Path(args.input),
        tokenizer_file=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        val_fraction=args.val_fraction
    )
