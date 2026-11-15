"""
High-performance Streaming Tokenizer for Large Datasets (JSON array, JSONL, or raw text).
Streams 1GB+ files item-by-item with minimal memory usage via ijson/json/line streaming,
and produces uint16 binary files (.bin) ready for pretraining.
"""

import argparse
import json
from pathlib import Path
from array import array
from tokenizers import Tokenizer
from tqdm import tqdm


def stream_json_array_or_jsonl(file_path: Path):
    """
    Generator that streams items from either a standard JSON array file [...]
    or a JSON Lines file (.jsonl) without loading the whole file into RAM.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # Peek at the first non-whitespace character
        pos = f.tell()
        first_char = ""
        while True:
            c = f.read(1)
            if not c:
                break
            if not c.isspace():
                first_char = c
                break

        f.seek(pos)

        if first_char == "[":
            # JSON array format: stream using an iterative decoder or chunking
            try:
                import ijson
                print("Using ijson streaming parser for large JSON array...")
                for item in ijson.items(f, "item"):
                    if isinstance(item, dict):
                        yield item.get("text", "")
                    elif isinstance(item, str):
                        yield item
            except ImportError:
                print("Streaming JSON array using chunked object parser...")
                # Fallback streaming parser for JSON array without external dependencies
                buffer = ""
                in_string = False
                escape = False
                depth = 0
                obj_start = -1

                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    buffer += chunk

                    # Parse top-level objects inside array
                    i = 0
                    while i < len(buffer):
                        ch = buffer[i]
                        if not in_string:
                            if ch == '"':
                                in_string = True
                            elif ch == '{':
                                if depth == 0:
                                    obj_start = i
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0 and obj_start != -1:
                                    obj_str = buffer[obj_start : i + 1]
                                    try:
                                        d = json.loads(obj_str)
                                        yield d.get("text", "")
                                    except Exception:
                                        pass
                                    buffer = buffer[i + 1 :]
                                    i = -1
                                    obj_start = -1
                        else:
                            if escape:
                                escape = False
                            elif ch == '\\':
                                escape = True
                            elif ch == '"':
                                in_string = False
                        i += 1
        else:
            # Standard JSONL (one JSON object per line)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") and line.endswith("}"):
                    try:
                        d = json.loads(line)
                        yield d.get("text", "")
                    except Exception:
                        yield line
                else:
                    yield line


def prepare_pretraining_bin(
    input_file: Path,
    tokenizer_file: Path,
    output_dir: Path,
    val_fraction: float = 0.05,
    flush_size: int = 200_000,
    max_samples: int = None
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

    print(f"Streaming and tokenizing from {input_file}...")

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

    count = 0
    val_interval = int(1.0 / val_fraction) if val_fraction > 0 else 20

    for text in tqdm(stream_json_array_or_jsonl(input_file), desc="Tokenizing"):
        if not text:
            continue

        ids = tokenizer.encode(text).ids
        if not ids:
            continue
        ids.append(eos_id)

        # Periodic split (every val_interval item goes to val)
        is_val = (count % val_interval == 0)

        if not is_val:
            train_buffer.extend(ids)
            train_tokens += len(ids)
            if len(train_buffer) >= flush_size:
                train_buffer = flush(train_buffer, train_path)
        else:
            val_buffer.extend(ids)
            val_tokens += len(ids)
            if len(val_buffer) >= flush_size:
                val_buffer = flush(val_buffer, val_path)

        count += 1
        if max_samples and count >= max_samples:
            break

    flush(train_buffer, train_path)
    flush(val_buffer, val_path)

    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE!")
    print(f"Total documents: {count:,}")
    print(f"Train Tokens:    {train_tokens:,} -> {train_path}")
    print(f"Val Tokens:      {val_tokens:,} -> {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/training_data.json")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--output_dir", type=str, default="data/pretraining")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    in_file = Path(args.input)
    if not in_file.exists():
        fallback = Path("data/training_data.jsonl")
        if fallback.exists():
            in_file = fallback

    prepare_pretraining_bin(
        input_file=in_file,
        tokenizer_file=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        val_fraction=args.val_fraction,
        max_samples=args.max_samples
    )
