
from pathlib import Path
from array import array
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

TEXT_FILE = ROOT / "data" / "raw" / "wikitext_train.txt"
TOKENIZER_FILE = ROOT / "data" / "tokenizer" / "tokenizer_combined.json"
OUTPUT_DIR = ROOT / "data" / "pretraining"

VALIDATION_FRACTION = 0.05
FLUSH_SIZE = 100_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
eos_id = tokenizer.token_to_id("[EOS]")

if eos_id is None:
    raise ValueError("[EOS] token is missing from tokenizer")

train_path = OUTPUT_DIR / "wikitext_train.bin"
val_path = OUTPUT_DIR / "wikitext_val.bin"

# Split by original line position, keeping validation at the end.
total_lines = sum(1 for _ in TEXT_FILE.open(encoding="utf-8"))
split_line = int(total_lines * (1 - VALIDATION_FRACTION))

train_tokens = 0
val_tokens = 0
train_buffer = array("H")
val_buffer = array("H")


def flush(buffer, path):
    if buffer:
        with path.open("ab") as f:
            buffer.tofile(f)
        buffer = array("H")
    return buffer


# Start fresh so rerunning the script doesn't append duplicate data.
train_path.write_bytes(b"")
val_path.write_bytes(b"")

print("Tokenizing WikiText...")
print("Total source lines:", total_lines)
print("Training lines:", split_line)
print("Validation lines:", total_lines - split_line)

with TEXT_FILE.open("r", encoding="utf-8") as source:
    for line_number, line in enumerate(source):
        text = line.strip()

        if not text:
            continue

        ids = tokenizer.encode(text).ids
        if not ids:
            continue

        ids.append(eos_id)

        if line_number < split_line:
            train_buffer.extend(ids)
            train_tokens += len(ids)

            if len(train_buffer) >= FLUSH_SIZE:
                train_buffer = flush(train_buffer, train_path)
        else:
            val_buffer.extend(ids)
            val_tokens += len(ids)

            if len(val_buffer) >= FLUSH_SIZE:
                val_buffer = flush(val_buffer, val_path)

        if (line_number + 1) % 10_000 == 0:
            print(f"Processed {line_number + 1:,}/{total_lines:,} lines")

# Write remaining tokens.
flush(train_buffer, train_path)
flush(val_buffer, val_path)

print("\nPretraining data ready!")
print("Training tokens:", f"{train_tokens:,}")
print("Validation tokens:", f"{val_tokens:,}")
print("Training file MB:", round(train_path.stat().st_size / 1024**2, 2))
print("Validation file MB:", round(val_path.stat().st_size / 1024**2, 2))
print("Saved in:", OUTPUT_DIR)
