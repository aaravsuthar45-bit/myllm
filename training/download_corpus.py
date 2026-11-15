
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep Hugging Face's cache on D:, not the nearly-full C: drive.
CACHE = ROOT / "data" / "hf_cache"
CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_DATASETS_CACHE"] = str(CACHE / "datasets")
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")

from datasets import load_dataset

OUTPUT = ROOT / "data" / "raw" / "wikitext_train.txt"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

MAX_LINES = 150_000
saved = 0

print("Opening WikiText-103 in streaming mode...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
    streaming=True
)

with OUTPUT.open("w", encoding="utf-8") as f:
    for row in dataset:
        text = row.get("text", "").strip()

        if len(text) < 80:
            continue

        f.write(text + "\n")
        saved += 1

        if saved % 10_000 == 0:
            print(f"Saved {saved:,} text lines")

        if saved >= MAX_LINES:
            break

print("\nCorpus download complete!")
print("Text lines saved:", saved)
print("Output:", OUTPUT)
