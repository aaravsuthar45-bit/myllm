
import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

from datasets import load_dataset
from ftfy import fix_text
from tqdm import tqdm


def clean_text(text):
    """Clean text without destroying useful punctuation or symbols."""
    if not isinstance(text, str):
        return ""

    text = fix_text(text)
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove control characters except whitespace
    text = "".join(
        c for c in text
        if c.isprintable() or c in "\n\t"
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_hash(text):
    """Generate a unique document fingerprint."""
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def estimate_tokens(text):
    """Rough estimate only, not actual tokenizer tokens."""
    return len(text.split())


def initialize_database(db_path):
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            hash TEXT PRIMARY KEY
        )
    """)

    conn.commit()
    return conn


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(
        args.output_dir, "training_data.jsonl"
    )

    db_file = os.path.join(
        args.output_dir, "dedup.sqlite"
    )

    conn = initialize_database(db_file)

    print("\nLoading dataset...")
    print("Dataset:", args.dataset)
    print("Configuration:", args.config)

    dataset = load_dataset(
        args.dataset,
        name=args.config or None,
        split=args.split,
        streaming=True
    )

    collected = 0
    skipped = 0
    total_chars = 0
    total_tokens_estimate = 0

    start_time = time.time()

    # Append mode allows collection to resume.
    with open(
        output_file,
        "a",
        encoding="utf-8"
    ) as output:

        progress = tqdm(
            total=args.max_docs,
            desc="Collecting documents"
        )

        for item in dataset:

            if collected >= args.max_docs:
                break

            raw_text = item.get(args.text_field, "")

            text = clean_text(raw_text)

            # Quality filter
            if len(text) < args.min_chars:
                skipped += 1
                continue

            if len(text) > args.max_chars:
                text = text[:args.max_chars]

            # Remove duplicates
            fingerprint = get_hash(text)

            try:
                conn.execute(
                    "INSERT INTO documents (hash) VALUES (?)",
                    (fingerprint,)
                )
            except sqlite3.IntegrityError:
                skipped += 1
                continue

            token_estimate = estimate_tokens(text)

            record = {
                "text": text,
                "source": args.dataset,
                "config": args.config,
                "language": args.language,
                "hash": fingerprint,
                "estimated_tokens": token_estimate,
                "collected_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            collected += 1
            total_chars += len(text)
            total_tokens_estimate += token_estimate

            if collected % 100 == 0:
                output.flush()
                conn.commit()

            progress.update(1)

        progress.close()

        output.flush()

    conn.commit()
    conn.close()

    elapsed = time.time() - start_time

    print("\n========== COLLECTION REPORT ==========")
    print("New documents:", collected)
    print("Skipped documents:", skipped)
    print("Characters:", total_chars)
    print("Estimated tokens:", total_tokens_estimate)
    print("Time taken:", round(elapsed, 2), "seconds")
    print("Output:", output_file)
    print("=======================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Automated LLM dataset collector"
    )

    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu"
    )

    parser.add_argument(
        "--config",
        default="sample-10BT"
    )

    parser.add_argument(
        "--split",
        default="train"
    )

    parser.add_argument(
        "--text-field",
        default="text"
    )

    parser.add_argument(
        "--language",
        default="en"
    )

    parser.add_argument(
        "--output-dir",
        default="data"
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=200
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=100000
    )

    args = parser.parse_args()
    main(args)
