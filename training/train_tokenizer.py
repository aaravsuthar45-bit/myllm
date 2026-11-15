
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "train.jsonl"
OUTPUT_DIR = ROOT / "data" / "tokenizer"

VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[USER]",
    "[ASSISTANT]",
]


def load_training_text():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    texts = []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)

            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                print(f"Skipping incomplete line {line_number}")
                continue

            text = (
                "[BOS]\n"
                "[USER]\n"
                f"{question}\n"
                "[ASSISTANT]\n"
                f"{answer}\n"
                "[EOS]"
            )

            texts.append(text)

    if not texts:
        raise ValueError("No valid training examples found.")

    return texts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training examples...")
    texts = load_training_text()

    print("Examples loaded:", len(texts))

    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("\nTraining BPE tokenizer...")

    tokenizer.train_from_iterator(
        texts,
        trainer=trainer
    )

    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    print("\nTokenizer saved:", tokenizer_path)
    print("Actual vocabulary size:", tokenizer.get_vocab_size())

    # Test encoding and decoding.
    test_text = (
        "[BOS]\n[USER]\n"
        "Explain how a neural network works.\n"
        "[ASSISTANT]\n"
        "A neural network learns patterns from data.\n"
        "[EOS]"
    )

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(
        encoded.ids,
        skip_special_tokens=False
    )

    print("\nTEST")
    print("Original:", test_text)
    print("Token IDs:", encoded.ids)
    print("Token count:", len(encoded.ids))
    print("Decoded:", decoded)

    assert decoded == test_text, "Tokenizer round-trip failed!"

    print("\nTOKENIZER TEST PASSED")


if __name__ == "__main__":
    main()
