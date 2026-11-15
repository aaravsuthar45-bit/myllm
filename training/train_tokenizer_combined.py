
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent

WIKI_FILE = ROOT / "data" / "raw" / "wikitext_train.txt"
GEMINI_FILE = ROOT / "data" / "train.jsonl"
CLAUDE_FILE = ROOT / "data" / "processed" / "claude_sft_clean.jsonl"

OUTPUT_DIR = ROOT / "data" / "tokenizer"
OUTPUT_FILE = OUTPUT_DIR / "tokenizer_combined.json"

VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[USER]",
    "[ASSISTANT]"
]


def text_iterator():
    # General English text
    print("Reading WikiText...")
    with WIKI_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

    # Gemini question-answer examples
    print("Reading Gemini conversations...")
    with GEMINI_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if question:
                yield question
            if answer:
                yield answer

    # Claude multi-turn conversations
    print("Reading Claude conversations...")
    with CLAUDE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            for message in item.get("messages", []):
                content = message.get("content", "").strip()
                if content:
                    yield content


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print("Training combined BPE tokenizer...")
    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer
    )

    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer training complete!")
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Saved to:", OUTPUT_FILE)

    test = "Hello! This is my new language model."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)

    print("\nTest text:", test)
    print("Token IDs:", encoded.ids)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()
