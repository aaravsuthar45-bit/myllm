
import os
import json
from pathlib import Path

# Input folder
INPUT_DIR = Path(
    r"D:\minecraft rtx\myllm\MY-LLM\data\conversations-000\claude-readable"
)

# Output JSON file
OUTPUT_FILE = INPUT_DIR / "combined_conversations.json"


def combine_markdown_files():
    conversations = []

    # Find all .md files recursively
    md_files = sorted(INPUT_DIR.rglob("*.md"))

    print(f"Found {len(md_files)} Markdown files.")

    for index, file_path in enumerate(md_files, start=1):
        try:
            content = file_path.read_text(encoding="utf-8")

            conversations.append({
                "id": index,
                "filename": file_path.name,
                "path": str(file_path.relative_to(INPUT_DIR)),
                "content": content
            })

            print(f"[{index}/{len(md_files)}] Processed: {file_path.name}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Save everything into one JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            conversations,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n================================")
    print("COMPLETED!")
    print(f"Files combined: {len(conversations)}")
    print(f"Output: {OUTPUT_FILE}")
    print("================================")


if __name__ == "__main__":
    combine_markdown_files()
