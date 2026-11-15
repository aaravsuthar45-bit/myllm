
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "train.jsonl"

MODEL_NAME = "gemini-3.5-flash"

TOTAL_EXAMPLES = 100
BATCH_SIZE = 5

CATEGORIES = [
    "general knowledge",
    "mathematics and logical reasoning",
    "science and nature",
    "Python programming and computer science",
    "English grammar and writing",
    "everyday practical problem solving",
    "history and geography",
    "technology and computers",
    "educational tutoring",
    "creative and analytical thinking",
]


# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Set it in PowerShell first."
    )

client = genai.Client(api_key=api_key)

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD EXISTING DATA (SUPPORT RESUMING)
# --------------------------------------------------

existing_questions = set()
existing_count = 0

if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            existing_questions.add(
                item["question"].strip().lower()
            )
            existing_count += 1

print("Existing examples:", existing_count)

if existing_count >= TOTAL_EXAMPLES:
    print("Dataset already has enough examples.")
    raise SystemExit(0)


# --------------------------------------------------
# GENERATE DATA
# --------------------------------------------------

remaining = TOTAL_EXAMPLES - existing_count
batch_number = 0

while remaining > 0:

    category = CATEGORIES[
        batch_number % len(CATEGORIES)
    ]

    amount = min(BATCH_SIZE, remaining)

    prompt = f"""
You are creating high-quality training data for a small
English-language AI assistant.

Generate exactly {amount} different question-and-answer
examples about: {category}.

Requirements:

1. Questions must be natural and useful.
2. Answers must be accurate, clear and self-contained.
3. Explain reasoning when it helps understanding.
4. Avoid repetitive questions.
5. Use correct English.
6. Do not invent facts or pretend uncertain information
   is certain.
7. Answers should generally be 60 to 150 words.
8. Include a mixture of easy and moderately difficult tasks.
9. Do not mention that you are generating training data.
10. Do not include markdown fences around the JSON.

Generate diverse examples suitable for teaching a
general-purpose conversational AI.
"""

    print(
        f"\nGenerating batch {batch_number + 1} "
        f"| Category: {category}"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema={
                "type": "OBJECT",
                "properties": {
                    "examples": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "question": {
                                    "type": "STRING"
                                },
                                "answer": {
                                    "type": "STRING"
                                }
                            },
                            "required": [
                                "question",
                                "answer"
                            ]
                        }
                    }
                },
                "required": ["examples"]
            },
            temperature=0.8,
            max_output_tokens=8192,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    result = json.loads(response.text)
    examples = result["examples"]

    saved = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

        for item in examples:

            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                continue

            key = question.lower()

            if key in existing_questions:
                continue

            record = {
                "question": question,
                "answer": answer,
                "category": category,
                "source": "gemini"
            }

            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            existing_questions.add(key)
            saved += 1

        f.flush()

    remaining = max(
        0,
        TOTAL_EXAMPLES - len(existing_questions)
    )

    print("Saved this batch:", saved)
    print("Total examples:", len(existing_questions))

    batch_number += 1

    if remaining > 0:
        time.sleep(2)


print("\nDATA GENERATION COMPLETE")
print("Dataset:", OUTPUT_FILE)
print("Total examples:", len(existing_questions))
