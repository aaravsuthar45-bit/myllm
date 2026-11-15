
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer
from model.transformer import MyLLM

CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and checkpoint
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

# Recreate the exact model architecture
model = MyLLM(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    n_embd=checkpoint["n_embd"],
    n_heads=checkpoint["n_heads"],
    n_layers=checkpoint["n_layers"],
    dropout=0.0
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Enter your question here
question = "Explain what artificial intelligence is."

# Build the prompt
bos_id = tokenizer.token_to_id("[BOS]")
user_id = tokenizer.token_to_id("[USER]")
assistant_id = tokenizer.token_to_id("[ASSISTANT]")

question_ids = tokenizer.encode(
    question,
    add_special_tokens=False
).ids

prompt_ids = [bos_id, user_id] + question_ids + [assistant_id]

input_ids = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=DEVICE
)

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=80
    )

# Decode only the newly generated tokens
generated_ids = output[0, len(prompt_ids):].tolist()
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\nQuestion:", question)
print("\nModel answer:")
print(answer)
