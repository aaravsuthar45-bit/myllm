
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from model.transformer import MyLLM


# -----------------------------
# Paths and configuration
# -----------------------------


TRAIN_PATH = ROOT / "data" / "processed" / "train.pt"
VAL_PATH = ROOT / "data" / "processed" / "validation.pt"
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 1.0
BLOCK_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Dataset
# -----------------------------

class TextDataset(Dataset):
    def __init__(self, path):
        self.examples = torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(examples, pad_id):
    max_len = max(len(item["input_ids"]) for item in examples)

    input_ids = []
    loss_masks = []

    for item in examples:
        ids = list(item["input_ids"])
        mask = list(item["loss_mask"])

        padding = max_len - len(ids)
        ids += [pad_id] * padding
        mask += [0] * padding

        input_ids.append(ids)
        loss_masks.append(mask)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    loss_masks = torch.tensor(loss_masks, dtype=torch.float32)

    # Predict each token from the tokens before it.
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    target_mask = loss_masks[:, 1:]

    return x, y, target_mask


# -----------------------------
# Loss calculation
# -----------------------------

def calculate_loss(model, batch):
    x, y, target_mask = batch

    result = model(x)
    logits = result[0] if isinstance(result, tuple) else result

    vocab_size = logits.size(-1)

    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
        reduction="none"
    ).view_as(y)

    masked_loss = token_losses * target_mask
    denominator = target_mask.sum().clamp(min=1)

    return masked_loss.sum() / denominator


# -----------------------------
# Load data
# -----------------------------

print("Device:", DEVICE)

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab_size = tokenizer.get_vocab_size()
pad_id = tokenizer.token_to_id("[PAD]")

if pad_id is None:
    raise ValueError("[PAD] token was not found in the tokenizer.")

train_dataset = TextDataset(TRAIN_PATH)
val_dataset = TextDataset(VAL_PATH)

print("Vocabulary size:", vocab_size)
print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda items: collate_batch(items, pad_id)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda items: collate_batch(items, pad_id)
)


# -----------------------------
# Create model
# -----------------------------

model = MyLLM(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=256,
    n_heads=4,
    n_layers=4,
    dropout=0.1
).to(DEVICE)

parameter_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {parameter_count:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.1
)


# -----------------------------
# Validation
# -----------------------------

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch in val_loader:
        batch = tuple(t.to(DEVICE) for t in batch)
        loss = calculate_loss(model, batch)

        total_loss += loss.item()
        batches += 1

    model.train()
    return total_loss / max(batches, 1)


# -----------------------------
# Training
# -----------------------------

best_val_loss = float("inf")

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_train_loss = 0.0
    batches = 0

    for batch in train_loader:
        batch = tuple(t.to(DEVICE) for t in batch)

        optimizer.zero_grad(set_to_none=True)

        loss = calculate_loss(model, batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )

        optimizer.step()

        total_train_loss += loss.item()
        batches += 1

    train_loss = total_train_loss / max(batches, 1)
    val_loss = evaluate()

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Validation loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "block_size": BLOCK_SIZE,
                "n_embd": 256,
                "n_heads": 4,
                "n_layers": 4,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            },
            checkpoint_path
        )

        print("  Saved best checkpoint.")

print("\nTraining finished.")
print("Best validation loss:", round(best_val_loss, 4))
print("Checkpoint:", CHECKPOINT_DIR / "best_model.pt")
