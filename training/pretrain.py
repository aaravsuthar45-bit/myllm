
import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import MyLLM


# ============================================================
# SETTINGS
# ============================================================

VOCAB_SIZE = 8000
BLOCK_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

WARMUP_RATIO = 0.05

EVAL_EVERY = 1000
EVAL_BATCHES = 20
LOG_EVERY = 50

N_EMBD = 256
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1

NUM_WORKERS = 0

# Enable only after benchmarking the uncompiled version.
USE_COMPILE = False

TRAIN_FILE = ROOT / "data" / "pretraining" / "wikitext_train.bin"
VAL_FILE = ROOT / "data" / "pretraining" / "wikitext_val.bin"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_PATH = CHECKPOINT_DIR / "pretrained_best.pt"
LATEST_PATH = CHECKPOINT_DIR / "pretrained_latest.pt"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("DEVICE INFORMATION")
print("=" * 55)

print("Device:", device)

if device.type == "cuda":

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

    # Faster matrix multiplication on supported GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Shapes are fixed during training.
    torch.backends.cudnn.benchmark = True


USE_AMP = device.type == "cuda"

# RTX 3060: FP16 is our default.
AMP_DTYPE = torch.float16

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ============================================================
# DATASET
# ============================================================

class TokenDataset(Dataset):

    def __init__(self, path):

        self.tokens = np.memmap(
            path,
            dtype=np.uint16,
            mode="r"
        )

        self.length = len(self.tokens) - BLOCK_SIZE

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # Keep data as uint16 on CPU.
        # Convert to int64 after transferring to GPU.

        chunk = np.array(
            self.tokens[index:index + BLOCK_SIZE + 1],
            dtype=np.uint16,
            copy=True
        )

        data = torch.from_numpy(chunk)

        return data[:-1], data[1:]


train_dataset = TokenDataset(TRAIN_FILE)
val_dataset = TokenDataset(VAL_FILE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=True
)

print("\nTraining sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

model = MyLLM(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT
).to(device)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("\nParameters:", f"{parameter_count:,}")


# Optional compilation.
if USE_COMPILE and device.type == "cuda":

    print("Compiling model...")

    model = torch.compile(
        model,
        mode="reduce-overhead"
    )


# ============================================================
# OPTIMIZER
# ============================================================

try:

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=(device.type == "cuda")
    )

    print("Using fused AdamW")

except (TypeError, RuntimeError):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print("Using standard AdamW")


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_original_model():

    if hasattr(model, "_orig_mod"):
        return model._orig_mod

    return model


def save_checkpoint(path, epoch, step, val_loss):

    checkpoint = {
        "model_state_dict": get_original_model().state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT
        },

        "epoch": epoch,
        "step": step,
        "val_loss": val_loss
    }

    torch.save(checkpoint, path)


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate():

    was_training = model.training
    model.eval()

    total_loss = torch.zeros(
        (),
        device=device,
        dtype=torch.float32
    )

    batches = 0

    for batch_index, (x, y) in enumerate(val_loader):

        if batch_index >= EVAL_BATCHES:
            break

        x = x.to(
            device,
            non_blocking=True
        ).long()

        y = y.to(
            device,
            non_blocking=True
        ).long()

        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP
        ):

            _, loss = model(x, y)

        total_loss += loss.detach().float()

        batches += 1

    if was_training:
        model.train()

    return (
        (total_loss / max(1, batches)).item()
    )


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_lr(step, total_steps):

    warmup_steps = max(
        1,
        int(total_steps * WARMUP_RATIO)
    )

    if step < warmup_steps:

        return LEARNING_RATE * (
            step + 1
        ) / warmup_steps

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps
    )

    progress = min(progress, 1.0)

    return LEARNING_RATE * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
global_step = 0

steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS

print("\n" + "=" * 55)
print("TRAINING")
print("=" * 55)

print("Batch size:", BATCH_SIZE)
print("Steps per epoch:", steps_per_epoch)
print("Total steps:", total_steps)
print("Mixed precision:", USE_AMP)
print("Compile:", USE_COMPILE)

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_index, (x, y) in enumerate(train_loader):

        # ----------------------------------------------------
        # TRANSFER DATA
        # ----------------------------------------------------

        x = x.to(
            device,
            non_blocking=True
        ).long()

        y = y.to(
            device,
            non_blocking=True
        ).long()

        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------

        lr = get_lr(
            global_step,
            total_steps
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP
        ):

            _, loss = model(x, y)

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # LOSS TRACKING
        # ----------------------------------------------------

        running_loss += loss.detach()

        global_step += 1

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        if global_step % LOG_EVERY == 0:

            avg_loss = (
                running_loss / LOG_EVERY
            ).item()

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {global_step}/{total_steps} | "
                f"Loss {avg_loss:.4f} | "
                f"LR {lr:.6g}"
            )

            running_loss = 0.0

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if global_step % EVAL_EVERY == 0:

            val_loss = evaluate()

            print(
                f"Validation loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_checkpoint(
                    BEST_PATH,
                    epoch + 1,
                    global_step,
                    val_loss
                )

                print("Saved best checkpoint!")

    # --------------------------------------------------------
    # END OF EPOCH
    # --------------------------------------------------------

    val_loss = evaluate()

    print(
        f"\nEpoch {epoch + 1} complete | "
        f"Validation loss: {val_loss:.4f}"
    )

    save_checkpoint(
        LATEST_PATH,
        epoch + 1,
        global_step,
        val_loss
    )

    print("Saved latest checkpoint!")

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_checkpoint(
            BEST_PATH,
            epoch + 1,
            global_step,
            val_loss
        )

        print("Saved best checkpoint!")

    if device.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated() / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved() / 1024**3
        )

        print(
            f"GPU memory: {allocated:.2f} GB allocated | "
            f"{reserved:.2f} GB reserved"
        )


# ============================================================
# COMPLETE
# ============================================================

elapsed = (time.time() - start_time) / 60

print("\n" + "=" * 55)
print("PRETRAINING COMPLETE")
print("=" * 55)

print("Best validation loss:", best_val_loss)
print("Time taken:", round(elapsed, 2), "minutes")
print("Best checkpoint:", BEST_PATH)
print("Latest checkpoint:", LATEST_PATH)
