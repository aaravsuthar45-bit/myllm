import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import MyLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain MyLLM")
    parser.add_argument("--train_file", type=str, default=str(ROOT / "data" / "pretraining" / "fineweb_train.bin"))
    parser.add_argument("--val_file", type=str, default=str(ROOT / "data" / "pretraining" / "fineweb_val.bin"))
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    return parser.parse_args()


class FastTokenDataset(Dataset):
    """
    Random-stride sample dataset over memory-mapped token binary.
    Instead of length = len(tokens) - block_size (which creates millions of 1-token-offset duplicate steps),
    we step by block_size so each token is trained on once per epoch.
    This provides an immediate 128x speedup per epoch with proper data coverage.
    """
    def __init__(self, path: Path, block_size: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Binary token file not found: {self.path}")

        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        # Step non-overlapping chunks (stride = block_size)
        self.num_chunks = (len(self.tokens) - 1) // block_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = np.array(
            self.tokens[start : start + self.block_size + 1],
            dtype=np.int64,
            copy=True
        )
        data = torch.from_numpy(chunk)
        return data[:-1], data[1:]


def main():
    args = parse_args()

    # Fallback to wikitext if fineweb_train.bin is not created yet
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    if not train_file.exists():
        fallback_train = ROOT / "data" / "pretraining" / "wikitext_train.bin"
        fallback_val = ROOT / "data" / "pretraining" / "wikitext_val.bin"
        if fallback_train.exists():
            print(f"[Notice] {train_file} not found. Falling back to {fallback_train}")
            train_file = fallback_train
            val_file = fallback_val

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "pretrained_best.pt"
    latest_path = checkpoint_dir / "pretrained_latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DEVICE & TRAINING CONFIGURATION")
    print("=" * 60)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        print("VRAM:", vram, "GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Loading datasets from {train_file.name}...")
    train_dataset = FastTokenDataset(train_file, args.block_size)
    val_dataset = FastTokenDataset(val_file, args.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=use_amp,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=use_amp,
        drop_last=True
    )

    print(f"Train samples (chunks): {len(train_dataset):,} | Val samples: {len(val_dataset):,}")
    print(f"Tokens per sequence: {args.block_size} | Batch size: {args.batch_size}")
    print(f"Tokens processed per step: {args.batch_size * args.block_size:,}")

    model = MyLLM(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    if args.compile and device.type == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda")
        )
        print("Using fused AdamW optimizer")
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    def get_lr(step, total_steps):
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
        if step < warmup_steps:
            return args.learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def get_orig_model():
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def save_checkpoint(path, epoch, step, val_loss):
        checkpoint = {
            "model_state_dict": get_orig_model().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": {
                "vocab_size": args.vocab_size,
                "block_size": args.block_size,
                "n_embd": args.n_embd,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout
            },
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss
        }
        torch.save(checkpoint, path)

    @torch.inference_mode()
    def evaluate():
        was_training = model.training
        model.eval()
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        batches = 0
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= args.eval_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            total_loss += loss.detach().float()
            batches += 1
        if was_training:
            model.train()
        return (total_loss / max(1, batches)).item()

    best_val_loss = float("inf")
    global_step = 0
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    print("=" * 60)
    print(f"Starting Training: {args.epochs} epochs | {steps_per_epoch} steps/epoch | Total: {total_steps} steps")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                denom = 1 if global_step == 1 else args.log_every
                avg_loss = (running_loss / denom).item()
                tokens_sec = (args.batch_size * args.block_size * denom) / max(1e-5, (time.time() - start_time) % 60)
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.6g}"
                )
                running_loss = 0.0

            if global_step % args.eval_every == 0:
                val_loss = evaluate()
                print(f"\n--- Validation Loss: {val_loss:.4f} ---")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(best_path, epoch + 1, global_step, val_loss)
                    print(f"Saved new best checkpoint to {best_path}\n")

        # End of epoch validation
        val_loss = evaluate()
        print(f"\nEpoch {epoch + 1} Done | Validation Loss: {val_loss:.4f}")
        save_checkpoint(latest_path, epoch + 1, global_step, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, epoch + 1, global_step, val_loss)
            print(f"Saved best checkpoint to {best_path}")

    elapsed_mins = (time.time() - start_time) / 60
    print("=" * 60)
    print(f"PRETRAINING COMPLETE in {elapsed_mins:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
