import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import MyLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning-fast MyLLM Pretraining")
    parser.add_argument("--train_file", type=str, default=str(ROOT / "data" / "pretraining" / "fineweb_train.bin"))
    parser.add_argument("--val_file", type=str, default=str(ROOT / "data" / "pretraining" / "fineweb_val.bin"))
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=3000, help="Total training steps (e.g. 2000-5000)")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    return parser.parse_args()


def load_tokens_to_ram(path: Path):
    """Load token binary directly into memory for zero disk-wait latency."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Loading {path.name} into RAM...")
    data = np.fromfile(str(path), dtype=np.uint16)
    tensor = torch.from_numpy(data.astype(np.int64))
    print(f"Loaded {len(tensor):,} tokens into RAM ({tensor.nbytes / 1024**2:.1f} MB)")
    return tensor


def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    max_idx = len(tokens) - block_size - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def main():
    args = parse_args()

    train_file = Path(args.train_file)
    val_file = Path(args.val_file)

    # Fallback to wikitext if fineweb_train.bin is not created yet
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
    print("=" * 65)
    print("LIGHTNING-FAST GPU TRAINING")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        print("VRAM:", vram, "GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_tokens = load_tokens_to_ram(train_file)
    val_tokens = load_tokens_to_ram(val_file)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    print(f"Batch size: {args.batch_size} | Block size: {args.block_size}")
    print(f"Total training steps: {args.max_steps:,} (~{args.max_steps * args.batch_size * args.block_size:,} tokens)")

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

    def get_lr(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(progress, 1.0)
        return args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def get_orig_model():
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def save_checkpoint(path, step, val_loss):
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
            "step": step,
            "val_loss": val_loss
        }
        torch.save(checkpoint, path)

    @torch.inference_mode()
    def evaluate():
        was_training = model.training
        model.eval()
        total_loss = 0.0
        for _ in range(args.eval_batches):
            x, y = get_batch(val_tokens, args.batch_size, args.block_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            total_loss += loss.item()
        if was_training:
            model.train()
        return total_loss / args.eval_batches

    print("=" * 65)
    print("STARTING TRAINING")
    print("=" * 65)

    best_val_loss = float("inf")
    model.train()
    running_loss = 0.0
    start_time = time.time()
    t_prev = start_time

    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_tokens, args.batch_size, args.block_size, device)

        lr = get_lr(step)
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

        running_loss += loss.detach().item()

        if step % args.log_every == 0 or step == 1:
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            denom = 1 if step == 1 else args.log_every
            avg_loss = running_loss / denom
            toks_per_sec = (args.batch_size * args.block_size * denom) / max(1e-5, dt)
            eta_mins = (args.max_steps - step) * (dt / denom) / 60
            print(
                f"Step {step:>5}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.5g} | "
                f"{toks_per_sec:,.0f} tok/s | "
                f"ETA: {eta_mins:.1f}m"
            )
            running_loss = 0.0

        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss = evaluate()
            print(f"\n--- [Step {step}] Validation Loss: {val_loss:.4f} ---")
            save_checkpoint(latest_path, step, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_path, step, val_loss)
                print(f"*** Saved New Best Checkpoint ({best_val_loss:.4f}) ***\n")

    total_time_mins = (time.time() - start_time) / 60
    print("=" * 65)
    print(f"TRAINING COMPLETE IN {total_time_mins:.2f} MINUTES!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved at: {best_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
