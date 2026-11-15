import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import MyLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Instruction Fine-Tuning (SFT)")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/pretrained_best.pt", help="Pretrained weights to start from")
    parser.add_argument("--train_path", type=str, default="data/processed/sft_train.pt")
    parser.add_argument("--val_path", type=str, default="data/processed/sft_val.pt")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/sft_best.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Lower learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


class SFTDataset(Dataset):
    def __init__(self, path):
        self.examples = torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_sft_batch(examples, pad_id):
    max_len = max(len(item["input_ids"]) for item in examples)
    input_ids = []
    loss_masks = []

    for item in examples:
        ids = list(item["input_ids"])
        mask = list(item["loss_mask"])
        pad_len = max_len - len(ids)

        ids += [pad_id] * pad_len
        mask += [0] * pad_len

        input_ids.append(ids)
        loss_masks.append(mask)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    loss_masks = torch.tensor(loss_masks, dtype=torch.float32)

    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    target_mask = loss_masks[:, 1:]

    return x, y, target_mask


def compute_masked_loss(model, x, y, target_mask, device, use_amp):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    target_mask = target_mask.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits, _ = model(x)
        vocab_size = logits.size(-1)
        token_losses = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="none"
        ).view_as(y)

        masked_loss = token_losses * target_mask
        denom = target_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denom


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("INSTRUCTION FINE-TUNING (SFT)")
    print("=" * 65)
    print("Device:", device)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0

    # Load base pretrained checkpoint
    base_ckpt_path = Path(args.base_checkpoint)
    if not base_ckpt_path.exists():
        fallback = ROOT / "checkpoints" / "best_model.pt"
        if fallback.exists():
            base_ckpt_path = fallback
        else:
            raise FileNotFoundError(f"Base checkpoint not found at {args.base_checkpoint}")

    print(f"Loading pretrained base model from: {base_ckpt_path}")
    checkpoint = torch.load(base_ckpt_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    vocab_size = config.get("vocab_size", checkpoint.get("vocab_size", 8000))
    block_size = config.get("block_size", checkpoint.get("block_size", 128))
    n_embd = config.get("n_embd", checkpoint.get("n_embd", 256))
    n_heads = config.get("n_heads", checkpoint.get("n_heads", 4))
    n_layers = config.get("n_layers", checkpoint.get("n_layers", 4))

    model = MyLLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.05
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print("Pretrained weights loaded successfully!")

    train_dataset = SFTDataset(args.train_path)
    val_dataset = SFTDataset(args.val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda items: collate_sft_batch(items, pad_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda items: collate_sft_batch(items, pad_id)
    )

    print(f"Training pairs: {len(train_dataset)} | Validation pairs: {len(val_dataset)}")

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        batches = 0
        for x, y, mask in val_loader:
            loss = compute_masked_loss(model, x, y, mask, device, use_amp)
            total_loss += loss.item()
            batches += 1
        model.train()
        return total_loss / max(1, batches)

    out_path = Path(args.output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    print("\nStarting fine-tuning...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for x, y, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_masked_loss(model, x, y, mask, device, use_amp)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / max(1, batches)
        val_loss = evaluate()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train SFT Loss: {train_loss:.4f} | "
            f"Val SFT Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_embd": n_embd,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                },
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }
            torch.save(save_dict, out_path)
            print(f" -> Saved improved SFT checkpoint to {out_path}")

    print("\n" + "=" * 65)
    print(f"INSTRUCTION FINE-TUNING FINISHED! Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved chatbot model: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
