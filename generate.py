import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.transformer import MyLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with trained MyLLM")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    return parser.parse_args()


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=0.8, top_k=40):
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context to block_size
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        # Focus only on the last time step
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Optional top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)
    return idx


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        fallback = ROOT / "checkpoints" / "best_model.pt"
        if fallback.exists():
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint} or {fallback}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    vocab_size = config.get("vocab_size", checkpoint.get("vocab_size", 8000))
    block_size = config.get("block_size", checkpoint.get("block_size", 128))
    n_embd = config.get("n_embd", checkpoint.get("n_embd", 256))
    n_heads = config.get("n_heads", checkpoint.get("n_heads", 4))
    n_layers = config.get("n_layers", checkpoint.get("n_layers", 4))

    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        fallback_tok = ROOT / "data" / "tokenizer" / "tokenizer.json"
        if fallback_tok.exists():
            tok_path = fallback_tok
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")

    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    model = MyLLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        prompt_ids = [0]

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print("\n" + "=" * 60)
    print(f"PROMPT: {args.prompt}")
    print("=" * 60)

    out_tokens = generate(
        model=model,
        idx=input_tensor,
        max_new_tokens=args.max_tokens,
        block_size=block_size,
        temperature=args.temperature,
        top_k=args.top_k
    )

    generated_text = tokenizer.decode(out_tokens[0].tolist())
    print("\nGENERATED TEXT:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)


if __name__ == "__main__":
    main()
