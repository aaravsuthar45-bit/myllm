"""
Upload your trained MyLLM model checkpoints and tokenizers to Hugging Face Hub.
Supports uploading as a private or public model repository.
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent


def upload_model(repo_id: str, checkpoint_path: str, tokenizer_path: str, private: bool = True, token: str = None):
    api = HfApi(token=token)

    print(f"Creating / verifying repository: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    tok_file = Path(tokenizer_path)
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_file}")

    print(f"Uploading checkpoint {ckpt_file.name} ({ckpt_file.stat().st_size / 1024**2:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(ckpt_file),
        path_in_repo="pytorch_model.pt",
        repo_id=repo_id,
        repo_type="model",
        token=token
    )

    print(f"Uploading tokenizer {tok_file.name}...")
    api.upload_file(
        path_or_fileobj=str(tok_file),
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="model",
        token=token
    )

    # Upload README model card
    readme_content = f"""---
language:
- en
tags:
- custom-llm
- pytorch
- causal-lm
pipeline_tag: text-generation
---

# {repo_id.split('/')[-1]}

Custom decoder-only Transformer model trained using PyTorch.

## Architecture
- **Embedding Dim**: 256
- **Heads**: 4
- **Layers**: 4
- **Context Length**: 128
- **Vocab Size**: 8,000 (Byte-level BPE)

## Usage
Load weights using the custom `MyLLM` architecture from the repository.
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token
    )

    print("\n" + "=" * 60)
    print(f"SUCCESS! Your model is live at: https://huggingface.co/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="e.g. your_username/myllm")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--public", action="store_true", help="Make repo public (default: private)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write)")
    args = parser.parse_args()

    upload_model(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        private=not args.public,
        token=args.token
    )
