"""
Upload your raw or processed dataset (e.g., training_data.json ~1GB) to Hugging Face Datasets Hub.
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_dataset(repo_id: str, file_path: str, private: bool = True, token: str = None):
    api = HfApi(token=token)
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    print(f"Creating / verifying dataset repo: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    print(f"Uploading {p.name} ({p.stat().st_size / 1024**2:.1f} MB) to Hugging Face...")
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=p.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )

    print("\n" + "=" * 60)
    print(f"SUCCESS! Dataset is live at: https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="e.g. your_username/myllm-dataset")
    parser.add_argument("--file", type=str, default="data/training_data.json")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default: private)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write)")
    args = parser.parse_args()

    upload_dataset(
        repo_id=args.repo_id,
        file_path=args.file,
        private=not args.public,
        token=args.token
    )
