from __future__ import annotations

"""Utility script to pre-download all required Hugging Face models for offline use."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

from src import config


@dataclass
class ModelSpec:
    repo_id: str
    local_dir_name: str


REQUIRED_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(repo_id="Qwen/Qwen2.5-1.5B-Instruct", local_dir_name="qwen"),
    ModelSpec(repo_id="intfloat/multilingual-e5-small", local_dir_name="embedding"),
    ModelSpec(repo_id="BAAI/bge-reranker-base", local_dir_name="reranker"),
)


def download_models(target_dir: Path, revision: str | None = None, skip_existing: bool = True) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for spec in REQUIRED_MODELS:
        destination = target_dir / spec.local_dir_name
        if skip_existing and destination.exists():
            print(f"[skip] {spec.repo_id} already exists at {destination}")
            continue

        print(f"[download] {spec.repo_id} -> {destination}")
        snapshot_download(
            repo_id=spec.repo_id,
            revision=revision,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models required for offline deployment")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Directory to store downloaded models (default: ./models)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision / tag for all models",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if target directory already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_models(target_dir=args.output, revision=args.revision, skip_existing=not args.overwrite)


if __name__ == "__main__":
    main()
