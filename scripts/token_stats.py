from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

from src import config


def iter_chunk_texts(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield payload["chunk_id"], payload["text"]


def compute_token_stats(chunk_path: Path, model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    token_counts = []
    chunk_lengths = {}

    for chunk_id, text in iter_chunk_texts(chunk_path):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        count = len(tokens)
        token_counts.append(count)
        chunk_lengths[chunk_id] = count

    if not token_counts:
        print("No chunks found.")
        return

    total_tokens = sum(token_counts)
    avg_tokens = statistics.mean(token_counts)
    median_tokens = statistics.median(token_counts)
    max_chunk = max(chunk_lengths, key=chunk_lengths.get)
    min_chunk = min(chunk_lengths, key=chunk_lengths.get)

    print(f"Chunks analysed: {len(token_counts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per chunk: {avg_tokens:.2f}")
    print(f"Median tokens per chunk: {median_tokens}")
    print(f"Max tokens: {chunk_lengths[max_chunk]} ({max_chunk})")
    print(f"Min tokens: {chunk_lengths[min_chunk]} ({min_chunk})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute token statistics for chunk files")
    parser.add_argument(
        "--file",
        type=Path,
        default=config.CHUNKS_PATH,
        help="Path to chunk JSONL file",
    )
    parser.add_argument(
        "--model",
        default=config.EMBEDDING_MODEL,
        help="Tokenizer model name",
    )
    args = parser.parse_args()

    compute_token_stats(args.file, args.model)


if __name__ == "__main__":
    main()
