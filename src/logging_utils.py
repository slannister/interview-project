from __future__ import annotations

"""Logging helpers for recording RAG interactions locally and optionally to Langfuse."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from zoneinfo import ZoneInfo

from . import config

try:
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover - optional dependency issues
    Langfuse = None  # type: ignore[misc, assignment]

_langfuse_client: Langfuse | None = None  # type: ignore[name-defined]
_langfuse_initialised = False
_LOG_TIMEZONE = ZoneInfo("Asia/Taipei")
_SESSION_START = datetime.now(_LOG_TIMEZONE)
_SESSION_ID = _SESSION_START.strftime("%Y%m%d_%H%M%S")
_SESSION_LOG_PATH = config.LOG_DIR / f"{config.LOG_FILE_PREFIX}_{_SESSION_ID}.json"
_records_cache: List[dict[str, object]] | None = None


def _get_langfuse_client() -> Langfuse | None:  # type: ignore[name-defined]
    global _langfuse_client, _langfuse_initialised
    if _langfuse_initialised:
        return _langfuse_client

    _langfuse_initialised = True
    if Langfuse is None:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not public_key or not secret_key:
        return None

    kwargs = {"public_key": public_key, "secret_key": secret_key}
    if host:
        kwargs["host"] = host
    try:
        _langfuse_client = Langfuse(**kwargs)
    except Exception:
        _langfuse_client = None
    return _langfuse_client


def _serialise_contexts(contexts: Iterable) -> list[dict[str, object]]:
    serialised = []
    for result in contexts:
        meta = result.chunk.metadata
        serialised.append(
            {
                "chunk_id": result.chunk.chunk_id,
                "score": result.score,
                "text": result.chunk.text,
                "article": meta.get("article"),
                "chapter": meta.get("chapter"),
            }
        )
    return serialised


def _legacy_jsonl_path(log_path: Path) -> Path | None:
    base = log_path.with_suffix("")
    candidate = base.with_suffix(".jsonl")
    return candidate if candidate.exists() else None


def _normalise_timestamp(value: object) -> str | object:
    if not isinstance(value, str):
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(_LOG_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _load_records(log_path: Path) -> List[dict[str, object]]:
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "timestamp" in entry:
                        entry["timestamp"] = _normalise_timestamp(entry.get("timestamp"))
                return data
        except json.JSONDecodeError:
            pass
        return []

    legacy_aggregate = config.LOG_DIR / f"{config.LOG_FILE_PREFIX}.json"
    if legacy_aggregate.exists():
        try:
            data = json.loads(legacy_aggregate.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "timestamp" in entry:
                        entry["timestamp"] = _normalise_timestamp(entry.get("timestamp"))
                return data
        except json.JSONDecodeError:
            pass

    legacy_path = _legacy_jsonl_path(log_path)
    if legacy_path:
        records: List[dict[str, object]] = []
        with legacy_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict) and "timestamp" in entry:
                        entry["timestamp"] = _normalise_timestamp(entry.get("timestamp"))
                    records.append(entry)
                except json.JSONDecodeError:
                    continue
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        return records

    return []


def _write_records(log_path: Path, records: List[dict[str, object]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log_interaction(question: str, answer: str, contexts: Iterable, success: bool) -> None:
    timestamp = datetime.now(_LOG_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "success": success,
        "contexts": _serialise_contexts(contexts),
    }

    global _records_cache
    log_path = _SESSION_LOG_PATH
    if _records_cache is None:
        _records_cache = _load_records(log_path)
    _records_cache.append(record)
    _write_records(log_path, _records_cache)

    client = _get_langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(
            id=str(uuid.uuid4()),
            name="rag-query",
            input=question,
            output=answer,
            metadata={
                "success": success,
                "context_count": len(record["contexts"]),
            },
        )
        if record["contexts"]:
            trace.span(
                name="retrieved_contexts",
                input=record["contexts"],
            )
        client.flush()
    except Exception:
        # Suppress network/logging errors so they do not impact the main flow.
        return
