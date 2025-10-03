from __future__ import annotations

"""Utilities for preparing source documents for RAG ingestion."""

import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

BULLET_PATTERN = re.compile(r"^[一二三四五六七八九十]+、")
HEADING_CHAPTER_PATTERN = re.compile(r"^第\s*[一二三四五六七八九十百千零]+\s*章")
HEADING_ARTICLE_PATTERN = re.compile(r"^第\s*\d+\s*條")
VALID_LINE_ENDINGS = ("。", "！", "？", "；", ":", "︰")
CJK_SPACE_PATTERN = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


@dataclass
class Chunk:
    """Container for a preprocessed text chunk."""

    chunk_id: str
    title: str
    revision_date: Optional[str]
    chapter: Optional[str]
    article: str
    order: int
    text: str


def clean_text(text: str) -> str:
    """Fix spacing issues introduced during PDF extraction."""

    compacted = CJK_SPACE_PATTERN.sub("", text.strip())
    compacted = re.sub(r"\s{2,}", " ", compacted)
    return compacted


def run_pdftotext(pdf_path: Path) -> str:
    """Extract raw text from a PDF file using pdftotext."""

    completed = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def normalise_line(raw_line: str) -> str:
    """Normalise whitespace while keeping intra-sentence spacing."""

    line = raw_line.replace("\u3000", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def sanitise_heading(line: str) -> str:
    """Collapse internal whitespace for legal headings."""

    return re.sub(r"\s+", "", line)


def iter_useful_lines(raw_text: str) -> Iterable[str]:
    for raw_line in raw_text.splitlines():
        line = normalise_line(raw_line)
        if not line or line == "\f" or re.fullmatch(r"\d+", line):
            continue
        yield line


def build_paragraphs(lines: List[str]) -> List[str]:
    """Group raw lines into semantic paragraphs or bullet items."""

    paragraphs: List[str] = []
    current: List[str] = []
    current_is_bullet = False

    def flush() -> None:
        nonlocal current, current_is_bullet
        if current:
            paragraphs.append(clean_text("".join(current)))
            current = []
            current_is_bullet = False

    for line in lines:
        is_bullet = bool(BULLET_PATTERN.match(line))
        if is_bullet:
            flush()
            current = [line]
            current_is_bullet = True
            if line.endswith(VALID_LINE_ENDINGS):
                flush()
            continue

        if not current:
            current = [line]
            current_is_bullet = False
        else:
            if current_is_bullet:
                current.append(line)
            else:
                separator = " " if not current[-1].endswith(" ") else ""
                current.append(f"{separator}{line}")

        if line.endswith(VALID_LINE_ENDINGS):
            flush()

    flush()
    return [para for para in paragraphs if para]


def merge_colon_blocks(paragraphs: List[str]) -> List[str]:
    merged: List[str] = []
    idx = 0
    total = len(paragraphs)
    while idx < total:
        current = paragraphs[idx]
        if current.endswith("："):
            collected = [current]
            cursor = idx + 1
            while cursor < total and BULLET_PATTERN.match(paragraphs[cursor]):
                collected.append(paragraphs[cursor])
                cursor += 1
            merged.append("\n".join(collected))
            idx = cursor
        else:
            merged.append(current)
            idx += 1
    return merged


@dataclass
class ArticleEntry:
    chapter: Optional[str]
    article: str
    paragraphs: List[str]


@dataclass
class PreprocessResult:
    title: str
    revision_date: Optional[str]
    chunks: List[Chunk]


def preprocess(
    pdf_path: Path,
    markdown_path: Path,
    chunks_path: Path,
    *,
    merge_colon_blocks_enabled: bool = False,
) -> PreprocessResult:
    raw_text = run_pdftotext(pdf_path)
    lines = list(iter_useful_lines(raw_text))

    title = ""
    revision_date: Optional[str] = None
    chapter: Optional[str] = None
    article: Optional[str] = None
    article_lines: List[str] = []
    chunks: List[Chunk] = []
    chunk_counter = 0
    articles: List[ArticleEntry] = []

    def emit_article() -> None:
        nonlocal article_lines, article, chunk_counter
        if not article:
            return
        paragraphs = build_paragraphs(article_lines)
        if merge_colon_blocks_enabled:
            paragraphs = merge_colon_blocks(paragraphs)
        articles.append(ArticleEntry(chapter=chapter, article=article, paragraphs=paragraphs))
        for order, text in enumerate(paragraphs, start=1):
            chunk_counter += 1
            chunk_id = f"{article.replace('第', 'art').replace('條', '')}-{order:02d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    title=title or "",
                    chapter=chapter,
                    article=article,
                    order=order,
                    text=text,
                    revision_date=revision_date,
                )
            )
        article_lines = []

    for line in lines:
        if line.startswith("法規名稱："):
            title = line.split("：", 1)[1].strip()
            continue
        if line.startswith("修正日期："):
            revision_date = line.split("：", 1)[1].strip()
            continue

        if HEADING_CHAPTER_PATTERN.match(line):
            emit_article()
            chapter = sanitise_heading(line)
            continue

        if HEADING_ARTICLE_PATTERN.match(line):
            emit_article()
            article = sanitise_heading(line)
            continue

        article_lines.append(line)

    emit_article()

    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_lines: List[str] = [f"# {title}" if title else "# 條文彙整"]
    if revision_date:
        markdown_lines.append(f"* 修正日期：{revision_date}")
    markdown_lines.append("")

    current_chapter: Optional[str] = None
    for entry in articles:
        if entry.chapter and entry.chapter != current_chapter:
            markdown_lines.append(f"## {entry.chapter}")
            markdown_lines.append("")
            current_chapter = entry.chapter
        markdown_lines.append(f"### {entry.article}")
        for paragraph in entry.paragraphs:
            markdown_lines.append(f"- {paragraph}")
        markdown_lines.append("")

    markdown_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")

    chunk_dicts = [asdict(chunk) for chunk in chunks]
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunk_dicts) + "\n",
        encoding="utf-8",
    )

    return PreprocessResult(title=title, revision_date=revision_date, chunks=chunks)


def main() -> None:
    pdf_path = Path("data/data.pdf")
    base_dir = Path("data/processed")
    base_dir.mkdir(parents=True, exist_ok=True)

    # preprocess(
    #     pdf_path,
    #     base_dir / "sex_harassment_prevention_act.md",
    #     base_dir / "chunks.jsonl",
    #     merge_colon_blocks_enabled=False,
    # )

    preprocess(
        pdf_path,
        base_dir / "sex_harassment_prevention_act.md",
        base_dir / "chunks.jsonl",
        merge_colon_blocks_enabled=True,
    )


if __name__ == "__main__":
    main()
