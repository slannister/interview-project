from __future__ import annotations

import argparse
from textwrap import indent
from typing import List, Tuple

from . import config
from .rag_pipeline import RAGPipeline, RAGResponse
from .logging_utils import log_interaction


def format_sources(pipeline_response) -> str:
    lines = []
    for idx, result in enumerate(pipeline_response.contexts, start=1):
        meta = result.chunk.metadata
        header = f"{meta.get('article')}"
        if meta.get("chapter"):
            header += f" · {meta['chapter']}"
        lines.append(f"{idx}. {header} (score={result.score:.3f}, chunk={result.chunk.chunk_id})")
        lines.append(indent(result.chunk.text, prefix="   "))
    return "\n".join(lines)


def handle_question(pipeline: RAGPipeline, question: str, history: List[Tuple[str, str]]) -> None:
    window_size = max(config.CONVERSATION_WINDOW, 0)
    recent_history = history[-window_size:] if window_size else []
    refined_question = pipeline.refine_question(question, recent_history)
    if refined_question != question:
        # print("\n> 改寫後問題：", refined_question)
        pass
    response = pipeline.ask(refined_question, history=recent_history)
    response = RAGResponse(question=question, answer=response.answer, contexts=response.contexts)
    print("\n> 回答：", response.answer)
    # if response.contexts:
    #     print("\n=== 參考來源 ===")
    #     print(format_sources(response))
    # else:
    #     print("\n（找不到足夠的參考來源）")
    success = bool(response.contexts) and response.answer.strip() != "無法回覆該問題"
    log_interaction(question, response.answer, response.contexts, success)
    history.append((refined_question, response.answer))


def interactive_loop(pipeline: RAGPipeline) -> None:
    print("輸入問題以查詢，輸入 exit / quit 結束。")
    history: List[Tuple[str, str]] = []
    while True:
        try:
            question = input("\n> 使用者： ").strip()
        except (KeyboardInterrupt, EOFError):
            print()  # newline
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        handle_question(pipeline, question, history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG CLI for 性騷擾防治法")
    parser.add_argument("question", nargs="?", help="要查詢的問題")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    history: List[Tuple[str, str]] = []

    if args.question:
        handle_question(pipeline, args.question, history)
    else:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()
