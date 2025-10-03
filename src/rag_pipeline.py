from __future__ import annotations

"""Core Retrieval-Augmented Generation (RAG) pipeline implementation."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from opencc import OpenCC
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification

from . import config


@dataclass
class DocumentChunk:
    """封裝 chunk 內容與中繼資料。"""

    chunk_id: str
    text: str
    metadata: dict
    text_hash: str


@dataclass
class RetrievalResult:
    """檢索結果與其相似分數。"""

    chunk: DocumentChunk
    score: float


@dataclass
class RAGResponse:
    """回傳給使用者的問答與引用段落資訊。"""

    question: str
    answer: str
    contexts: List[RetrievalResult]


class EmbeddingModel:
    """Sentence embedding model using Hugging Face transformers."""

    def __init__(self, model_name: str, device: torch.device | None = None, max_length: int = 512) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked = last_hidden_state * attention_mask.unsqueeze(-1)
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        return summed / counts.unsqueeze(-1)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        sentence_embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().astype("float32")


class CrossEncoderReranker:
    """Cross-encoder reranker powered by a Hugging Face sequence classification model."""

    def __init__(self, model_name: str, device: torch.device | None = None, max_length: int = 512) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """根據交叉編碼器分數挑選前 `top_k` 段落。"""
        if not candidates:
            return []

        texts = [result.chunk.text for result in candidates]
        encoded = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**encoded).logits.squeeze(-1).detach().cpu().numpy()

        scored = [
            RetrievalResult(chunk=result.chunk, score=float(score))
            for result, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class QdrantVectorStore:
    """Qdrant 向量資料庫的封裝。"""

    def __init__(self, collection_name: str, storage_path: Path | None = None) -> None:
        if storage_path is None:
            self.client = QdrantClient(path=":memory:")
        else:
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))
        self.collection_name = collection_name
        self.chunk_lookup: dict[str, DocumentChunk] = {}

    def _set_documents(self, documents: Sequence[DocumentChunk]) -> None:
        self.chunk_lookup = {doc.chunk_id: doc for doc in documents}

    def is_up_to_date(self, documents: Sequence[DocumentChunk]) -> bool:
        try:
            self.client.get_collection(self.collection_name)
        except UnexpectedResponse:
            return False

        expected = {doc.chunk_id: doc.text_hash for doc in documents}
        if not expected:
            return False

        count_response = self.client.count(self.collection_name, exact=True)
        if count_response.count != len(expected):
            return False

        actual: dict[str, str] = {}
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=256,
                offset=next_offset,
            )
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                text_hash = payload.get("text_hash")
                if chunk_id:
                    actual[chunk_id] = text_hash
            if next_offset is None:
                break

        return actual == expected

    def rebuild(self, embeddings: np.ndarray, documents: Sequence[DocumentChunk]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        dim = embeddings.shape[1]
        self._set_documents(documents)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

        payloads = []
        for doc in documents:
            payload = {
                "chunk_id": doc.chunk_id,
                "text": doc.text,
                "text_hash": doc.text_hash,
            }
            for key, value in doc.metadata.items():
                if value is not None:
                    payload[key] = value
            payloads.append(payload)

        self.client.upsert(
            collection_name=self.collection_name,
            points=qmodels.Batch(
                ids=list(range(len(documents))),
                vectors=embeddings.tolist(),
                payloads=payloads,
            ),
        )

    def ensure_collection(
        self,
        documents: Sequence[DocumentChunk],
        embedding_fn: Callable[[], np.ndarray],
        force_refresh: bool = False,
    ) -> None:
        if force_refresh or not self.is_up_to_date(documents):
            embeddings = embedding_fn()
            self.rebuild(embeddings, documents)
        else:
            self._set_documents(documents)

    def search(self, embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("Expected a single embedding with shape (1, dim)")
        query_vector = embedding[0].astype("float32").tolist()
        results: List[RetrievalResult] = []
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        for hit in hits:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id")
            if not chunk_id:
                continue
            chunk = self.chunk_lookup.get(chunk_id)
            if not chunk:
                text = payload.get("text", "")
                metadata = {
                    "title": payload.get("title"),
                    "chapter": payload.get("chapter"),
                    "article": payload.get("article"),
                    "order": payload.get("order"),
                    "revision_date": payload.get("revision_date"),
                }
                text_hash = payload.get("text_hash") or hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk = DocumentChunk(chunk_id=chunk_id, text=text, metadata=metadata, text_hash=text_hash)
                self.chunk_lookup[chunk_id] = chunk
            results.append(RetrievalResult(chunk=chunk, score=float(hit.score)))
        return results


class LocalQwenGenerator:
    """使用 Qwen 模型離線生成回應。"""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.converter = OpenCC("s2t")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch_dtype,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch_dtype,
            )
            self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def _format_history(history: Sequence[Tuple[str, str]] | None) -> str:
        """將歷史問答整理成提示文字。

        Args:
            history: 近期的問答紀錄，每筆為 `(問題, 回答)`。

        Returns:
            str: 依序列出過去數輪對話的多行說明，無資料時回傳預設文字。
        """
        if not history:
            return "（無先前對話）"
        lines: List[str] = []
        for idx, (past_question, past_answer) in enumerate(history, start=1):
            lines.append(f"第 {idx} 輪問題：{past_question}")
            lines.append(f"第 {idx} 輪回答：{past_answer}")
        return "\n".join(lines)

    def generate(self, context: str, question: str, history: Sequence[Tuple[str, str]] | None = None) -> str:
        system_prompt = (
            "你是一位專業且謹慎的法律助理，熟悉《性騷擾防治法》。"
            "請僅根據提供的參考內容回答，若資料不足需直接回覆「無法回覆該問題」。"
            "回答須使用繁體中文。"
            "保持客觀語氣，不得加入臆測或引用未提供的來源。"
        )
        history_block = self._format_history(history)
        user_prompt = (
            "歷史對話（僅供了解上下文，不可直接引用）：\n"
            "<history>{history_block}</history>\n\n"
            "請依據下列參考內容回答問題：\n"
            "<context>\n{context}\n</context>\n\n"
            "回答時請將重點條列並附上段落編號。若資訊不足，回覆「無法回覆該問題」。\n"
            "問題：{question}"
        ).format(history_block=history_block, context=context, question=question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return self.converter.convert(answer)

    def refine_question(self, question: str, history: Sequence[Tuple[str, str]] | None = None) -> str:
        """利用模型改寫問題，使其更具體且利於檢索。"""

        history_block = self._format_history(history)
        system_prompt = (
            "你是一位問題改寫助手，負責將使用者的原始提問改寫為更具體、" \
            "更可執行的檢索語句。改寫時需保留關鍵資訊與語意，避免推測未知細節。"
        )
        user_prompt = (
            "以下是近期對話背景與原始問題，請輸出單一句子的改寫版本：\n"
            "對話摘要：\n{history_block}\n\n"
            "原始問題：{question}\n"
            "改寫要求：\n"
            "1. 保留時間、人事物等關鍵資訊，但可以補足推論所需的必要條件。\n"
            "2. 使句子清楚描述要查詢的主題與限制條件。\n"
            "3. 僅輸出改寫後的問題，不需要任何解釋或多餘字詞。"
        ).format(history_block=history_block, question=question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][input_ids.shape[-1]:]
        refined = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        refined = self.converter.convert(refined)
        return refined if refined else question


def load_chunks(chunks_path: Path) -> List[DocumentChunk]:
    """讀取 chunk JSONL 並轉為 `DocumentChunk` 物件。"""
    chunks: List[DocumentChunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metadata = {
                "title": payload.get("title"),
                "chapter": payload.get("chapter"),
                "article": payload.get("article"),
                "order": payload.get("order"),
                "revision_date": payload.get("revision_date"),
            }
            text = payload["text"]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=payload["chunk_id"],
                    text=text,
                    metadata=metadata,
                    text_hash=text_hash,
                )
            )
    return chunks


class RAGPipeline:
    """整合檢索、重排序與生成的主流程。"""

    def __init__(self, embed_model_name: str | None = None, llm_model_name: str | None = None) -> None:
        self.embed_model = EmbeddingModel(embed_model_name or config.EMBEDDING_MODEL)
        self.chunks = load_chunks(config.CHUNKS_PATH)
        storage_path = config.QDRANT_STORAGE_PATH if config.QDRANT_STORAGE_PATH else None
        self.vector_store = QdrantVectorStore(config.QDRANT_COLLECTION, storage_path)

        def embed_documents() -> np.ndarray:
            return self.embed_model.embed([chunk.text for chunk in self.chunks])

        self.vector_store.ensure_collection(
            documents=self.chunks,
            embedding_fn=embed_documents,
            force_refresh=config.QDRANT_REFRESH,
        )
        self.generator = LocalQwenGenerator(llm_model_name or config.GENERATION_MODEL)
        self.reranker: Optional[CrossEncoderReranker] = None
        if config.ENABLE_RERANKER:
            try:
                self.reranker = CrossEncoderReranker(config.RERANKER_MODEL)
            except Exception as exc:
                print(f"[WARN] Reranker unavailable: {exc}")
                self.reranker = None

    @staticmethod
    def _build_context(results: Sequence[RetrievalResult]) -> str:
        """將檢索結果轉成供 LLM 使用的上下文文字。"""
        lines: List[str] = []
        for idx, result in enumerate(results, start=1):
            meta = result.chunk.metadata
            header_parts = [str(meta.get("article"))]
            if meta.get("chapter"):
                header_parts.append(str(meta["chapter"]))
            header = " / ".join(part for part in header_parts if part and part != "None")
            lines.append(f"[{idx}] {header}")
            lines.append(result.chunk.text)
        return "\n".join(lines)

    def ask(
        self,
        question: str,
        top_k: int = config.TOP_K,
        history: Sequence[Tuple[str, str]] | None = None,
    ) -> RAGResponse:
        """執行單次查詢並取得回答。"""
        candidate_k = max(top_k, config.RERANKER_POOL_SIZE) if self.reranker else top_k

        queries: List[str] = [question]
        rewritten = self._rewrite_query(question, history)
        if rewritten and rewritten != question:
            queries.insert(0, rewritten)

        combined: Dict[str, RetrievalResult] = {}
        for query_text in queries:
            query_embedding = self.embed_model.embed([query_text])
            retrieved = self.vector_store.search(query_embedding, candidate_k)
            for result in retrieved:
                chunk_id = result.chunk.chunk_id
                existing = combined.get(chunk_id)
                if not existing or result.score > existing.score:
                    combined[chunk_id] = result

        results = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        if not results:
            return RAGResponse(question=question, answer="無法回覆該問題", contexts=[])

        if self.reranker:
            results = self.reranker.rerank(question, results, top_k)
        else:
            results = results[:top_k]

        limited_results: List[RetrievalResult] = []
        character_budget = config.MAX_CONTEXT_CHARS
        running_total = 0
        for result in results:
            chunk_len = len(result.chunk.text)
            if running_total + chunk_len > character_budget and limited_results:
                break
            limited_results.append(result)
            running_total += chunk_len

        context_block = self._build_context(limited_results)
        answer = self.generator.generate(context_block, question, history=history)
        return RAGResponse(question=question, answer=answer, contexts=limited_results)

    @staticmethod
    def batch_query(pipeline: "RAGPipeline", questions: Iterable[str]) -> List[RAGResponse]:
        """批次執行多筆問題的查詢流程。"""
        return [pipeline.ask(question) for question in questions]

    def _rewrite_query(self, question: str, history: Sequence[Tuple[str, str]] | None) -> str:
        """根據近期歷史組合新的檢索 Query。

        Args:
            question: 使用者最新提問。
            history: 近期問答紀錄，每筆為 `(問題, 回答)`。

        Returns:
            str: 將歷史摘要與最新問題串成的查詢文字，若無歷史則回傳原問題。
        """
        if not history:
            return question

        window_size = max(config.CONVERSATION_WINDOW, 0)
        relevant_history = history[-window_size:] if window_size else history

        parts: List[str] = []
        for idx, (past_question, past_answer) in enumerate(relevant_history, start=1):
            parts.append(f"先前問題{idx}：{past_question}；先前回答{idx}：{past_answer}")
        parts.append(f"最新問題：{question}")
        return "\n".join(parts)

    def refine_question(self, question: str, history: Sequence[Tuple[str, str]] | None = None) -> str:
        """對外提供問題改寫功能。"""

        refined = self.generator.refine_question(question, history)
        return refined or question
