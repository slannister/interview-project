# RAG 系統建置說明

本文件說明整個 RAG（Retrieval-Augmented Generation）系統的設計與建置流程，涵蓋資料處理、檢索、生成、記錄、部署等重點。所有程式碼位於 `src/` 目錄，資料與模型請依下述步驟準備。

---

## 1. 系統架構總覽
1. **資料前處理**：將 `data/data.pdf` 轉成對應章節、條文、條列段落的 Markdown 與 chunk JSONL。
2. **向量化與索引**：使用 `intfloat/multilingual-e5-small` 嵌入模型取得 384 維句向量，寫入 Qdrant 嵌入式資料庫（cosine distance）。
3. **問題改寫**：用 Qwen 模型先將使用者問題改寫成更具體的檢索語句，再搭配近期對話生成 query。
4. **檢索與重排**：改寫後的 query 與原問題分別向量檢索；如啟用 reranker，使用 `BAAI/bge-reranker-base` 交叉編碼器重排序。
5. **生成回答**：`Qwen/Qwen2.5-1.5B-Instruct` 僅依參考段落生成繁體中文條列式回覆，回答必須附上段落編號；不足時回覆「無法回覆該問題」。
6. **紀錄與分析**：問答過程寫入 `logs/rag_queries_YYYYMMDD_HHMMSS.json`；若環境有 Langfuse key，會同步雲端紀錄。

---

## 2. 前置作業
1. **安裝依賴**
   ```bash
   pip install -r requirements.txt
   ```
   系統額外需 `pdftotext`（Poppler 套件）；macOS 可透過 Homebrew (`brew install poppler`) 安裝。

2. **離線下載模型**（連網環境執行一次即可）
   ```bash
   python -m scripts.download_models --output models
   ```
   會將以下模型存至 `models/`：
   - `Qwen/Qwen2.5-1.5B-Instruct`（生成）
   - `intfloat/multilingual-e5-small`（Embedding）
   - `BAAI/bge-reranker-base`（Cross-encoder reranker）

3. **資料前處理**
   ```bash
   python -m src.data_processing
   ```
   產出：
   - `data/processed/sex_harassment_prevention_act.md`
   - `data/processed/chunks.jsonl`
   （預設啟用合併「：」後的條列項）

---

## 3. 主要模組說明
- `src/config.py`：集中管理路徑、模型、檢索與生成參數。
- `src/data_processing.py`：PDF 解析與 chunk 產生。
- `src/rag_pipeline.py`
  - `EmbeddingModel`：Hugging Face sentence embedding。
  - `QdrantVectorStore`：負責 Qdrant collection 的建置與檢索。
  - `CrossEncoderReranker`：可選的交叉編碼排序器。
  - `LocalQwenGenerator`：生成回答 & 問題改寫工具。
  - `RAGPipeline`：整合上述組件；流程包含「問題改寫 → 多 query 檢索 → 重排 → 上下文整理 → 生成回答」。
- `src/cli.py`：命令列介面，支援單題查詢與互動式對話。
- `src/logging_utils.py`：問答紀錄，支援本地 JSON 與 Langfuse。
- `src/token_stats.py`：計算 chunk token 數統計。
- `scripts/download_models.py`：一次下載全部需要的模型。

更多函式細節請參考 `docs/code_overview.md`。

---

## 4. 啟動與操作
1. **單次查詢**
   ```bash
   python -m src.cli "權勢性騷擾的定義是什麼？"
   ```
   CLI 會先顯示改寫後的問題，再輸出條列式回覆。

2. **互動模式**
   ```bash
   python -m src.cli
   ```
   持續輸入問題即可，輸入 `exit`/`quit` 或 `Ctrl+C` 離開。系統保留最近 `CONVERSATION_WINDOW`（預設 3）輪對話，作為下一題的 query 改寫依據。

3. **檢視紀錄**
   每次啟動 CLI 會產生一個新 JSON 檔於 `logs/`。若設定環境變數 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`（與必要的 `LANGFUSE_HOST`），會同步寫入 Langfuse。

---

## 5. 重點設計與最佳實務
- **問題改寫與多 Query 檢索**：`refine_question` 會先產生更具體的檢索語句，並與原問題一起檢索，以提升 recall。
- **Chunk Hash 檢查**：`QdrantVectorStore.ensure_collection` 透過 text hash 判定資料是否更新，必要時自動重建索引。
- **Prompt 約束**：system prompt 嚴格規範回答格式與引用，避免幻覺。
- **繁體中文輸出**：所有生成結果經 OpenCC 轉換，確保繁中一致性。
- **可擴充性**：如需新增其他資料來源或模型，只需調整 `data_processing`、`config`、`download_models` 即可。

---

## 6. 常見檢查清單
- 檢查 `data/processed/chunks.jsonl` 是否存在且內容可讀。
- 確認 `models/` 已下載三個模型（或更新 `config` 指向正確位置）。
- Qdrant 若改為持久化，可將 `QDRANT_STORAGE_PATH` 改成專用資料夾。
- 若 prompt 未帶入上下文，可在 `LocalQwenGenerator.generate` 暫時 `print(user_prompt)` 以檢查。
- `MAX_CONTEXT_CHARS` 可依需求調整，避免上下文過長導致生成出錯。

---

## 7. 延伸方向
- **多路徑檢索**：可加上 BM25 或關鍵詞匹配與向量檢索融合。
- **回答驗證**：導入 citation check 確保引用段落與回答一致。
- **API/前端**：在 CLI 之外再包一層 RESTful API 或 UI。
- **模型評估**：撰寫腳本定期測試 recall、回答正確率。

---

如需更深入的流程示意或函式說明，請參考 `docs/code_overview.md` 或對應模組的 docstrings。若環境有更新（例如改用不同模型或資料來源），記得同步調整本文件。祝面試順利。
