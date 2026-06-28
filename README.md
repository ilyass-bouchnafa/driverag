<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-2.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

<h1 align="center">🔍 DriveRAG — Academic Document Intelligence</h1>

<p align="center">
  <strong>A production-grade Retrieval-Augmented Generation (RAG) system that turns a student's Google Drive into a personal, searchable, citation-backed knowledge base.</strong>
</p>

<p align="center">
  <em>Built as a 3-month engineering project (April – June 2026) — from initial prototype to evaluated, multi-modal, full-stack application.</em>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [RAG Pipeline — Deep Dive](#-rag-pipeline--deep-dive)
  - [1. Ingestion Layer](#1-ingestion-layer)
  - [2. Retrieval Layer](#2-retrieval-layer)
  - [3. Generation Layer](#3-generation-layer)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Evaluation & Metrics (RAGAS)](#-evaluation--metrics-ragas)
- [Engineering Decisions & Trade-offs](#-engineering-decisions--trade-offs)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Roadmap & Evolution](#-roadmap--evolution)
- [License](#-license)

---

## 🎯 Project Overview

**DriveRAG** is a full-stack RAG application designed for engineering students at ENSA. It allows a student to:

1. **Connect their Google Drive** containing lecture notes, slides, and course PDFs
2. **Ask questions in natural language** (French, English, or Arabic)
3. **Receive precise, citation-backed answers** grounded exclusively in their own documents — not hallucinated from general knowledge

The system is a personal alternative to Google's NotebookLM, built with full control over the retrieval pipeline, evaluation framework, and citation mechanism.

### Why This Project Exists

| Problem | DriveRAG Solution |
|---------|-------------------|
| Students have hundreds of unstructured course PDFs | Automatic ingestion & intelligent chunking with section awareness |
| Finding specific information across 50+ documents is slow | Hybrid search (semantic + keyword) with sub-second retrieval |
| Generic chatbots hallucinate facts and cite nonexistent sources | Forced citation system with [Cx] ID mapping — impossible to invent a source |
| Existing tools don't support multilingual academic content (FR/EN/AR) | Multilingual embeddings, tokenizer, and language-aware prompting |
| No way to evaluate if the RAG actually works | Full RAGAS evaluation pipeline with 20 ground-truth questions |

---

## ✨ Key Features

### 🔄 Google Drive Synchronization
- **Recursive folder traversal** — scans nested subfolders with full path reconstruction
- **Smart incremental sync** — compares `modifiedTime` timestamps, only re-indexes changed files
- **Background auto-sync** — daemon thread runs every 30 minutes, zero user intervention
- **Direct file upload** — upload documents from the UI, automatically pushed to Drive and indexed

### 🔍 Advanced Hybrid Retrieval
- **Dual-vector search** — dense (semantic) + sparse (BM25) vectors stored in Qdrant
- **Native server-side RRF fusion** — Reciprocal Rank Fusion computed by Qdrant, not Python
- **Multi-query expansion** — LLM generates 3 reformulations of each question
- **HyDE (Hypothetical Document Embeddings)** — generates a synthetic answer, uses it as a search query
- **Parallel execution** — all query variants searched simultaneously via `asyncio.gather`
- **Cross-encoder reranking** — multilingual reranker with relative threshold filtering

### 📝 Anti-Hallucination Citation System
- **Forced [Cx] IDs** — each chunk given a short label ([C1], [C2], ...) in the prompt
- **Post-generation validation** — hallucinated IDs silently removed from the response
- **Human-readable remapping** — [C1] → [FileName.pdf, Page 3] after LLM generation
- **Only cited sources shown** — UI displays only the sources the LLM actually referenced

### 🎙️ Voice Input (Whisper)
- **Audio transcription** with OpenAI Whisper (local, no API calls)
- **Automatic language detection** with FR/EN bias for ambiguous short clips
- **Singleton model loading** — loaded once, reused across requests

### 💬 Dual Conversation Modes
- **RAG Mode** — answers strictly from documents with mandatory citations
- **Direct Mode** — general knowledge answers (no documents, no citations)
- **Asymmetric context bridge** — Direct mode can read RAG history for continuity, but RAG mode never sees Direct history (preserves traceability guarantee)

### 💾 Persistent Conversations
- **SQLite-backed history** — survives server restarts (not in-memory)
- **Thread-scoped isolation** — multiple conversations never leak into each other
- **Conversation sidebar** — browse and resume past conversations

### 📊 Automated Evaluation
- **RAGAS framework** with 4 metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **20 ground-truth Q&A pairs** covering databases, AI, image processing, file systems
- **Groq-powered evaluator** using llama-3.3-70b-versatile as judge LLM

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React 18)                        │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ Sidebar  │  │ ChatWindow │  │ InputBar  │  │   SyncBadge      │ │
│  │ (threads │  │ (messages, │  │ (text,    │  │   (countdown,    │ │
│  │  files)  │  │  sources)  │  │  voice,   │  │    file count)   │ │
│  │          │  │            │  │  upload)  │  │                  │ │
│  └──────────┘  └────────────┘  └───────────┘  └──────────────────┘ │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTP (REST API)
┌────────────────────────────▼─────────────────────────────────────────┐
│                      BACKEND (FastAPI + Uvicorn)                     │
│                                                                      │
│  /chat ─────────┐    /sync ──────┐    /transcribe ──┐    /upload     │
│                 │               │                  │               │
│  ┌──────────────▼───────────────▼──────────────────▼──────────────┐ │
│  │                    RAG PIPELINE (async)                         │ │
│  │                                                                │ │
│  │  ┌─────────────────────────────────────────────────────────┐   │ │
│  │  │  1. QUERY PROCESSING (query_processor.py)               │   │ │
│  │  │     ├─ Multi-query generation (3 reformulations)        │   │ │
│  │  │     ├─ HyDE generation (synthetic document)     ┐       │   │ │
│  │  │     └─ All generated in parallel (asyncio)      │       │   │ │
│  │  └─────────────────────────────────────────────────┘       │   │ │
│  │                          │                                 │   │ │
│  │  ┌───────────────────────▼─────────────────────────────┐   │   │ │
│  │  │  2. HYBRID SEARCH × N variants (qdrant_store.py)    │   │   │ │
│  │  │     ├─ Dense: paraphrase-multilingual-MiniLM-L12    │   │   │ │
│  │  │     ├─ Sparse: BM25 (custom encoder, Qdrant native) │   │   │ │
│  │  │     ├─ Fusion: RRF (server-side, k=60)              │   │   │ │
│  │  │     └─ All variants searched in parallel             │   │   │ │
│  │  └─────────────────────────────────────────────────────┘   │   │ │
│  │                          │                                 │   │ │
│  │  ┌───────────────────────▼─────────────────────────────┐   │   │ │
│  │  │  3. RERANKING (reranker.py)                         │   │   │ │
│  │  │     ├─ CrossEncoder: mmarco-mMiniLMv2-L12-H384      │   │   │ │
│  │  │     ├─ Relative threshold (best_score - 2.5)        │   │   │ │
│  │  │     └─ Floor score safety net (-8.0)                │   │   │ │
│  │  └─────────────────────────────────────────────────────┘   │   │ │
│  │                          │                                 │   │ │
│  │  ┌───────────────────────▼─────────────────────────────┐   │   │ │
│  │  │  4. GENERATION (llm_chain.py)                       │   │   │ │
│  │  │     ├─ Context with forced [Cx] IDs                 │   │   │ │
│  │  │     ├─ LLM: Groq (llama-3.1-8b-instant)            │   │   │ │
│  │  │     ├─ Citation resolution [Cx] → [File, Page]      │   │   │ │
│  │  │     └─ History: SQLite, scoped by thread_id         │   │   │ │
│  │  └─────────────────────────────────────────────────────┘   │   │ │
│  └────────────────────────────────────────────────────────────┘   │ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  INGESTION PIPELINE (sync_manager.py)                          │ │
│  │    Google Drive API → file_router.py → chunker.py → Qdrant    │ │
│  │    (PDF, DOCX, TXT, MD, PPTX, Google Docs)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │   Qdrant    │   │   SQLite    │   │   Google    │
   │  (vectors + │   │ (history,   │   │   Drive     │
   │   payloads) │   │  threads)   │   │   (files)   │
   └─────────────┘   └─────────────┘   └─────────────┘
```

---

## 🔬 RAG Pipeline — Deep Dive

### 1. Ingestion Layer

The ingestion pipeline transforms raw documents from Google Drive into indexed, searchable chunks in Qdrant.

#### Document Loading (`gdrive_loader.py`)
- **OAuth2 authentication** with token persistence (pickle) and automatic refresh
- **Recursive folder traversal** — rebuilds the full path for each file (e.g., `Courses/S5/TP3.pdf`)
- **MIME type filtering** — only processes supported formats: PDF, DOCX, TXT, Markdown, PPTX, Google Docs
- **Google Docs export** — native Google Docs are exported to DOCX on-the-fly via the Drive API

#### Text Extraction (`file_router.py`)
- **Format-specific loaders** via LangChain: `PyPDFLoader`, `Docx2txtLoader`, `TextLoader`, `UnstructuredPowerPointLoader`
- **Page-level granularity** for PDF (preserves real page numbers) and slide-level for PPTX
- **Temporary file strategy** — writes bytes to a temp file (LangChain loaders require file paths), cleans up after extraction

#### Structure-Aware Chunking (`chunker.py`)

> **Key engineering decision**: Standard `RecursiveCharacterTextSplitter` cuts by character count, blind to document structure. A chunk can start mid-definition or mid-theorem — making it semantically meaningless for retrieval.

**Solution**: Section-first chunking with size-based fallback:

1. **Section detection** — regex-based heading detection for:
   - Markdown headings (`#`, `##`, `###`)
   - Numbered academic headings (`1.2.3 Introduction`, `Chapitre 3 :`, `III. Concepts`)
2. **Section title preservation** — each chunk carries its `section_title` in metadata
3. **Size-based splitting only within sections** — `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) applied only if a section exceeds `CHUNK_SIZE`
4. **Title reattachment** — if the splitter isolates a heading as a micro-chunk, it's merged with the next chunk

#### Chunk Identity (`chunk_identity.py`)

> **Bug fixed**: The original `chunk_id = f"{source}_p{page}_c{chunk_index}"` used the **file name** as key. Two files named `TD1.pdf` in different Drive folders generated identical IDs → deletion/update of one corrupted the other.

**Fix**: `chunk_uid = f"{drive_id}_p{page}_c{chunk_index}"` — uses Google Drive's stable, unique file ID. Deterministic UUID5 derived from `chunk_uid` for Qdrant point IDs → re-sync replaces instead of duplicating.

#### Sync Manager (`sync_manager.py`)
- **Incremental sync** — compares `drive_modified_time` per `drive_id`, skips unchanged files
- **Force-all mode** — re-downloads and re-indexes everything (used for migration)
- **BM25 stats refresh** — recalculates global vocabulary, document frequency, and average document length **once per sync** (not per query)
- **Stats persistence** — pickles BM25 stats to disk; loaded at startup to avoid cold-start delay

---

### 2. Retrieval Layer

The retrieval pipeline transforms a user question into the top-5 most relevant document chunks.

#### Embedding (`embedder.py`)
- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- **Multilingual** — native support for French, English, Arabic, and 50+ languages
- **Singleton pattern** — model loaded once (~90MB), shared across all requests
- **Batch encoding** — `batch_size=32` for efficient bulk indexing

#### Sparse Encoding (`sparse_encoder.py`)

> **Why custom BM25 instead of a library?** Full control over multilingual tokenization (FR/EN/AR without heavy NLP dependencies), and the vocabulary is persisted with stable integer indices (critical: if indices change between syncs, all stored sparse vectors become inconsistent).

- **BM25 formula**: Robertson/Sparck Jones with `k1=1.5`, `b=0.75`
- **Tokenizer**: Unicode-aware regex (`[^\W\d_]+|\d+`) — handles accented characters, Arabic script, etc. No stemming (language-agnostic)
- **Vocabulary class**: Append-only (indices never change), serializable to JSON
- **Document encoding**: Full BM25 weight per term (IDF × normalized TF)
- **Query encoding**: IDF-only weighting (no document length normalization for queries)

#### Qdrant Vector Store (`qdrant_store.py`)

> **Migration from ChromaDB**: The original architecture used ChromaDB for dense search + `rank_bm25` rebuilt in Python at **every query** (loading the entire corpus, retokenizing everything). This became O(corpus) per request. Qdrant's native sparse vector support enables server-side BM25 scoring with an inverted index → O(log n).

- **Dual vector space**: Named vectors `dense` (cosine) + `bm25` (sparse, dot product)
- **Native RRF fusion**: `FusionQuery(fusion=Fusion.RRF)` — fuses sparse and dense results server-side
- **Prefetch strategy**: Retrieves `k×2` candidates from each vector space before fusion
- **Stable point IDs**: UUID5 derived from `chunk_uid` → idempotent upsert on re-sync

#### Query Processing (`query_processor.py`)

The query processor orchestrates advanced retrieval with full parallelism:

```
User Question
     │
     ├──────────────────┐ (parallel, asyncio.gather)
     ▼                  ▼
Multi-Query LLM     HyDE LLM
(3 reformulations)  (synthetic document)
     │                  │
     ▼                  ▼
[Q_orig, Q1, Q2, Q3, HyDE_doc]   ← 5 query variants
     │
     ├── hybrid_search(Q_orig)  ─┐
     ├── hybrid_search(Q1)      │
     ├── hybrid_search(Q2)      ├── all in parallel (asyncio.gather)
     ├── hybrid_search(Q3)      │
     └── hybrid_search(HyDE)   ─┘
                │
                ▼
     RRF Level 2 (between variants)
                │
                ▼
         Top-K candidates (20)
```

- **Two levels of RRF**: Level 1 (sparse+dense within Qdrant per query), Level 2 (between query variants in Python)
- **Pre-calculated BM25 stats**: Passed as parameters, never recalculated at query time

#### Cross-Encoder Reranking (`reranker.py`)

> **Bug fixed**: The original absolute threshold (`rerank_score > -5.0`) was never calibrated. CrossEncoder score scales differ between models. A threshold of -5.0 can be "terrible" for one model and "average" for another.

**Fix**: **Relative threshold** — keeps chunks within `RELATIVE_MARGIN = 2.5` points of the best score:
```
threshold = best_score - 2.5
```
- **Floor score safety** (`-8.0`): If even the best chunk is below this, the corpus likely doesn't contain the answer
- **Model**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, lightweight)

#### Reciprocal Rank Fusion (`rrf.py`)

> **Why RRF instead of weighted sum (alpha)?** The alpha method (`alpha × dense + (1-alpha) × bm25`) mixes two incomparable score scales. RRF only looks at **rank position** — a document ranked 1st in dense AND 3rd in sparse naturally rises, with no manual tuning.

```python
score(doc) = Σ  1 / (k + rank_in_list_i)
```
- Standard `k=60` smoothing constant
- Used in production by Elasticsearch, Qdrant, and most hybrid search engines

---

### 3. Generation Layer

#### LLM Chain (`llm_chain.py`)
- **LLM**: Groq API with `llama-3.1-8b-instant` (low latency, free tier)
- **Structured system prompt**: Strict rules for citation format, language handling, and refusal behavior
- **7-step pipeline**: Retrieve → Rerank → Build context → Load history → LLM call → Resolve citations → Persist

#### Citation Guard (`citation_guard.py`)

> **Core anti-hallucination mechanism**: Instead of letting the LLM write `[FileName.pdf, Page 12]` and hoping it's correct, we give it **closed, enumerated labels** `[C1], [C2], ...` and verify after generation.

1. **Before LLM**: Build context with forced IDs:
   ```
   [C1] (source: TP3.pdf, page 5 — Définitions)
   <chunk text>
   
   [C2] (source: Cours_IA.pdf, page 12 — SVM)
   <chunk text>
   ```
2. **After LLM**: Regex extracts all `[Cx]` from the response:
   - Valid ID → replaced with `[TP3.pdf, Page 5]`
   - Invalid ID (hallucinated) → silently removed
3. **Source list**: Only citations **actually used** in the response appear in the UI

#### Conversation Store (`conversation_store.py`)

> **Bug fixed**: The original implementation used a single global `_history = []` list. All conversations (regardless of thread_id) shared the same history → messages from conversation A appeared in conversation B.

**Fix**: SQLite database scoped by `thread_id`:
- **Persistent** — survives server restarts (not in-memory)
- **Isolated** — different threads never leak
- **Window-limited** — only the last 5 turns sent to the LLM (cost/latency/attention control)
- **Full history available** — via `/history/{thread_id}` endpoint (no truncation)

#### Direct LLM Mode (`llm_direct.py`)

> **Asymmetric context by design**: Direct mode reads RAG history (for continuity: "explain that again"), but RAG mode **never** reads Direct history. Why? RAG's value proposition is "answers ONLY from documents." Injecting unverified Direct-mode statements would break traceability.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, Lucide React, react-markdown | Modern chat UI with markdown rendering |
| **Backend** | FastAPI, Uvicorn, Pydantic | Async REST API with automatic validation |
| **LLM** | Groq API (llama-3.1-8b-instant) | Fast, free-tier inference |
| **Embeddings** | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) | Multilingual dense vectors (384d) |
| **Reranker** | CrossEncoder (mmarco-mMiniLMv2-L12-H384-v1) | Multilingual reranking |
| **Vector DB** | Qdrant (Docker) | Hybrid search (dense + sparse), native RRF |
| **Sparse Search** | Custom BM25 encoder | Language-agnostic sparse vectors for Qdrant |
| **Transcription** | OpenAI Whisper (local) | Voice-to-text with language detection |
| **Document Parsing** | LangChain loaders (PyPDF, Docx2txt, Unstructured) | Multi-format text extraction |
| **Storage** | Google Drive API v3 | Document source, OAuth2 authentication |
| **Persistence** | SQLite | Conversation history, thread management |
| **Cache** | Redis (optional) | BM25 corpus caching layer |
| **Evaluation** | RAGAS, Groq (llama-3.3-70b-versatile) | Automated RAG quality metrics |
| **Observability** | LangSmith | Trace & debug LLM chains |
| **DevOps** | Docker, docker-compose | One-command deployment |

---

## 📁 Project Structure

```
driverag/
├── backend/
│   └── main.py                    # FastAPI app — REST API (chat, sync, upload, transcribe, threads)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Main React app — sidebar, chat, topbar layout
│   │   ├── api.js                 # API client — all fetch calls to backend
│   │   ├── index.css              # Global styles — dark theme, design tokens
│   │   └── components/
│   │       ├── ChatWindow.jsx     # Message list, welcome screen, suggestions
│   │       ├── InputBar.jsx       # Text input, voice recording, mode switch, file upload
│   │       ├── MessageBubble.jsx  # Individual message — markdown, sources, copy
│   │       └── SyncBadge.jsx      # Sync status indicator with countdown
│   └── package.json
│
├── src/
│   ├── config.py                  # Centralized settings — models, API keys, hyperparameters
│   │
│   ├── ingestion/                 # ═══ DOCUMENT INGESTION ═══
│   │   ├── gdrive_loader.py       # Google Drive OAuth2, file listing, download
│   │   ├── file_router.py         # Multi-format text extraction (PDF, DOCX, TXT, PPTX, MD)
│   │   ├── chunker.py             # Structure-aware chunking (sections → size fallback)
│   │   ├── chunk_identity.py      # Stable chunk UID (drive_id-based, collision-free)
│   │   ├── drive_uploader.py      # Upload files TO Google Drive + immediate indexing
│   │   └── sync_manager.py        # Smart sync, auto-sync daemon, BM25 stats lifecycle
│   │
│   ├── retrieval/                 # ═══ SEARCH & RETRIEVAL ═══
│   │   ├── qdrant_store.py        # Qdrant client — indexing, hybrid search, native RRF
│   │   ├── embedder.py            # Dense embedding (multilingual sentence-transformers)
│   │   ├── sparse_encoder.py      # Custom BM25 → sparse vector (Qdrant-compatible)
│   │   ├── query_processor.py     # Multi-query + HyDE + parallel hybrid search
│   │   ├── reranker.py            # CrossEncoder reranking with relative threshold
│   │   ├── rrf.py                 # Reciprocal Rank Fusion (level 2, between variants)
│   │   ├── hybrid_search.py       # Legacy hybrid search (ChromaDB era, kept for reference)
│   │   ├── vectorstore.py         # Legacy ChromaDB store (pre-migration)
│   │   └── redis_corpus.py        # Optional Redis cache for BM25 corpus
│   │
│   ├── generation/                # ═══ LLM & RESPONSE ═══
│   │   ├── llm_chain.py           # Full RAG chain — retrieve, rerank, generate, cite
│   │   ├── llm_direct.py          # Direct LLM mode (no RAG, general knowledge)
│   │   ├── citation_guard.py      # [Cx] forced citation + post-generation validation
│   │   └── conversation_store.py  # SQLite conversation persistence, thread isolation
│   │
│   └── transcription/            # ═══ VOICE INPUT ═══
│       └── whisper_service.py     # Whisper transcription, language detection with FR/EN bias
│
├── evaluation/                    # ═══ QUALITY METRICS ═══
│   ├── ragas_eval.py              # RAGAS evaluation pipeline (4 metrics, 20 questions)
│   ├── test_questions.py          # Ground-truth Q&A dataset (French academic content)
│   └── last_results.json          # Latest evaluation scores
│
├── scripts/
│   └── migrate_chroma_to_qdrant.py  # One-time migration from ChromaDB to Qdrant
│
├── tests/
│   ├── test_file_router.py        # Unit tests for document extraction
│   └── test_query_processor.py    # Unit tests for query processing
│
├── app.py                         # Streamlit UI (legacy, kept for quick prototyping)
├── Dockerfile                     # Production container image
├── docker-compose.yml             # Full stack: Qdrant + DriveRAG
├── docker-compose.dev.yml         # Development overrides
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── LICENSE                        # MIT License
```

---

## 📊 Evaluation & Metrics (RAGAS)

The system is evaluated using the [RAGAS framework](https://docs.ragas.io/) with 20 ground-truth question-answer pairs covering the student's actual course material (databases/PL-SQL, artificial intelligence, image processing, file systems).

### Latest Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Faithfulness** | 0.875 | ≥ 0.80 | ✅ |
| **Answer Relevancy** | 0.915 | ≥ 0.80 | ✅ |
| **Context Precision** | 0.772 | ≥ 0.80 | ⚠️ |
| **Context Recall** | 0.750 | ≥ 0.80 | ⚠️ |

### What Each Metric Measures

| Metric | Measures | Why It Matters |
|--------|----------|----------------|
| **Faithfulness** | Does the answer contain only information from the retrieved context? | Hallucination detection — ensures the LLM doesn't invent facts |
| **Answer Relevancy** | Is the answer relevant to the question asked? | Ensures the LLM answers what was asked, not something tangential |
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | Measures retrieval quality — are we fetching the right documents? |
| **Context Recall** | Does the retrieved context cover all the information needed to answer? | Measures retrieval completeness — are we missing important chunks? |

### Evaluation Setup
- **Evaluator LLM**: `llama-3.3-70b-versatile` via Groq (stronger model for judging)
- **Embeddings**: Same multilingual model as production
- **Ground truths**: Hand-written by domain expert, in French
- **Execution**: Sequential with retry logic (rate-limit-aware)

---

## 🧠 Engineering Decisions & Trade-offs

### 1. Qdrant over ChromaDB
**Problem**: ChromaDB + Python BM25 required loading the entire corpus and retokenizing at every query → O(corpus) per request.  
**Decision**: Migrate to Qdrant with native sparse vector support → O(log n) with inverted index, server-side RRF fusion.  
**Trade-off**: Additional Docker container to manage. Acceptable for a local-first student tool.

### 2. Custom BM25 Encoder over fastembed
**Problem**: Needed full control over multilingual tokenization (FR/EN/AR) and stable vocabulary indices across syncs.  
**Decision**: Built a custom BM25 sparse encoder with append-only vocabulary.  
**Trade-off**: More code to maintain, but no external dependency mismatch and guaranteed index stability.

### 3. Relative Reranking Threshold over Absolute
**Problem**: Absolute threshold (`-5.0`) is model-specific and never calibrated → either rejects everything or keeps too much.  
**Decision**: `best_score - 2.5` adapts to each query's difficulty and the model's score distribution.  
**Trade-off**: Slightly more complex logic, but automatically correct across model changes.

### 4. Forced Citation IDs over Free-Form Citations
**Problem**: LLMs hallucinate file names and page numbers when asked to cite sources.  
**Decision**: Enumerate chunks as [C1], [C2], force the LLM to only use these labels, then post-process.  
**Trade-off**: Extra post-processing step, but **zero** false citations in production.

### 5. SQLite Conversations over In-Memory
**Problem**: Global `_history = []` leaked between conversations and was lost on restart.  
**Decision**: SQLite file scoped by `thread_id`, with 5-turn context window for LLM.  
**Trade-off**: Tiny disk I/O per message, but conversations persist across sessions — expected behavior for a study tool used over weeks.

### 6. Asymmetric Mode Bridging
**Problem**: Should Direct mode know about RAG history? Should RAG mode know about Direct history?  
**Decision**: Direct reads RAG (for continuity), RAG never reads Direct (preserves traceability guarantee).  
**Trade-off**: A student can't say "use what I told you in Direct mode" in RAG mode. This is intentional — RAG's value is strict document grounding.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for React frontend)
- Docker & Docker Compose (for Qdrant)
- Google Cloud Console project with Drive API enabled
- Groq API key (free at [console.groq.com](https://console.groq.com))
- FFmpeg (required for Whisper audio transcription)

### 1. Clone & Configure

```bash
git clone https://github.com/ilyass-bouchnafa/driverag.git
cd driverag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   GROQ_API_KEY=your_groq_api_key
#   GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
```

### 2. Google Drive Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable the **Google Drive API**
4. Create **OAuth 2.0 credentials** (Desktop application)
5. Download `credentials.json` → place in `credentials/credentials.json`
6. On first run, a browser window will open for authentication

### 3. Start Infrastructure

```bash
# Start Qdrant vector database
docker-compose up -d qdrant

# Verify Qdrant is running
curl http://localhost:6333/healthz
```

### 4. Start Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The server will:
- Start the auto-sync daemon (every 30 minutes)
- Pre-load BM25 stats from disk
- Expose the API at `http://localhost:8000`

### 5. Start Frontend

```bash
cd frontend
npm install
npm start
```

The React app opens at `http://localhost:3000` and proxies API calls to `http://localhost:8000`.

### 6. First Sync

Click the **Sync** button in the UI, or call:
```bash
curl -X POST http://localhost:8000/sync
```

This downloads all supported files from your Drive folder, chunks them, computes embeddings, and indexes them in Qdrant.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message (RAG or Direct mode) |
| `GET` | `/health` | Server status, sync info, indexed file count |
| `GET` | `/files` | List all indexed files |
| `POST` | `/sync` | Trigger manual Drive synchronization |
| `POST` | `/upload` | Upload a file (PDF, DOCX, TXT, MD, PPTX) |
| `POST` | `/clear` | Clear a specific conversation's history |
| `POST` | `/transcribe` | Transcribe audio to text (Whisper) |
| `GET` | `/threads` | List all past conversation threads |
| `GET` | `/history/{thread_id}` | Get full history of a conversation |

### Example: Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Qu'est-ce qu'un filtre médian en traitement d'image ?",
    "mode": "rag",
    "thread_id": "my-session-123"
  }'
```

### Example: Response

```json
{
  "answer": "Le filtre médian est un filtre non linéaire qui remplace la valeur d'un pixel par la médiane de ses voisins [Cours_TI.pdf, Page 15]. Il est particulièrement efficace pour supprimer le bruit de type 'Poivre et Sel' tout en préservant les contours [Cours_TI.pdf, Page 16].",
  "sources": [
    {
      "file": "Cours_TI.pdf",
      "path": "S5/Traitement d'Images/Cours_TI.pdf",
      "page": 15,
      "section_title": "Filtres non linéaires",
      "format": "pdf",
      "score": 2.847
    }
  ],
  "mode": "rag",
  "thread_id": "my-session-123"
}
```

---

## 🗺 Roadmap & Evolution

### Project Timeline

| Phase | Period | Milestone |
|-------|--------|-----------|
| **Phase 1** — Foundation | April 2026 | Initial RAG setup: Google Drive integration, PDF extraction, ChromaDB, basic Streamlit UI |
| **Phase 2** — Core Features | May 2026 | Direct LLM mode, smart sync with timestamps, dark UI, file upload, multi-format support |
| **Phase 3** — Architecture | May 2026 | FastAPI + React rewrite, Whisper transcription, Redis caching, RAGAS evaluation |
| **Phase 4** — Optimization | June 2026 | Qdrant migration, custom BM25 sparse encoder, RRF fusion, parallel retrieval, citation guard, reranker calibration, conversation persistence |

### Completed Optimizations
- [x] ChromaDB → Qdrant migration (O(corpus) → O(log n) per query)
- [x] Sequential → parallel query processing (5 searches in parallel)
- [x] Absolute → relative reranking threshold (auto-calibrating)
- [x] File-name → drive_id chunk identity (collision-free)
- [x] In-memory → SQLite conversation history (persistent, isolated)
- [x] Free-form → forced citation IDs (zero false citations)
- [x] Per-query → per-sync BM25 stats (amortized cost)

### Potential Future Work
- [ ] Streaming responses (Server-Sent Events)
- [ ] Multi-user support with user-scoped collections
- [ ] Table and figure extraction from PDFs
- [ ] Knowledge graph overlay for cross-document reasoning
- [ ] Fine-tuned embedding model on academic French corpus
- [ ] Progressive Web App (PWA) for mobile access

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ by <a href="https://github.com/ilyass-bouchnafa">Ilyass Bouchnafa</a></strong>
  <br/>
  <em>ENSA / IMT Mines Alès — 2026</em>
</p>
