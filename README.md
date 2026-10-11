# RAG Chatbot — chat with your own documents

A retrieval-augmented generation chatbot anyone can point at their own material: upload PDFs, text files, or Markdown; paste a web page URL; or paste raw text. The app chunks, embeds, and indexes the content, then answers questions with cited sources, not hallucinations.

The [Learn with Jashwanth](https://learnwithjashwanth.substack.com) newsletter ships as the built-in demo collection so first-time visitors see it working immediately.

Built as Week 1 of a weekly portfolio project series: one production-grade data/AI project per week, built in daily slices.

**Live demo:** https://learnwithjashwanth-rag-ilswex5xz8ea5mewpgg2b9.streamlit.app/

**Launch announcements:** [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:share:7514826295290269696/) · [Substack note](https://learnwithjashwanth.substack.com/p/my-week-1-project-turned-into-a-real)


## Features

- **Bring your own content.** PDF / TXT / Markdown uploads, web page URLs (article text extraction), and pasted text, all indexed into your own collection from the sidebar.
- **Hybrid search.** Dense vector search fused with BM25 keyword search via Reciprocal Rank Fusion: semantic understanding plus exact-term matching (function names, error codes, jargon).
- **Cross-encoder reranking.** Top candidates are re-scored by a model that reads the query and chunk together, for meaningfully better precision than bi-encoder retrieval alone. Toggleable in the sidebar.
- **Conversational follow-ups.** Short follow-ups like "tell me more" are expanded with the previous question for retrieval, and LLM backends see the last exchange for context.
- **Streaming answers.** OpenAI-backend answers stream token by token instead of making you wait.
- **Retrieval transparency.** Every source shows its relevance scores (vector distance, rerank score) so you can see *why* it was cited.
- **Grounded answers with citations.** Every factual claim links back to the exact source it came from (title + URL).
- **Honest abstention.** A relevance gate (cosine distance threshold, calibrated on the eval set) makes the bot say "not covered" instead of inventing answers for out-of-scope questions.
- **Two answer backends.** Extractive (deterministic, no API key) and HuggingFace seq2seq, switchable from the UI sidebar.
- **Streamlit chat UI.** Collection picker (demo vs. your documents), suggested starter questions, cited Sources expander per answer, chunk-count slider, and index stats in the sidebar.
- **Eval harness.** 10-question eval set measuring retrieval hit rate, keyword coverage, citation presence, abstention behavior, and crash-freedom, with before/after tuning results.
- **Answer caching.** The Streamlit UI caches answers (per question + backend + top-k) and shares one loaded retriever across calls, so repeat questions are instant and no model is reloaded per question.
- **Friendly error handling.** Typed errors (`IndexNotFoundError`, `ModelLoadError`, `BackendError`) surface actionable messages; the UI keeps chat history intact and offers a clear-cache-and-retry action.

## How it works

```
Your docs / Substack RSS  →  ingest  →  chunk + embed  →  ChromaDB index  →  retrieve  →  answer (cited)
```

1. **Ingest** (`app/sources.py` for user uploads/URLs; `scripts/ingest.py` for the Substack RSS demo feed): extracts clean text into source documents (title, URL, full text).
2. **Index** (`app/rag.py::index_documents`, or `scripts/build_index.py` for the demo): splits documents into overlapping chunks, embeds them with sentence-transformers, and stores vectors in a persistent ChromaDB collection (one per collection).
3. **Answer** (`app/rag.py`): embeds the question, retrieves the top-k chunks, applies the relevance gate, and generates an answer constrained to the retrieved context, with per-source title/link citations.
4. **Chat UI** (`app/app.py`): Streamlit interface. Pick a collection, add your own content, ask a question, read the answer, expand Sources to verify every claim.
5. **Evaluate** (`eval/run_eval.py`): scores the system on `eval/eval_questions.json`. See [eval/README.md](eval/README.md) for methodology and results.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Embeddings | sentence-transformers (HuggingFace) |
| Vector store | ChromaDB (persistent local) |
| Answer backends | Extractive (built-in) / HuggingFace seq2seq |
| UI | Streamlit |
| Ingestion | feedparser (Substack RSS demo), pypdf, readability-lxml |

## Project structure

```
├── app/
│   ├── app.py          # Streamlit chat UI (collections, uploads, Q&A)
│   ├── rag.py          # retrieval + answer engine, relevance gate, backends
│   └── sources.py      # user content ingestion: PDF/TXT/MD, URLs, pasted text
├── scripts/
│   ├── ingest.py       # RSS → data/posts.json
│   ├── build_index.py  # chunk + embed → ChromaDB
│   └── make_demo_gif.py # renders the walkthrough GIF in docs/
├── docs/
│   └── demo.gif        # animated walkthrough of the app
├── eval/
│   ├── eval_questions.json  # 10-question eval set
│   ├── run_eval.py          # eval harness
│   ├── README.md            # methodology + before/after results
│   └── results_*.json       # recorded runs
├── data/
│   ├── posts.json      # ingested newsletter posts (regenerated by ingest.py)
│   └── chroma/         # persistent vector index (gitignored, rebuilt by build_index.py)
├── requirements.txt
└── WEEK_PLAN.md        # the 7-day build plan
```

## Run locally

```bash
pip install -r requirements.txt

# 1. Ingest the newsletter
python scripts/ingest.py

# 2. Build the vector index (downloads the embedding model on first run)
python scripts/build_index.py

# 3. Launch the chat UI
streamlit run app/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Pick a backend in the sidebar: Extractive works with zero API keys.

To run the eval:

```bash
python eval/run_eval.py
```

## Deploy to Streamlit Cloud

The repo is deploy-ready: no secrets are required, and the vector index
(`data/chroma/`, gitignored) rebuilds itself from the committed
`data/posts.json` on first launch, so a fresh checkout just works.

1. Push the repo to GitHub (`dasarijashwanth/learnwithjashwanth-rag`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app →
   pick the repo, branch `main`, main file path `app/app.py`.
3. (Optional, best answers) Add a secret: `OPENAI_API_KEY = "..."` in the
   app's Secrets panel. Without it the app falls back to the local
   HuggingFace backend, then to the extractive backend. Nothing breaks.
4. Deploy. First load builds the index (~a minute) and caches it.

> Demo URL goes here after the first deploy:
> **https://learnwithjashwanth-rag.streamlit.app**

## Eval results (Day 5 quality pass)

| Metric | Before tuning | After tuning |
|---|---|---|
| Ran without crashing | 10/10 | 10/10 |
| Retrieval hit rate | 6/6 | 6/6 |
| Mean keyword coverage | 0.89 | 0.96 |
| Correct abstentions | 0/3 | 3/3 |

The biggest fix: out-of-scope questions used to get confident answers quoting irrelevant passages. The relevance gate now catches them. Full methodology in [eval/README.md](eval/README.md).

## 7-day build plan

| Day | Slice |
|---|---|
| 1 (Oct 3) | Scaffold, week plan, RSS ingestion |
| 2 (Oct 4) | Chunking, embeddings, persistent ChromaDB index |
| 3 (Oct 5) | Retrieval + cited answer engine |
| 4 (Oct 6) | Streamlit chat UI |
| 5 (Oct 7) | Quality pass: 10-question eval + prompt tuning |
| 6 (Oct 8) | Polish, error handling, caching |
| 7 (Oct 9) | Deploy to Streamlit Cloud |

## Roadmap

- More newsletter posts indexed automatically on a schedule
- Hybrid retrieval (dense + BM25)
- Conversation memory for follow-up questions
- Hosted demo link (after Streamlit Cloud deploy)

## Author

**Jashwanth Dasari** — AI Engineer, MS Data Science @ Florida Atlantic University.
Newsletter: [Learn with Jashwanth](https://learnwithjashwanth.substack.com) · LinkedIn: [Jashwanth Dasari](https://www.linkedin.com/in/jashwanth-dasari/)
