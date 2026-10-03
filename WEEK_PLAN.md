# Week 1 plan: RAG chatbot over Learn with Jashwanth

Repo: `dasarijashwanth/learnwithjashwanth-rag` (created by Jashwanth, 2026-10-03)
Deploy target: Streamlit Cloud (to confirm)

## Day-by-day

- **Day 1 — Sat Oct 3**: Scaffold repo. Substack RSS ingestion → `data/posts.json` with title, date, URL, full text.
- **Day 2 — Sun Oct 4**: Chunking strategy + embeddings (sentence-transformers) + ChromaDB index build.
- **Day 3 — Mon Oct 5**: Retrieval pipeline + LLM answering with citations (`app/rag.py`).
- **Day 4 — Tue Oct 6**: Streamlit chat UI (`app/app.py`) — ask, answer, sources, suggested questions.
- **Day 5 — Wed Oct 7**: Quality pass — eval on 10 sample questions, prompt tuning, edge cases.
- **Day 6 — Thu Oct 8**: Polish — README with demo GIF, error handling, caching.
- **Day 7 — Fri Oct 9**: Deploy to Streamlit Cloud. Announce on LinkedIn + Substack.

Each day's code is pushed to GitHub the same day.
