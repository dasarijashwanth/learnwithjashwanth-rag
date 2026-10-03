# Learn with Jashwanth RAG Chatbot

Ask questions, get answers grounded in the [Learn with Jashwanth](https://learnwithjashwanth.substack.com) newsletter.

## How it works

1. **Ingest** (`scripts/ingest.py`): pulls every post from the Substack RSS feed into `data/posts.json`.
2. **Index** (`scripts/build_index.py`): chunks posts, embeds them, stores vectors in ChromaDB.
3. **Answer** (`app/rag.py`): retrieves the most relevant chunks for a question and asks an LLM to answer using only those chunks, with citations.
4. **Chat UI** (`app/app.py`): Streamlit interface — ask anything, see answers with linked sources.

## Run locally

```bash
pip install -r requirements.txt
python scripts/ingest.py
python scripts/build_index.py
streamlit run app/app.py
```

## Week 1 plan

See [WEEK_PLAN.md](WEEK_PLAN.md).
