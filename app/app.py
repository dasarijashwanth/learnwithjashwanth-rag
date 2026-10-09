"""RAG chatbot over your own documents — with a built-in demo dataset.

Anyone can use this: upload PDFs, text files, or Markdown; paste a web
page URL; or paste raw text. The app chunks, embeds, and indexes the
content, then answers questions with per-source citations.

The "Learn with Jashwanth" newsletter ships as the demo collection so
first-time visitors see it working immediately.

Run:
    streamlit run app/app.py

Note on the hosted demo: uploads live in your browser session only.
Run it locally (or on your own Streamlit Cloud account) for persistence.
"""

import time
import traceback

import streamlit as st

from rag import (COLLECTION_NAME, RAGError, Retriever, USER_COLLECTION_NAME,
                 answer_question, clear_collection, index_documents)
from sources import extract_upload, fetch_url, make_text_document

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot — chat with your own documents",
    page_icon="🤖",
    layout="centered",
)

DEMO_KEY = "demo"
USER_KEY = "mine"

COLLECTIONS = {
    DEMO_KEY: {
        "label": "Demo: Learn with Jashwanth newsletter",
        "collection": COLLECTION_NAME,
        "collection_label": "The Learn with Jashwanth newsletter",
        "placeholder": "Ask about the newsletter...",
        "spinner": "Searching the newsletter...",
        "suggestions": [
            "How do I clean messy Excel data with pandas?",
            "Do I need machine learning for my first data job?",
            "Why did you start the Learn with Jashwanth newsletter?",
            "What should I learn first as a data analyst?",
        ],
    },
    USER_KEY: {
        "label": "My documents",
        "collection": USER_COLLECTION_NAME,
        "collection_label": "Your documents",
        "placeholder": "Ask about your documents...",
        "spinner": "Searching your documents...",
        "suggestions": [
            "What are the main topics in these documents?",
            "Summarize the key points.",
        ],
    },
}

BACKEND_HELP = {
    "auto": "OpenAI if OPENAI_API_KEY is set, else local HuggingFace, else extractive.",
    "openai": "OpenAI chat completions (needs OPENAI_API_KEY).",
    "hf": "Local HuggingFace model (google/flan-t5-base). No API key.",
    "extractive": "No LLM: quotes the most relevant passages verbatim.",
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading index...")
def get_retriever(collection_name: str) -> Retriever:
    return Retriever(collection_name=collection_name)


@st.cache_data(show_spinner=False, ttl=3600)
def get_answer(question: str, top_k: int, backend: str,
               collection_name: str, collection_label: str) -> dict:
    """Answer a question, cached for an hour.

    The retriever is intentionally NOT a cache key: the single shared
    resource from get_retriever() is reused, so repeat questions cost
    only one embedding + LLM call, and identical repeats cost nothing.
    Returns a plain dict so Streamlit can serialize the cache entry.
    """
    answer = answer_question(question, top_k=top_k, backend=backend,
                             retriever=get_retriever(collection_name),
                             collection_label=collection_label)
    return {
        "text": answer.text,
        "backend": answer.backend,
        "sources": [{"title": s.title, "url": s.url}
                    for s in answer.sources],
    }


def show_error(exc: BaseException) -> None:
    """Friendly error panel; raw traceback hidden in an expander."""
    st.error("Something went wrong while answering. "
             "Your chat history is safe.")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(exc)), language="text")
    if st.button("🔄 Clear cache and retry", key="retry-clear-cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


def _index_and_refresh(docs, verb: str) -> None:
    """Index extracted docs, then rebuild the retriever so counts refresh."""
    with st.spinner(f"Indexing {verb}..."):
        try:
            n_chunks = index_documents(docs)
        except (RAGError, ValueError, RuntimeError) as exc:
            st.error(f"Could not index: {exc}")
            return
    st.cache_resource.clear()  # retriever re-opens the updated collection
    st.cache_data.clear()
    st.success(f"Added {len(docs)} document(s), {n_chunks} chunks indexed.")
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    collection_key = st.radio(
        "Collection",
        options=[DEMO_KEY, USER_KEY],
        format_func=lambda k: COLLECTIONS[k]["label"],
    )
    cfg = COLLECTIONS[collection_key]
    backend = st.selectbox(
        "Answering backend",
        options=["auto", "openai", "hf", "extractive"],
        help=BACKEND_HELP["auto"],
    )
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10,
                      value=5)

    st.divider()
    st.header("📚 Index")
    try:
        retriever = get_retriever(cfg["collection"])
        st.metric("Indexed chunks",
                  retriever._collection.count())
    except RAGError as exc:
        st.error(f"Index unavailable: {exc}")
        st.stop()
    except Exception as exc:  # unexpected: show details, offer recovery
        st.error("Index unavailable due to an unexpected error.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)),
                    language="text")
        if st.button("🔄 Clear cache and retry"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    if collection_key == USER_KEY:
        st.divider()
        st.header("➕ Add your content")
        st.caption("PDF, TXT, Markdown, web pages, or pasted text.")
        uploads = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )
        if st.button("Add uploaded files",
                     disabled=not uploads):
            docs = []
            for f in uploads:
                try:
                    docs.append(
                        extract_upload(f.name, f.getvalue()))
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
            if docs:
                _index_and_refresh(docs, "files")

        url = st.text_input("Web page URL",
                            placeholder="https://example.com/article")
        if st.button("Add URL", disabled=not url.strip()):
            try:
                doc = fetch_url(url)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                _index_and_refresh([doc], "URL")

        with st.expander("Paste text"):
            pasted_title = st.text_input("Title", key="paste-title")
            pasted_text = st.text_area("Text", key="paste-text",
                                       height=150)
            if st.button("Add pasted text",
                         disabled=not pasted_text.strip()):
                try:
                    doc = make_text_document(pasted_title, pasted_text)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    _index_and_refresh([doc], "text")

        if st.button("🗑️ Clear my documents"):
            clear_collection(USER_COLLECTION_NAME)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.messages = []
            st.success("Your documents were cleared.")
            st.rerun()

    st.divider()
    st.caption("Answers are grounded in the selected collection. "
               "Every claim cites its source.")
    if collection_key == USER_KEY:
        st.caption("On the hosted demo, your uploads live in this "
                   "browser session only.")


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("🤖 RAG Chatbot")
st.caption("Chat with your own documents. Answers cite their sources. "
           "Start with the demo, or add your content in the sidebar.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("collection_key") != collection_key:
    # Switching collections starts a fresh conversation.
    st.session_state.collection_key = collection_key
    st.session_state.messages = []

# Suggested questions (only until the first question is asked)
if not st.session_state.messages:
    if collection_key == USER_KEY:
        try:
            empty = get_retriever(cfg["collection"])._collection.count() == 0
        except Exception:
            empty = True
        if empty:
            st.info("Your collection is empty. Add PDFs, text files, "
                    "a web page, or pasted text in the sidebar to begin.")
    st.subheader("Try one of these:")
    for q in cfg["suggestions"]:
        if st.button(q, use_container_width=True, key=f"suggest-{q}"):
            st.session_state.pending_question = q
            st.rerun()

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"]):
                    label = f"[{i + 1}] {src['title']}"
                    if src["url"]:
                        st.markdown(f"- [{label}]({src['url']})")
                    else:
                        st.markdown(f"- {label}")

# New question: typed or picked from suggestions
question = st.chat_input(cfg["placeholder"], max_chars=2000)
pending = st.session_state.pop("pending_question", None)
question = pending or question

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(cfg["spinner"]):
            started = time.perf_counter()
            try:
                result = get_answer(question, top_k=top_k,
                                    backend=backend,
                                    collection_name=cfg["collection"],
                                    collection_label=cfg["collection_label"])
            except (RAGError, ValueError) as exc:
                st.error(f"Could not answer: {exc}")
                st.stop()
            except Exception as exc:
                show_error(exc)
                st.stop()
            elapsed = time.perf_counter() - started
        st.markdown(result["text"])
        sources = result["sources"]
        if sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources):
                    label = f"[{i + 1}] {src['title']}"
                    if src["url"]:
                        st.markdown(f"- [{label}]({src['url']})")
                    else:
                        st.markdown(f"- {label}")
        cached_note = " ⚡ served from cache" if elapsed < 0.2 else ""
        st.caption(f"backend: {result['backend']} · "
                   f"{elapsed:.1f}s{cached_note}")

    st.session_state.messages.append({
        "role": "assistant",
        "text": result["text"],
        "sources": result["sources"],
    })
    # Re-render so the suggested-question buttons disappear after Q1.
    if pending:
        st.rerun()
