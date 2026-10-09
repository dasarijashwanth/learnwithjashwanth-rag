"""Day 4: Streamlit chat UI for the Learn with Jashwanth RAG chatbot.
Day 6 polish: typed error handling, answer caching, retriever reuse.

Run:
    streamlit run app/app.py

Features:
    - Chat-style Q&A over the newsletter (Day 3 retrieval + answering).
    - Cited Sources expander under every answer (title + link).
    - Suggested starter questions drawn from the actual ingested posts.
    - Sidebar controls: answering backend, top-k chunks, index stats.
    - Cached answers (repeat questions are instant) and a shared
      retriever (no model reload per question).
    - Friendly error panel with a clear-cache-and-retry action.
"""

import time
import traceback

import streamlit as st

from rag import RAGError, Retriever, answer_question

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Learn with Jashwanth — RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

SUGGESTED_QUESTIONS = [
    "How do I clean messy Excel data with pandas?",
    "Do I need machine learning for my first data job?",
    "Why did you start the Learn with Jashwanth newsletter?",
    "What should I learn first as a data analyst?",
]

BACKEND_HELP = {
    "auto": "OpenAI if OPENAI_API_KEY is set, else local HuggingFace, else extractive.",
    "openai": "OpenAI chat completions (needs OPENAI_API_KEY).",
    "hf": "Local HuggingFace model (google/flan-t5-base). No API key.",
    "extractive": "No LLM: quotes the most relevant passages verbatim.",
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading newsletter index...")
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_data(show_spinner=False, ttl=3600)
def get_answer(question: str, top_k: int,
               backend: str) -> dict:
    """Answer a question, cached for an hour.

    The retriever is intentionally NOT a cache key: the single shared
    resource from get_retriever() is reused, so repeat questions cost
    only one embedding + LLM call, and identical repeats cost nothing.
    Returns a plain dict so Streamlit can serialize the cache entry.
    """
    answer = answer_question(question, top_k=top_k, backend=backend,
                             retriever=get_retriever())
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
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
        retriever = get_retriever()
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

    st.divider()
    st.caption("Answers are grounded in the Learn with Jashwanth "
               "newsletter. Every claim cites its source.")


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("🤖 Learn with Jashwanth")
st.caption("Ask anything about the newsletter. Answers cite their sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested questions (only until the first question is asked)
if not st.session_state.messages:
    st.subheader("Try one of these:")
    for q in SUGGESTED_QUESTIONS:
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
question = st.chat_input("Ask about the newsletter...", max_chars=2000)
pending = st.session_state.pop("pending_question", None)
question = pending or question

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the newsletter..."):
            started = time.perf_counter()
            try:
                result = get_answer(question, top_k=top_k,
                                    backend=backend)
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
