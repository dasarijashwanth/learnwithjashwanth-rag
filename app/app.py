"""Day 4: Streamlit chat UI for the Learn with Jashwanth RAG chatbot.

Run:
    streamlit run app/app.py

Features:
    - Chat-style Q&A over the newsletter (Day 3 retrieval + answering).
    - Cited Sources expander under every answer (title + link).
    - Suggested starter questions drawn from the actual ingested posts.
    - Sidebar controls: answering backend, top-k chunks, index stats.
"""

import streamlit as st

from rag import Retriever, answer_question

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
    except Exception as exc:  # missing index, no model cache, etc.
        st.error(f"Index unavailable: {exc}")
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
question = st.chat_input("Ask about the newsletter...")
pending = st.session_state.pop("pending_question", None)
question = pending or question

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the newsletter..."):
            try:
                answer = answer_question(question, top_k=top_k,
                                         backend=backend)
            except (FileNotFoundError, RuntimeError,
                    ValueError) as exc:
                st.error(f"Could not answer: {exc}")
                st.stop()
        st.markdown(answer.text)
        sources = answer.sources
        if sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources):
                    label = f"[{i + 1}] {src.title}"
                    if src.url:
                        st.markdown(f"- [{label}]({src.url})")
                    else:
                        st.markdown(f"- {label}")
        st.caption(f"backend: {answer.backend}")

    st.session_state.messages.append({
        "role": "assistant",
        "text": answer.text,
        "sources": [
            {"title": s.title, "url": s.url} for s in answer.sources
        ] if answer.sources else [],
    })
    # Re-render so the suggested-question buttons disappear after Q1.
    if pending:
        st.rerun()
