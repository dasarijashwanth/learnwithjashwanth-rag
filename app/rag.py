"""Day 3: Retrieval pipeline + LLM answering with citations.

Usage:
    python app/rag.py "How do I clean messy Excel data with pandas?"
    python app/rag.py --backend extractive "your question"

Pipeline:
    1. Retriever pulls the top-k most relevant chunks from the ChromaDB
       index built on Day 2 (same all-MiniLM-L6-v2 embeddings).
    2. An LLM backend answers using ONLY those chunks, citing each claim
       as [1], [2], ... mapped to a Sources list with title + URL.
    3. If the chunks don't cover the question, the model says so instead
       of hallucinating.

Backends (--backend, default "auto"):
    openai     - OpenAI chat completions (needs OPENAI_API_KEY). Model via
                 RAG_OPENAI_MODEL (default gpt-4o-mini).
    hf         - Local HuggingFace seq2seq model via transformers pipeline
                 (RAG_HF_MODEL, default google/flan-t5-base). No API key.
    extractive - No LLM. Composes an answer from the top chunks verbatim.
                 Always works offline; useful for tests and fallback.
    auto       - openai if OPENAI_API_KEY is set, else hf, else extractive.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "learnwithjashwanth"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
# Day 5: cosine-distance gate. Chunks scoring worse (higher) than this are
# treated as "the newsletter does not cover this". Calibrated on the Day 5
# eval set: in-scope questions scored 0.31-0.72, out-of-scope/gibberish
# scored 0.81-0.94 on the 3-post corpus. Override with RAG_RELEVANCE_THRESHOLD.
RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.78"))
OPENAI_MODEL = os.environ.get("RAG_OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL = os.environ.get("RAG_HF_MODEL", "google/flan-t5-base")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """One retrieved chunk with its source metadata and similarity score."""
    text: str
    title: str
    url: str
    published: str
    score: float  # cosine distance from Chroma (lower = closer)


class Retriever:
    """Loads the Day 2 ChromaDB index once, then answers similarity queries."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        import chromadb

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Index not found at {CHROMA_DIR}. "
                "Run scripts/build_index.py first."
            )
        self._embedder = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K
                 ) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for the query."""
        query = query.strip()
        if not query:
            return []
        embedding = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(1, self._collection.count())),
        )
        chunks = []
        for doc, meta, dist in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["distances"][0]):
            chunks.append(RetrievedChunk(
                text=doc,
                title=meta.get("title", "Untitled"),
                url=meta.get("url", ""),
                published=meta.get("published", ""),
                score=float(dist),
            ))
        return chunks


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the "
    "'Learn with Jashwanth' newsletter using ONLY the numbered context "
    "entries below.\n"
    "Rules:\n"
    "- Answer in the newsletter's friendly, practical tone. Be concise.\n"
    "- Attach a citation like [1] or [2] to EVERY factual sentence, "
    "matching the numbered context entries. Never cite a number that was "
    "not given to you.\n"
    "- Never invent post titles, URLs, or details that are not in the "
    "context.\n"
    "- If the context covers only part of the question, answer that part "
    "and plainly say which part is not covered.\n"
    "- If the context does not cover the question at all, say so in one "
    "sentence, do not invent details, and suggest what the newsletter "
    "does cover instead."
)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format the grounded prompt with numbered context entries."""
    context = "\n\n".join(
        f"[{i + 1}] {c.title} ({c.url})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return (f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:")


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    """A generated answer plus its cited sources."""
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    backend: str = ""


class LLMBackend:
    """Interface every answering backend implements."""

    name = "base"

    def generate(self, question: str,
                 chunks: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    """OpenAI chat completions (gpt-4o-mini by default)."""

    name = "openai"

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "pip install -r requirements.txt") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI()
        self._model = model

    def generate(self, question: str,
                 chunks: list[RetrievedChunk]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": "Context:\n" + "\n\n".join(
                     f"[{i + 1}] {c.title}\n{c.text}"
                     for i, c in enumerate(chunks))
                 + f"\n\nQuestion: {question}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


class HFBackend(LLMBackend):
    """Local HuggingFace seq2seq model (default flan-t5-base). No API key.

    Uses the tokenizer + model generate() API directly instead of the
    text2text-generation pipeline alias, which newer transformers
    versions removed (found during the Day 5 quality pass).
    """

    name = "hf"

    def __init__(self, model_name: str = HF_MODEL) -> None:
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str,
                 chunks: list[RetrievedChunk]) -> str:
        import torch
        prompt = build_prompt(question, chunks)
        inputs = self._tokenizer(prompt, return_tensors="pt",
                                 truncation=True, max_length=2048)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=256, do_sample=False)
        return self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True).strip()


class ExtractiveBackend(LLMBackend):
    """No-LLM fallback: quotes the most relevant retrieved chunks verbatim.

    Quotes each chunk's lead (~120 words), which carries the key point,
    instead of rewriting. Always works offline; useful for tests and
    fallback. Day 5 eval compares LLM backends against it.

    (Day 5 note: a query-term sentence-selection variant was tried and
    reverted — on this corpus the lead consistently covered more of the
    expected answer keywords than overlap-ranked sentences.)
    """

    name = "extractive"
    SNIPPET_WORDS = 120

    def generate(self, question: str,
                 chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks[:3]):
            text = re.sub(r"\s+", " ", chunk.text).strip()
            snippet = " ".join(text.split()[:self.SNIPPET_WORDS])
            parts.append(f"[{i + 1}] {snippet}...")
        body = "\n\n".join(parts)
        return ("Based on the newsletter, here is what I found:\n\n"
                f"{body}\n\n"
                "This is a retrieval-only answer: it quotes the most "
                "relevant passages instead of rewriting them.")


def get_backend(name: str = "auto") -> LLMBackend:
    """Resolve a backend name to an instance, falling back gracefully."""
    name = name.lower()
    if name == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIBackend()
        try:
            return HFBackend()
        except Exception:
            return ExtractiveBackend()
    backends = {"openai": OpenAIBackend, "hf": HFBackend,
                "extractive": ExtractiveBackend}
    if name not in backends:
        raise ValueError(f"Unknown backend {name!r}. "
                         f"Choose from: auto, {', '.join(backends)}")
    return backends[name]()


# ---------------------------------------------------------------------------
# Main answering entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, top_k: int = DEFAULT_TOP_K,
                    backend: str = "auto",
                    max_distance: float = RELEVANCE_THRESHOLD) -> Answer:
    """Retrieve relevant chunks and generate a cited answer.

    Returns an Answer with .text (citations as [1], [2], ...) and .sources
    (the RetrievedChunk list the numbers refer to).

    Relevance gate (Day 5): if even the best chunk scores worse than
    max_distance, the newsletter is treated as not covering the question
    and a graceful abstention is returned instead of answering from
    irrelevant passages.
    """
    retriever = Retriever()
    if not question.strip():
        return Answer(text="Please ask a question.", backend=backend)
    chunks = retriever.retrieve(question, top_k=top_k)
    if not chunks:
        return Answer(text="The index is empty. Run scripts/ingest.py and "
                           "scripts/build_index.py first.",
                      backend=backend)
    if chunks[0].score > max_distance:
        topics = ", ".join(f"'{t}'" for t in
                           dict.fromkeys(c.title for c in chunks))
        return Answer(
            text=("The Learn with Jashwanth newsletter does not cover "
                  f"that topic. So far it covers: {topics}. "
                  "Try asking about one of those."),
            sources=[],
            backend="relevance-gate",
        )
    llm = get_backend(backend)
    text = llm.generate(question, chunks)
    return Answer(text=text, sources=chunks, backend=llm.name)


def format_answer(answer: Answer) -> str:
    """Render the answer with a numbered Sources list for the CLI."""
    lines = [answer.text, "", "Sources:"]
    for i, src in enumerate(answer.sources):
        ref = f"{src.url} — {src.title}" if src.url else src.title
        lines.append(f"[{i + 1}] {ref}")
    lines.append(f"\n(backend: {answer.backend})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer questions grounded in the newsletter.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "openai", "hf", "extractive"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    try:
        answer = answer_question(args.question, top_k=args.top_k,
                                 backend=args.backend)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_answer(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
