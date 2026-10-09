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
COLLECTION_NAME = "learnwithjashwanth"   # demo: his Substack newsletter
USER_COLLECTION_NAME = "user_documents"  # user-provided content
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
# Day 5: cosine-distance gate. Chunks scoring worse (higher) than this are
# treated as "the newsletter does not cover this". Calibrated on the Day 5
# eval set: in-scope questions scored 0.31-0.72, out-of-scope/gibberish
# scored 0.81-0.94 on the 3-post corpus. Override with RAG_RELEVANCE_THRESHOLD.
RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.78"))
OPENAI_MODEL = os.environ.get("RAG_OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL = os.environ.get("RAG_HF_MODEL", "google/flan-t5-base")

# Day 6: cap absurdly long questions before they hit the embedder/LLM.
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 10


# ---------------------------------------------------------------------------
# Errors (Day 6: typed errors so callers can show friendly messages)
# ---------------------------------------------------------------------------

class RAGError(Exception):
    """Base class for all errors raised by the RAG pipeline."""


class IndexNotFoundError(RAGError):
    """The ChromaDB index has not been built yet."""


class ModelLoadError(RAGError):
    """An embedding or LLM model could not be loaded (often network)."""


class BackendError(RAGError):
    """An answering backend failed to initialize or generate."""


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
    rerank_score: float = 0.0  # cross-encoder score (higher = better)
    chunk_id: str = ""  # chroma id, used for hybrid fusion


class Retriever:
    """Loads a ChromaDB collection once, then answers similarity queries.

    Day 7: self-healing index. On a fresh deploy (e.g. Streamlit Cloud) the
    chroma directory is not committed, so if the demo collection is missing
    the index is built automatically from the committed data/posts.json on
    first use. User collections start empty and are created on first write.
    """

    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.collection_name = collection_name
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
        except ImportError as exc:
            raise ModelLoadError(
                "Required packages are missing. "
                "Run: pip install -r requirements.txt"
            ) from exc
        if (collection_name == COLLECTION_NAME
                and not CHROMA_DIR.exists()):
            self._build_index_from_posts()
        try:
            self._embedder = SentenceTransformer(EMBED_MODEL)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load embedding model {EMBED_MODEL!r}. "
                "Check your network connection and try again."
            ) from exc
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise IndexNotFoundError(
                f"Could not open the index at {CHROMA_DIR}. "
                "It may be corrupted; try scripts/build_index.py --recreate."
            ) from exc

    @staticmethod
    def _build_index_from_posts() -> None:
        """Build the vector index from data/posts.json when it is missing.

        This is what makes the repo deploy-ready: the chroma directory is
        gitignored, so a fresh checkout (Streamlit Cloud, a new machine)
        has no index. The posts file is committed, so the index can be
        rebuilt here on the fly instead of crashing with IndexNotFound.
        """
        posts_path = BASE_DIR / "data" / "posts.json"
        if not posts_path.exists():
            raise IndexNotFoundError(
                f"Index not found at {CHROMA_DIR} and no {posts_path} "
                "to build one from. Run scripts/ingest.py, then "
                "scripts/build_index.py."
            )
        try:
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            from scripts.build_index import build_index
            build_index()
        except (Exception, SystemExit) as exc:
            # build_index signals fatal errors via SystemExit.
            raise IndexNotFoundError(
                f"No index at {CHROMA_DIR} and the automatic build "
                f"failed: {exc}. Run scripts/build_index.py manually."
            ) from exc

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K
                 ) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for the query."""
        return [c for _, c in
                self._retrieve_with_ids(query, top_k=top_k)]

    def _retrieve_with_ids(
            self, query: str,
            top_k: int = DEFAULT_TOP_K) -> list[tuple[str, RetrievedChunk]]:
        """Dense retrieval returning (chroma_id, chunk) pairs."""
        query = query.strip()
        if not query:
            return []
        embedding = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(1, self._collection.count())),
        )
        out = []
        for cid, doc, meta, dist in zip(results["ids"][0],
                                        results["documents"][0],
                                        results["metadatas"][0],
                                        results["distances"][0]):
            out.append((cid, RetrievedChunk(
                text=doc,
                title=meta.get("title", "Untitled"),
                url=meta.get("url", ""),
                published=meta.get("published", ""),
                score=float(dist),
                chunk_id=cid,
            )))
        return out

    # -- Hybrid search (dense + BM25) ------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25_index(self):
        """Build (and cache) a BM25 index over the collection's chunks.

        Rebuilt automatically when the chunk count changes, so newly
        added documents are searchable immediately.
        """
        count = self._collection.count()
        if (getattr(self, "_bm25_built_for", -1) == count
                and hasattr(self, "_bm25")):
            return self._bm25, self._bm25_items
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ModelLoadError(
                "Hybrid search needs the 'rank-bm25' package. "
                "Run: pip install -r requirements.txt"
            ) from exc
        data = self._collection.get(include=["documents", "metadatas"])
        items, corpus = [], []
        for cid, doc, meta in zip(data["ids"], data["documents"],
                                  data["metadatas"]):
            items.append({"id": cid, "doc": doc, "meta": meta})
            # Title gets extra weight: repeat it so title terms count more.
            corpus.append(self._tokenize(
                f"{meta.get('title', '')} {meta.get('title', '')} {doc}"))
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_items = items
        self._bm25_built_for = count
        return self._bm25, self._bm25_items

    def hybrid_retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                        dense_k: int = 20, bm25_k: int = 20
                        ) -> list[RetrievedChunk]:
        """Fuse dense vector search and BM25 keyword search (RRF).

        Dense search catches meaning ("ways to clean data"), BM25 catches
        exact terms ("pandas read_excel"). Reciprocal Rank Fusion merges
        both ranked lists without needing comparable scores.
        """
        dense = self._retrieve_with_ids(query, top_k=dense_k)
        rrf: dict[str, float] = {}
        seen: dict[str, RetrievedChunk] = {}
        for rank, (cid, chunk) in enumerate(dense):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
            seen[cid] = chunk
        bm25, items = self._bm25_index()
        if bm25 is not None:
            scores = bm25.get_scores(self._tokenize(query))
            ranked = sorted(range(len(scores)),
                            key=lambda i: scores[i], reverse=True)[:bm25_k]
            for rank, i in enumerate(ranked):
                if scores[i] <= 0:
                    continue
                cid = items[i]["id"]
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
                if cid not in seen:
                    meta = items[i]["meta"]
                    seen[cid] = RetrievedChunk(
                        text=items[i]["doc"],
                        title=meta.get("title", "Untitled"),
                        url=meta.get("url", ""),
                        published=meta.get("published", ""),
                        score=1.0,  # BM25-only hit: no dense distance
                        chunk_id=cid,
                    )
        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
        return [seen[cid] for cid, _ in fused[:top_k]]


# ---------------------------------------------------------------------------
# Indexing user documents
# ---------------------------------------------------------------------------

def index_documents(docs: "list[SourceDocument]",
                    collection_name: str = USER_COLLECTION_NAME,
                    recreate: bool = False) -> int:
    """Chunk and index user-provided documents into a Chroma collection.

    Returns the number of chunks indexed. Raises RAGError subclasses on
    model/index failures and ValueError when there is nothing to index.
    """
    if not docs:
        raise ValueError("No documents to index.")
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as exc:
        raise ModelLoadError(
            "Required packages are missing. "
            "Run: pip install -r requirements.txt"
        ) from exc
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from scripts.build_index import CHUNK_OVERLAP, CHUNK_WORDS, chunk_text

    documents, metadatas = [], []
    for doc in docs:
        for i, chunk in enumerate(
                chunk_text(doc.text, CHUNK_WORDS, CHUNK_OVERLAP)):
            documents.append(f"{doc.title}\n\n{chunk}")
            metadatas.append({
                "title": doc.title,
                "url": doc.url,
                "published": "",
                "chunk": i,
            })
    if not documents:
        raise ValueError("The documents contained no indexable text.")
    try:
        model = SentenceTransformer(EMBED_MODEL)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load embedding model {EMBED_MODEL!r}. "
            "Check your network connection and try again."
        ) from exc
    embeddings = model.encode(documents, batch_size=32).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    start = collection.count()
    ids = [f"userdoc-{start + i}" for i in range(len(documents))]
    collection.upsert(ids=ids, documents=documents,
                      metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def clear_collection(collection_name: str) -> None:
    """Delete every chunk in a collection (used for "start over")."""
    try:
        import chromadb
    except ImportError as exc:
        raise ModelLoadError(
            "Required packages are missing. "
            "Run: pip install -r requirements.txt"
        ) from exc
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # already empty / never created


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

_reranker = None


def get_reranker():
    """Lazily load the cross-encoder reranker (cached for the process)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ModelLoadError(
                "Reranking needs the 'sentence-transformers' package. "
                "Run: pip install -r requirements.txt"
            ) from exc
        try:
            _reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            raise ModelLoadError(
                "Could not load the reranker model. "
                "Check your network connection and try again."
            ) from exc
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk],
           top_k: int) -> list[RetrievedChunk]:
    """Reorder candidates with a cross-encoder (query, chunk) scorer.

    Bi-encoder retrieval is fast but shallow; the cross-encoder reads the
    query and chunk together and is far better at judging true relevance.
    Runs over a wider candidate pool, then keeps the best top_k.
    """
    if len(chunks) <= top_k:
        return chunks
    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]),
                    reverse=True)
    out = []
    for chunk, s in ranked[:top_k]:
        chunk.rerank_score = float(s)
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "numbered context entries below.\n"
    "Rules:\n"
    "- Be friendly, practical, and concise.\n"
    "- Attach a citation like [1] or [2] to EVERY factual sentence, "
    "matching the numbered context entries. Never cite a number that was "
    "not given to you.\n"
    "- Never invent titles, URLs, or details that are not in the "
    "context.\n"
    "- If the context covers only part of the question, answer that part "
    "and plainly say which part is not covered.\n"
    "- If the context does not cover the question at all, say so in one "
    "sentence, do not invent details, and suggest what the collection "
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

    def generate(self, question: str, chunks: list[RetrievedChunk],
                 history_note: str = "") -> str:
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

    def generate(self, question: str, chunks: list[RetrievedChunk],
                 history_note: str = "") -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": (history_note + "Context:\n" + "\n\n".join(
                     f"[{i + 1}] {c.title}\n{c.text}"
                     for i, c in enumerate(chunks))
                 + f"\n\nQuestion: {question}")},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, question: str, chunks: list[RetrievedChunk],
                        history_note: str = ""):
        """Yield answer tokens as they arrive (OpenAI streaming)."""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": (history_note + "Context:\n" + "\n\n".join(
                     f"[{i + 1}] {c.title}\n{c.text}"
                     for i, c in enumerate(chunks))
                 + f"\n\nQuestion: {question}")},
            ],
            temperature=0.2,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta


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
                 chunks: list[RetrievedChunk],
                 history_note: str = "") -> str:
        import torch
        prompt = (history_note + build_prompt(question, chunks)
                  if history_note else build_prompt(question, chunks))
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
                 chunks: list[RetrievedChunk],
                 history_note: str = "") -> str:
        parts = []
        for i, chunk in enumerate(chunks[:3]):
            text = re.sub(r"\s+", " ", chunk.text).strip()
            snippet = " ".join(text.split()[:self.SNIPPET_WORDS])
            parts.append(f"[{i + 1}] {snippet}...")
        body = "\n\n".join(parts)
        return ("Based on the collection, here is what I found:\n\n"
                f"{body}\n\n"
                "This is a retrieval-only answer: it quotes the most "
                "relevant passages instead of rewriting them.")


def get_backend(name: str = "auto") -> LLMBackend:
    """Resolve a backend name to an instance, falling back gracefully.

    Raises BackendError with an actionable message when the requested
    backend cannot start (missing key, failed model download, ...).
    """
    name = name.lower()
    if name == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return _init_backend(OpenAIBackend)
        try:
            return _init_backend(HFBackend)
        except BackendError:
            return ExtractiveBackend()
    backends = {"openai": OpenAIBackend, "hf": HFBackend,
                "extractive": ExtractiveBackend}
    if name not in backends:
        raise ValueError(f"Unknown backend {name!r}. "
                         f"Choose from: auto, {', '.join(backends)}")
    return _init_backend(backends[name])


def _init_backend(cls: type[LLMBackend]) -> LLMBackend:
    """Instantiate a backend, converting failures into BackendError."""
    try:
        return cls()
    except RAGError:
        raise
    except Exception as exc:
        hint = {
            "openai": "Is OPENAI_API_KEY set and valid?",
            "hf": ("Could not download the HuggingFace model. "
                   "Check your network connection and try again."),
        }.get(getattr(cls, "name", ""), "Check the logs and try again.")
        raise BackendError(
            f"Backend {cls.name!r} failed to start. {hint}"
        ) from exc


# ---------------------------------------------------------------------------
# Main answering entry point
# ---------------------------------------------------------------------------

def resolve_backend_name(backend: str = "auto") -> str:
    """Return the concrete backend name without instantiating it."""
    backend = (backend or "auto").lower()
    if backend == "openai":
        return "openai"
    if backend == "hf":
        return "hf"
    if backend == "extractive":
        return "extractive"
    # auto: OpenAI if a key is set, else local HuggingFace, else extractive
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    try:
        import transformers  # noqa: F401
        return "hf"
    except ImportError:
        return "extractive"


def _expand_followup(question: str,
                     history: "list[dict] | None") -> str:
    """Make short follow-up questions standalone for retrieval.

    "tell me more" alone retrieves nothing useful; "tell me more" +
    the previous question retrieves the right neighborhood. Long,
    self-contained questions pass through untouched.
    """
    if not history or len(question.split()) > 12:
        return question
    last_q = (history[-1].get("question") or "").strip()
    if not last_q:
        return question
    return f"{last_q} {question}"


def _gather(question: str, top_k: int, max_distance: float,
            retriever: "Retriever | None",
            collection_label: str,
            history: "list[dict] | None",
            use_hybrid: bool,
            use_rerank: bool) -> tuple[str, list[RetrievedChunk],
                                       "Answer | None"]:
    """Shared retrieval pipeline for the answer and stream entry points.

    Returns (standalone_question, prompt_chunks, gate_answer). When
    gate_answer is not None, retrieval decided the question is out of
    scope (or input was empty) and it should be returned directly.
    """
    question = (question or "").strip()
    if not question:
        return question, [], Answer(text="Please ask a question.",
                                    backend="none")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Question is too long ({len(question)} chars; "
            f"max {MAX_QUESTION_CHARS}). Please shorten it."
        )
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}."
        )
    retriever = retriever or Retriever()
    standalone = _expand_followup(question, history)
    if use_hybrid:
        try:
            candidates = retriever.hybrid_retrieve(
                standalone, top_k=max(top_k * 4, 20))
        except (ModelLoadError, RAGError):
            candidates = retriever.retrieve(standalone,
                                            top_k=max(top_k * 4, 20))
    else:
        candidates = retriever.retrieve(standalone,
                                        top_k=max(top_k * 4, 20))
    if not candidates:
        return standalone, [], Answer(
            text=("This collection is empty. Add some documents "
                  "first, then ask away."),
            backend="none")
    # Relevance gate (Day 5): judged on the best dense cosine distance,
    # which is what the threshold was calibrated on.
    if candidates[0].score > max_distance:
        topics = ", ".join(f"'{t}'" for t in
                           dict.fromkeys(c.title for c in candidates))
        return standalone, [], Answer(
            text=(f"{collection_label} does not cover "
                  f"that topic. So far it covers: {topics}. "
                  "Try asking about one of those."),
            sources=[],
            backend="relevance-gate",
        )
    chunks = (rerank(standalone, candidates, top_k) if use_rerank
              else candidates[:top_k])
    return standalone, chunks, None


def _history_block(history: "list[dict] | None") -> str:
    """Format the last exchange so LLM backends resolve follow-ups."""
    if not history:
        return ""
    last = history[-1]
    q = (last.get("question") or "")[:500]
    a = (last.get("answer") or "")[:800]
    if not q:
        return ""
    return (f"Previous exchange (for context on follow-up questions):\n"
            f"Q: {q}\nA: {a}\n\n")


def answer_question(question: str, top_k: int = DEFAULT_TOP_K,
                    backend: str = "auto",
                    max_distance: float = RELEVANCE_THRESHOLD,
                    retriever: "Retriever | None" = None,
                    collection_label: str = "this collection",
                    history: "list[dict] | None" = None,
                    use_hybrid: bool = True,
                    use_rerank: bool = True) -> Answer:
    """Retrieve relevant chunks and generate a cited answer.

    Returns an Answer with .text (citations as [1], [2], ...) and .sources
    (the RetrievedChunk list the numbers refer to).

    Advanced retrieval: hybrid dense + BM25 search fused with RRF, then a
    cross-encoder rerank over a wide candidate pool. Short follow-up
    questions are expanded with the previous question for retrieval, and
    LLM backends see the last exchange for conversational context.

    Day 6: pass an already-loaded Retriever to skip reloading the
    embedding model on every call. Raises ValueError on invalid input
    and RAGError subclasses on index/model/backend failures.
    """
    standalone, chunks, gate = _gather(
        question, top_k, max_distance, retriever, collection_label,
        history, use_hybrid, use_rerank)
    if gate is not None:
        return gate
    llm = get_backend(backend)
    context_note = _history_block(history)
    text = llm.generate(standalone, chunks,
                        history_note=context_note)
    return Answer(text=text, sources=chunks, backend=llm.name)


def answer_question_stream(question: str, top_k: int = DEFAULT_TOP_K,
                           backend: str = "auto",
                           max_distance: float = RELEVANCE_THRESHOLD,
                           retriever: "Retriever | None" = None,
                           collection_label: str = "this collection",
                           history: "list[dict] | None" = None,
                           use_hybrid: bool = True,
                           use_rerank: bool = True):
    """Streaming variant of answer_question.

    Yields (event, payload) tuples: ("meta", Answer-with-empty-text) once
    the sources are known, then ("token", str) for each streamed token,
    then ("done", Answer). Falls back to non-streaming for backends
    without stream support.
    """
    standalone, chunks, gate = _gather(
        question, top_k, max_distance, retriever, collection_label,
        history, use_hybrid, use_rerank)
    if gate is not None:
        yield "done", gate
        return
    name = resolve_backend_name(backend)
    if name != "openai":
        llm = get_backend(backend)
        text = llm.generate(standalone, chunks,
                            history_note=_history_block(history))
        yield "done", Answer(text=text, sources=chunks,
                             backend=llm.name)
        return
    llm = _init_backend(OpenAIBackend)
    meta = Answer(text="", sources=chunks, backend=llm.name)
    yield "meta", meta
    full = []
    for tok in llm.generate_stream(standalone, chunks,
                                   history_note=_history_block(history)):
        full.append(tok)
        yield "token", tok
    yield "done", Answer(text="".join(full), sources=chunks,
                         backend=llm.name)


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
    except (RAGError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected: surface it, don't swallow it
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 2
    print(format_answer(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
