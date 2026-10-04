"""Day 2: Chunk posts, embed them, and build a persistent ChromaDB vector index.

Usage: python scripts/build_index.py [--recreate]
Output: data/chroma/ — persistent ChromaDB collection "learnwithjashwanth"

Chunking: sliding word window (500 words, 100 overlap) so no lesson gets
split mid-thought. Each chunk keeps its source title, URL, and publish date
as metadata for citations at answer time.
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
POSTS_PATH = BASE_DIR / "data" / "posts.json"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "learnwithjashwanth"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)] if words else []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def load_chunks() -> tuple[list[str], list[dict]]:
    """Load posts and return (documents, metadatas) for every chunk."""
    posts = json.loads(POSTS_PATH.read_text())
    documents, metadatas = [], []
    for post in posts:
        for i, chunk in enumerate(chunk_text(post["text"])):
            documents.append(f"{post['title']}\n\n{chunk}")
            metadatas.append({
                "title": post["title"],
                "url": post["url"],
                "published": post.get("published", ""),
                "chunk": i,
            })
    return documents, metadatas


def build_index(recreate: bool = False) -> None:
    documents, metadatas = load_chunks()
    if not documents:
        print("No chunks to index. Run scripts/ingest.py first.")
        return

    print(f"Embedding {len(documents)} chunks with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(documents, batch_size=32,
                             show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [f"chunk-{i}" for i in range(len(documents))]
    collection.upsert(ids=ids, documents=documents,
                      metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {collection.count()} chunks -> {CHROMA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and rebuild the collection")
    args = parser.parse_args()
    build_index(recreate=args.recreate)


if __name__ == "__main__":
    main()
