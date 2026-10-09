"""Document ingestion for user-provided content.

This is what turns the project from "Jashwanth's newsletter chatbot"
into a real-world RAG app anyone can use: point it at your own material
and chat with it.

Supported inputs:
    - File uploads: PDF, TXT, Markdown
    - Web page URLs (main article text is extracted)
    - Pasted raw text

Each input becomes a SourceDocument (title, text, url), which the caller
chunks and indexes with rag.index_documents().
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass


@dataclass
class SourceDocument:
    """One user-provided document, ready for chunking and indexing."""
    title: str
    text: str
    url: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pdf(data: bytes) -> str:
    """Extract text from PDF bytes. Returns "" when nothing is readable."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs the 'pypdf' package. "
            "Run: pip install -r requirements.txt"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue  # one bad page must not kill the whole document
    return "\n\n".join(p for p in parts if p.strip())


def extract_text_file(data: bytes, filename: str = "pasted.txt") -> str:
    """Decode a TXT/Markdown upload. Light markdown-noise stripping."""
    text = data.decode("utf-8", errors="replace")
    if filename.lower().endswith(".md"):
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
        text = re.sub(r"[*_`~>]", "", text)
    return text.strip()


def extract_upload(filename: str, data: bytes) -> SourceDocument:
    """Route an uploaded file to the right extractor by extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        text = extract_pdf(data)
    elif name.endswith((".txt", ".md", ".markdown")):
        text = extract_text_file(data, filename)
    else:
        raise ValueError(
            f"Unsupported file type: {filename!r}. "
            "Upload a PDF, TXT, or Markdown file."
        )
    if len(text) < 50:
        raise ValueError(
            f"Could not extract readable text from {filename!r}. "
            "Scanned-image PDFs are not supported yet."
        )
    return SourceDocument(title=filename, text=text)


def fetch_url(url: str, timeout: int = 20) -> SourceDocument:
    """Fetch a web page and extract its main article text."""
    try:
        from readability import Document
    except ImportError as exc:
        raise RuntimeError(
            "URL ingestion needs the 'readability-lxml' package. "
            "Run: pip install -r requirements.txt"
        ) from exc
    import requests

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (RAG-Chatbot/1.0)"},
        )
        resp.raise_for_status()
    except Exception as exc:
        raise ValueError(f"Could not fetch {url}: {exc}") from exc
    doc = Document(resp.text)
    title = (doc.title() or url).strip()
    text = re.sub(r"<[^>]+>", " ", doc.summary())
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 50:
        raise ValueError(
            f"No readable article text found at {url}. "
            "Paywalled or script-heavy pages may not work."
        )
    return SourceDocument(title=title, text=text, url=url)


def make_text_document(title: str, text: str) -> SourceDocument:
    """Build a document from pasted raw text."""
    title = (title or "Pasted text").strip()
    text = (text or "").strip()
    if len(text) < 50:
        raise ValueError("Paste at least a paragraph of text (50+ characters).")
    return SourceDocument(title=title, text=text)
