"""Day 1: Ingest every post from the Learn with Jashwanth Substack RSS feed.

Usage: python scripts/ingest.py
Output: data/posts.json — list of {title, url, published, text}
"""

import json
import re
from html import unescape
from pathlib import Path

import feedparser

FEED_URL = "https://learnwithjashwanth.substack.com/feed"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "posts.json"


def clean_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    feed = feedparser.parse(FEED_URL)
    posts = []
    for entry in feed.entries:
        content = ""
        if entry.get("content"):
            content = entry.content[0].get("value", "")
        elif entry.get("summary"):
            content = entry.summary
        posts.append(
            {
                "title": entry.get("title", "").strip(),
                "url": entry.get("link", ""),
                "published": entry.get("published", ""),
                "text": clean_html(content),
            }
        )
    posts = [p for p in posts if p["text"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(posts, indent=2, ensure_ascii=False))
    print(f"Ingested {len(posts)} posts -> {OUT_PATH}")


if __name__ == "__main__":
    main()
