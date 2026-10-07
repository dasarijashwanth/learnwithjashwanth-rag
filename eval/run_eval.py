"""Day 5: quality-pass evaluation harness.

Runs the 10 sample questions in eval/eval_questions.json through the
Day 3 RAG pipeline and scores four things per question:

    retrieval_hit  - the expected post is among the retrieved sources
    keywords       - fraction of expected keywords present in the answer
    citations      - the answer carries [n] citation markers
    abstention     - for out-of-scope / gibberish questions, the answer
                     admits the newsletter does not cover them

Usage:
    python eval/run_eval.py [--backend extractive] [--out eval/results.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

from rag import Retriever, answer_question  # noqa: E402

EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"

ABSTAIN_PHRASES = [
    "does not cover", "do not cover", "doesn't cover", "don't cover",
    "not covered", "isn't covered", "no information", "cannot answer",
    "can't answer", "not in the newsletter", "ask a question",
]

CITATION_RE = re.compile(r"\[\d+\]")


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ABSTAIN_PHRASES)


def evaluate(backend: str) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text())
    retriever = Retriever()
    results = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"],
               "question": q["question"]}
        try:
            chunks = retriever.retrieve(q["question"], top_k=5)
            answer = answer_question(q["question"], top_k=5,
                                     backend=backend)
            row["crashed"] = False
            row["error"] = None
            titles = [c.title for c in chunks]
            row["retrieved_titles"] = titles
            row["best_distance"] = chunks[0].score if chunks else None
            row["answer_preview"] = answer.text[:400]
            row["answer_backend"] = answer.backend

            if q.get("expected_post"):
                row["retrieval_hit"] = q["expected_post"] in titles
            else:
                row["retrieval_hit"] = None

            keywords = q.get("keywords", [])
            if keywords:
                lowered = answer.text.lower()
                hits = sum(1 for k in keywords if k.lower() in lowered)
                row["keyword_coverage"] = round(hits / len(keywords), 2)
            else:
                row["keyword_coverage"] = None

            row["citations"] = bool(CITATION_RE.search(answer.text))

            if q.get("want_abstain"):
                row["abstained"] = is_abstention(answer.text)
            else:
                row["abstained"] = None
        except Exception as exc:  # an edge case must never crash the bot
            row["crashed"] = True
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return {"backend": backend, "questions": results}


def summarize(report: dict) -> str:
    rows = report["questions"]
    ok = [r for r in rows if not r["crashed"]]
    lines = [f"Backend: {report['backend']} | "
             f"{len(ok)}/{len(rows)} questions ran without crashing"]
    hits = [r["retrieval_hit"] for r in ok if r["retrieval_hit"] is not None]
    cov = [r["keyword_coverage"] for r in ok
           if r["keyword_coverage"] is not None]
    cit = [r["citations"] for r in ok]
    abs_wanted = [r["abstained"] for r in ok if r["abstained"] is not None]
    if hits:
        lines.append(f"retrieval hit rate: {sum(hits)}/{len(hits)}")
    if cov:
        lines.append(f"mean keyword coverage: {sum(cov)/len(cov):.2f}")
    if cit:
        lines.append(f"answers with citations: {sum(cit)}/{len(cit)}")
    if abs_wanted:
        lines.append(f"correct abstentions: {sum(abs_wanted)}/"
                     f"{len(abs_wanted)}")
    lines.append("")
    for r in rows:
        flag = "CRASH" if r["crashed"] else (
            f"hit={r['retrieval_hit']} cov={r['keyword_coverage']} "
            f"cite={r['citations']} abstain={r['abstained']} "
            f"d={r.get('best_distance') and round(r['best_distance'], 3)}")
        lines.append(f"  {r['id']} [{r['type']}] {flag}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Day 5 eval set.")
    parser.add_argument("--backend", default="extractive",
                        choices=["auto", "openai", "hf", "extractive"])
    parser.add_argument("--out", default=str(EVAL_DIR / "results.json"))
    args = parser.parse_args()

    report = evaluate(args.backend)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(summarize(report))
    print(f"\nFull report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
