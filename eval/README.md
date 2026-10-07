# Day 5 — quality pass

Eval set: `eval_questions.json` (10 questions).

- **q1–q6 (in_scope):** two questions per newsletter post, each with an
  expected source post and answer keywords.
- **q7–q8 (out_of_scope):** questions the newsletter does not cover;
  the bot must abstain, not invent.
- **q9 (edge_empty):** empty string — must not crash.
- **q10 (edge_gibberish):** nonsense input — must not crash, must abstain.

Harness: `run_eval.py` scores retrieval hit rate, keyword coverage,
citation presence, abstention behavior, and crash-freedom.

## Results (extractive backend, deterministic baseline)

| metric | baseline | after tuning |
|---|---|---|
| ran without crashing | 10/10 | 10/10 |
| retrieval hit rate | 6/6 | 6/6 |
| mean keyword coverage | 0.89 | 0.96 |
| answers with citations | 9/10 | 6/6 answerable* |
| correct abstentions | **0/3** | **3/3** |

\* The 4 non-cited rows are the 3 correct abstentions plus the empty
question — none of them should carry citations.

## Findings and fixes

1. **No abstention (worst finding).** Out-of-scope and gibberish
   questions got confident answers quoting irrelevant passages
   ("Based on the newsletter, here is what I found...").
   Fix: `answer_question()` now has a relevance gate — if the best
   chunk's cosine distance exceeds `RELEVANCE_THRESHOLD` (default 0.78,
   calibrated on this eval: in-scope 0.31–0.72, out-of-scope 0.81–0.94),
   it returns a graceful "not covered" message listing what the
   newsletter does cover. Override with the `RAG_RELEVANCE_THRESHOLD`
   env var or the `max_distance` argument.
2. **Prompt tuning.** `SYSTEM_PROMPT` now requires a citation on every
   factual sentence, forbids inventing titles/URLs, and spells out
   partial-coverage behavior (answer what's covered, name what isn't).
3. **HF backend was broken.** `pipeline("text2text-generation")` no
   longer exists in transformers 5.x. `HFBackend` now uses
   `AutoTokenizer` + `AutoModelForSeq2SeqLM.generate()` directly.
4. **Extractive snippet experiment.** A query-term sentence-selection
   variant was tried and reverted: on this corpus the chunk lead
   covered more expected keywords (0.96 vs 0.74 mean coverage).
   Snippet length raised 80 → 120 words instead.
