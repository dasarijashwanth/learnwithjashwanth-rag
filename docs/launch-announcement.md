# Day 7 launch announcement — DRAFT, needs Jashwanth's approval before posting

Fill in the demo + repo URLs after deploy. No em dashes (his rule).
The LinkedIn post should carry his self @mention when scheduled natively.

## LinkedIn post

🤖 I built an AI chatbot over my own newsletter, and it only answers from my actual posts.

Week 1 of my weekly project series is DONE. This one is a RAG chatbot
that reads every "Learn with Jashwanth" post and answers your questions
with citations back to the exact source.

Ask it something the newsletter doesn't cover and it says so instead of
hallucinating. Honestly, that abstention part is the piece I am proudest of.

How it came together, 7 daily slices:
✅ RSS ingestion into clean JSON
✅ Chunking + embeddings + ChromaDB vector index
✅ Retrieval pipeline with cited answers (OpenAI, HuggingFace, or extractive fallback)
✅ Streamlit chat UI with source expanders
✅ 10-question eval harness: 6/6 retrieval hits, 3/3 honest abstentions
✅ Error handling, answer caching, and a self-building index for deploy
✅ Shipped to Streamlit Cloud

🔴 Try the live demo: <DEMO_URL>
💻 Full source on GitHub: <REPO_URL>

**Question for you:** what is the first thing you would ask a chatbot
trained on YOUR content? 👇

#AI #RAG #MachineLearning #Python #Streamlit #BuildInPublic #DataScience

— @[Jashwanth Dasari] (add native @mention when scheduling)

## Substack note

Quick one: I turned the newsletter into a chatbot 🤖

Ask it anything about the posts and it answers with citations, straight
from the source. If it doesn't know something, it says so. No made-up facts.

Try it here: <DEMO_URL>

This was Week 1 of a weekly build series. One production-grade data/AI
project every week, built in daily slices, code open on GitHub.

What should Week 2 be? Tell me in the comments 👇

— Jashwanth Dasari
