"""
test_rag.py

Tests the RAG service end-to-end:
  1. Ingests everything in data/ into the vector store
  2. Asks a sample question and shows what gets retrieved

Run this AFTER test_setup.py works.

Note: the first run will download a small embedding model (~90MB) from
Hugging Face. That's normal and only happens once — it's cached after.
"""

from services.rag_service import RAGService

rag = RAGService()

num_chunks = rag.ingest_folder()
print(f"Ingested {num_chunks} chunks from data/\n")

query = "What is a deterministic finite automaton?"
results = rag.retrieve(query, k=2)

print(f"Top matches for: '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"--- Match {i} (from {doc.metadata.get('source')}) ---")
    print(doc.page_content[:300])
    print()
