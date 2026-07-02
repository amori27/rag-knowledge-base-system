# RAG Knowledge Base System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A clean, lightweight RAG (Retrieval-Augmented Generation) pipeline in pure Python. Document chunking, embeddings, vector storage, and retrieval — all in a few hundred lines, with a swappable backend at each layer.

---

## Architecture

```
┌────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Documents     │──▶ │ DocumentProcessor│──▶ │ EmbeddingGener. │──▶ │ VectorStore  │
│  (text/files)  │    │  (chunking)      │    │ (text → vec)    │    │ (FAISS/Chroma)│
└────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────┘
                                                                              │
                                                                              ▼
                                                                  ┌──────────────────┐
                                                                  │  RAGPipeline     │
                                                                  │  retrieve + QA   │
                                                                  └──────────────────┘
```

---

## Quick Start

```python
from src.document import DocumentProcessor
from src.embedding import EmbeddingGenerator
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# 1. Build a pipeline
pipeline = RAGPipeline(
    embedding_model="text-embedding-ada-002",  # or any local sentence-transformer
    vector_store_type="faiss",                # or "chroma"
)

# 2. Add documents
docs = [
    {"id": "doc1", "text": "Retrieval-augmented generation grounds LLMs in your data."},
    {"id": "doc2", "text": "Vector search finds semantically similar chunks via embeddings."},
]
pipeline.add_documents(docs)

# 3. Query
hits = pipeline.retrieve("How does RAG reduce hallucination?", k=3)
for h in hits:
    print(f"[{h['score']:.3f}] {h['text']}")
```

---

## Modules

| Module | Purpose |
|---|---|
| `document.DocumentProcessor` | Splits text into overlapping chunks (`chunk_size`, `chunk_overlap`) |
| `embedding.EmbeddingGenerator` | Generates embedding vectors; backend-agnostic |
| `vector_store.VectorStore` | FAISS or Chroma backends; add/remove/query with metadata |
| `rag_pipeline.RAGPipeline` | Glue class: `add_documents`, `retrieve`, full QA flow |

---

## Configuration

| Knob | Default | Effect |
|---|---|---|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between adjacent chunks |
| `embedding_dim` | 1536 | Vector dimension (match your embedding model) |
| `store_type` | `faiss` | `faiss` (in-memory) or `chroma` (persistent) |
| `k` | 5 | Top-k chunks returned per query |

---

## Project Structure

```
rag-knowledge-base-system/
├── src/
│   ├── document.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
