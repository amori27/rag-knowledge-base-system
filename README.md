# RAG Knowledge Base System
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Retrieval-Augmented Generation system with vector database integration, document processing, semantic search, and question answering pipeline.

## Description

Production-ready RAG system for building knowledge bases with efficient retrieval. Features document chunking, embeddings, vector storage, and hybrid search capabilities. Supports multiple vector databases and embedding models.

## Skills & Technologies

- Python 3.9+
- LangChain
- Chroma/Pinecone
- OpenAI Embeddings
- FAISS
- Document Processing
- Semantic Search
- RAG Pipeline

## Installation

```bash
git clone https://github.com/amori27/rag-knowledge-base-system.git
cd rag-knowledge-base-system
pip install -r requirements.txt
```

## Usage

### Create Knowledge Base

```python
from src.vector_store import VectorStore
from src.document import DocumentProcessor

processor = DocumentProcessor()
documents = processor.process_file("data/doc.pdf")
```

### Query Knowledge Base

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
answer = rag.query("What is machine learning?")
```

## Project Structure

```
rag-knowledge-base-system/
├── src/
│   ├── vector_store.py      # Vector storage
│   ├── document.py          # Document processing
│   ├── embedding.py         # Embedding generation
│   └── rag_pipeline.py       # RAG pipeline
├── requirements.txt
└── README.md
```

## References

- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chroma Documentation](https://docs.trychroma.com/)

## License

MIT License
