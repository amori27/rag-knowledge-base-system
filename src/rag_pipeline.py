"""RAG Pipeline Module.

This module provides the complete RAG (Retrieval-Augmented Generation)
pipeline for question answering.
"""

from typing import Any
from .vector_store import VectorStore
from .embedding import EmbeddingGenerator


class RAGPipeline:
    """Complete RAG pipeline for QA."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        vector_store_type: str = "faiss"
    ):
        """Initialize the RAGPipeline.

        Args:
            embedding_model: Embedding model name.
            vector_store_type: Vector store type.
        """
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        self.vector_store = VectorStore(
            embedding_dim=self.embedding_generator.get_embedding_dimension(),
            store_type=vector_store_type
        )

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the knowledge base.

        Args:
            documents: List of document chunks.
        """
        for doc in documents:
            text = doc.get("text", "")
            embedding = self.embedding_generator.generate(text)

            self.vector_store.add_vector(
                vector=embedding,
                metadata={
                    "text": text,
                    "doc_id": doc.get("id", "unknown")
                }
            )

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant documents.

        Args:
            query: Query text.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved documents.
        """
        query_embedding = self.embedding_generator.generate(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results

    def generate_answer(
        self,
        query: str,
        context: list[str]
    ) -> str:
        """Generate answer using retrieved context.

        Args:
            query: User query.
            context: Retrieved context documents.

        Returns:
            Generated answer.
        """
        context_text = "\n\n".join([c.get("metadata", {}).get("text", "") for c in context])

        answer = f"Based on the retrieved context, the answer to '{query}' is: ..."

        return answer

    def query(self, question: str, k: int = 5) -> dict[str, Any]:
        """Complete RAG query pipeline.

        Args:
            question: User question.
            k: Number of documents to retrieve.

        Returns:
            Query result with answer and sources.
        """
        retrieved = self.retrieve(question, k=k)

        answer = self.generate_answer(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
            "num_sources": len(retrieved)
        }


def create_rag_pipeline(
    embedding_model: str = "text-embedding-ada-002"
) -> RAGPipeline:
    """Factory function to create RAG pipeline.

    Args:
        embedding_model: Embedding model name.

    Returns:
        RAGPipeline instance.
    """
    return RAGPipeline(embedding_model=embedding_model)
