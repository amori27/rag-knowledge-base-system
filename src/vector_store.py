"""Vector Store Module.

This module provides vector storage and retrieval capabilities
using FAISS and Chroma.
"""

from typing import Any
import numpy as np


class VectorStore:
    """Handles vector storage and similarity search."""

    def __init__(self, embedding_dim: int = 1536, store_type: str = "faiss"):
        """Initialize the VectorStore.

        Args:
            embedding_dim: Dimension of embeddings.
            store_type: Type of vector store (faiss, chroma).
        """
        self.embedding_dim = embedding_dim
        self.store_type = store_type
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict[str, Any]] = []
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize the vector store backend."""
        if self.store_type == "faiss":
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = None

    def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a vector to the store.

        Args:
            vector: Embedding vector.
            metadata: Optional metadata.
        """
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape[0] != self.embedding_dim:
            vector = vector.reshape(-1)

        if self.store_type == "faiss":
            self.index.add(vector.reshape(1, -1).astype(np.float32))

        self.vectors.append(vector)
        self.metadata.append(metadata or {})

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding.
            k: Number of results.

        Returns:
            List of search results with scores and metadata.
        """
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        if self.store_type == "faiss":
            distances, indices = self.index.search(query_vector, k)
        else:
            distances = [0.0] * k
            indices = [[i for i in range(min(k, len(self.vectors)))]]

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                results.append({
                    "rank": i + 1,
                    "distance": float(dist),
                    "metadata": self.metadata[idx]
                })

        return results

    def save(self, path: str) -> None:
        """Save the vector store to disk.

        Args:
            path: Path to save to.
        """
        if self.store_type == "faiss":
            import faiss
            faiss.write_index(self.index, f"{path}.index")

    def load(self, path: str) -> None:
        """Load the vector store from disk.

        Args:
            path: Path to load from.
        """
        if self.store_type == "faiss":
            import faiss
            self.index = faiss.read_index(f"{path}.index")


def create_vector_store(
    embedding_dim: int = 1536,
    store_type: str = "faiss"
) -> VectorStore:
    """Factory function to create a vector store.

    Args:
        embedding_dim: Embedding dimension.
        store_type: Store type.

    Returns:
        VectorStore instance.
    """
    return VectorStore(embedding_dim=embedding_dim, store_type=store_type)
