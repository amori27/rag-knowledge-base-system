"""Tests for vector store operations."""

import sys
import numpy as np
from unittest.mock import MagicMock, patch


def _make_vector_store(dim=4, store_type="faiss"):
    """Create a VectorStore with faiss mocked via sys.modules."""
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatL2.return_value = mock_index

    with patch.dict(sys.modules, {"faiss": mock_faiss}):
        if "src.vector_store" in sys.modules:
            del sys.modules["src.vector_store"]
        from src.vector_store import VectorStore
        vs = VectorStore(embedding_dim=dim, store_type=store_type)
        return vs, mock_faiss, mock_index


class TestVectorStoreInitialization:
    """Test VectorStore initialization."""

    def test_default_dimensions(self):
        vs, _, _ = _make_vector_store()
        assert vs.embedding_dim == 4

    def test_store_type_faiss(self):
        vs, _, _ = _make_vector_store()
        assert vs.store_type == "faiss"

    def test_initially_empty(self):
        vs, _, _ = _make_vector_store()
        assert vs.vectors == []
        assert vs.metadata == []

    def test_creates_faiss_index(self):
        _, mock_faiss, _ = _make_vector_store()
        mock_faiss.IndexFlatL2.assert_called_once_with(4)


class TestVectorStoreAddVector:
    """Test adding vectors to the store."""

    def test_add_numpy_vector(self):
        vs, _, _ = _make_vector_store()
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        vs.add_vector(vec)
        assert len(vs.vectors) == 1

    def test_add_list_vector(self):
        vs, _, _ = _make_vector_store()
        vs.add_vector([1.0, 2.0, 3.0, 4.0])
        assert len(vs.vectors) == 1

    def test_add_with_metadata(self):
        vs, _, _ = _make_vector_store()
        vs.add_vector([1.0, 2.0, 3.0, 4.0], metadata={"doc_id": "abc"})
        assert vs.metadata[0] == {"doc_id": "abc"}

    def test_add_without_metadata(self):
        vs, _, _ = _make_vector_store()
        vs.add_vector([1.0, 2.0, 3.0, 4.0])
        assert vs.metadata[0] == {}

    def test_add_multiple_vectors(self):
        vs, _, _ = _make_vector_store()
        for i in range(5):
            vs.add_vector([float(i)] * 4)
        assert len(vs.vectors) == 5
        assert len(vs.metadata) == 5


class TestVectorStoreSearch:
    """Test similarity search."""

    def test_search_returns_results(self):
        vs, _, mock_index = _make_vector_store()
        mock_index.search.return_value = (
            np.array([[0.0, 1.0]]),
            np.array([[0, 1]]),
        )
        vs.add_vector([1.0, 0.0, 0.0, 0.0])
        vs.add_vector([0.0, 1.0, 0.0, 0.0])
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_search_with_list_query(self):
        vs, _, mock_index = _make_vector_store()
        mock_index.search.return_value = (
            np.array([[0.0]]),
            np.array([[0]]),
        )
        vs.add_vector([1.0, 2.0, 3.0, 4.0])
        results = vs.search([1.0, 2.0, 3.0, 4.0], k=1)
        assert len(results) == 1

    def test_search_result_structure(self):
        vs, _, mock_index = _make_vector_store()
        mock_index.search.return_value = (
            np.array([[0.5]]),
            np.array([[0]]),
        )
        vs.add_vector([1.0, 2.0, 3.0, 4.0], metadata={"text": "hello"})
        results = vs.search([1.0, 2.0, 3.0, 4.0], k=1)
        assert len(results) == 1
        r = results[0]
        assert "rank" in r
        assert "distance" in r
        assert "metadata" in r
        assert r["rank"] == 1
        assert r["distance"] == 0.5

    def test_search_respects_k(self):
        vs, _, mock_index = _make_vector_store()
        mock_index.search.return_value = (
            np.array([[0.0, 0.1, 0.2]]),
            np.array([[0, 1, 2]]),
        )
        for i in range(10):
            vs.add_vector([float(i)] * 4)
        results = vs.search([0.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_search_empty_store(self):
        vs, _, mock_index = _make_vector_store()
        mock_index.search.return_value = (np.array([[]]), np.array([[]]))
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []


class TestVectorStoreSaveLoad:
    """Test save and load operations."""

    def test_save_calls_faiss(self):
        vs, mock_faiss, _ = _make_vector_store()
        vs.add_vector([1.0, 2.0, 3.0, 4.0])
        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            vs.save("/tmp/test_store")
        mock_faiss.write_index.assert_called_once()

    def test_load_calls_faiss(self):
        vs, mock_faiss, _ = _make_vector_store()
        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            vs.load("/tmp/test_store")
        mock_faiss.read_index.assert_called_once()

    def test_save_passes_correct_path(self):
        vs, mock_faiss, _ = _make_vector_store()
        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            vs.save("/data/my_store")
        args = mock_faiss.write_index.call_args[0]
        assert args[1] == "/data/my_store.index"


class TestCreateVectorStore:
    """Test factory function."""

    def test_create_with_mock(self):
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatL2.return_value = MagicMock()
        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            if "src.vector_store" in sys.modules:
                del sys.modules["src.vector_store"]
            from src.vector_store import create_vector_store
            vs = create_vector_store()
            assert vs.embedding_dim == 1536
            assert vs.store_type == "faiss"

    def test_create_custom(self):
        vs, _, _ = _make_vector_store(dim=256, store_type="chroma")
        assert vs.embedding_dim == 256
        assert vs.store_type == "chroma"
