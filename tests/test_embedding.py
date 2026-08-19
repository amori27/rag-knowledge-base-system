"""Tests for embedding generation."""

from src.embedding import EmbeddingGenerator, generate_random_embedding


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""

    def test_default_model(self):
        gen = EmbeddingGenerator()
        assert gen.model == "text-embedding-ada-002"

    def test_custom_model(self):
        gen = EmbeddingGenerator(model="text-embedding-3-small")
        assert gen.model == "text-embedding-3-small"

    def test_generate_returns_list(self):
        gen = EmbeddingGenerator()
        result = gen.generate("hello world")
        assert isinstance(result, list)

    def test_generate_embedding_dimension(self):
        gen = EmbeddingGenerator()
        result = gen.generate("test")
        assert len(result) == 16

    def test_generate_values_between_zero_and_one(self):
        gen = EmbeddingGenerator()
        result = gen.generate("some text")
        for val in result:
            assert 0.0 <= val <= 1.0

    def test_generate_deterministic(self):
        gen = EmbeddingGenerator()
        r1 = gen.generate("same input")
        r2 = gen.generate("same input")
        assert r1 == r2

    def test_generate_different_for_different_inputs(self):
        gen = EmbeddingGenerator()
        r1 = gen.generate("alpha")
        r2 = gen.generate("beta")
        assert r1 != r2

    def test_generate_batch(self):
        gen = EmbeddingGenerator()
        results = gen.generate_batch(["a", "b", "c"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)
            assert len(r) == 16

    def test_generate_batch_empty(self):
        gen = EmbeddingGenerator()
        results = gen.generate_batch([])
        assert results == []

    def test_get_embedding_dimension(self):
        gen = EmbeddingGenerator()
        assert gen.get_embedding_dimension() == 1536

    def test_generate_empty_string(self):
        gen = EmbeddingGenerator()
        result = gen.generate("")
        assert isinstance(result, list)
        assert len(result) == 16

    def test_generate_long_text(self):
        gen = EmbeddingGenerator()
        long_text = "word " * 10000
        result = gen.generate(long_text)
        assert isinstance(result, list)
        assert len(result) == 16


class TestGenerateRandomEmbedding:
    """Test generate_random_embedding helper."""

    def test_default_dimension(self):
        result = generate_random_embedding()
        assert len(result) == 1536

    def test_custom_dimension(self):
        result = generate_random_embedding(dim=128)
        assert len(result) == 128

    def test_values_between_zero_and_one(self):
        result = generate_random_embedding(dim=10)
        for val in result:
            assert 0.0 <= val <= 1.0

    def test_returns_list(self):
        result = generate_random_embedding(dim=5)
        assert isinstance(result, list)
