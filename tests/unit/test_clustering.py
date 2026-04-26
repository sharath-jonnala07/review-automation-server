"""Tests for embeddings and clustering."""

from pathlib import Path

import numpy as np
import pytest

from app.clustering.cache import EmbeddingCache
from app.clustering.clusterer import ReviewClusterer
from app.clustering.embeddings import _normalize_huggingface_embeddings
from app.core.exceptions import ClusteringError


class TestEmbeddingCache:
    """Embedding cache tests."""

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        """Should store and retrieve embeddings."""
        cache = EmbeddingCache(cache_dir=tmp_path / "embeddings")
        text = "This is a test review"
        embedding = [0.1, 0.2, 0.3, 0.4]

        cache.set(text, embedding)
        retrieved = cache.get(text)

        assert retrieved == embedding

    def test_cache_miss(self, tmp_path: Path) -> None:
        """Should return None for uncached text."""
        cache = EmbeddingCache(cache_dir=tmp_path / "embeddings")
        result = cache.get("never seen before")
        assert result is None

    def test_cache_hit_rate(self, tmp_path: Path) -> None:
        """Should calculate hit rate correctly."""
        cache = EmbeddingCache(cache_dir=tmp_path / "embeddings")
        texts = ["one", "two", "three"]
        embeddings = [[0.1], [0.2], [0.3]]

        cache.set_batch(texts[:2], embeddings[:2])
        hit_rate = cache.hit_rate(texts)
        assert hit_rate == pytest.approx(2 / 3)

    def test_cache_namespaces_do_not_collide(self, tmp_path: Path) -> None:
        """Should keep embeddings isolated per provider/model namespace."""
        local_cache = EmbeddingCache(cache_dir=tmp_path / "embeddings", namespace="hf-local:model-a")
        api_cache = EmbeddingCache(cache_dir=tmp_path / "embeddings", namespace="hf-api:model-b")

        local_cache.set("same text", [0.1, 0.2, 0.3])

        assert local_cache.get("same text") == [0.1, 0.2, 0.3]
        assert api_cache.get("same text") is None


class TestReviewClusterer:
    """Clustering algorithm tests."""

    def test_clusterer_initialization(self) -> None:
        """Should initialize with default params."""
        clusterer = ReviewClusterer()
        assert clusterer.n_components == 15
        assert clusterer.min_cluster_size == 8

    def test_not_enough_reviews(self) -> None:
        """Should raise error when not enough reviews."""
        clusterer = ReviewClusterer(min_cluster_size=100)
        embeddings = [[0.1, 0.2]] * 5
        texts = ["short"] * 5

        with pytest.raises(ClusteringError):
            clusterer.cluster_reviews(embeddings, texts)

    def test_clusterer_retries_to_reach_target_clusters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should relax cluster settings when the initial pass under-produces clusters."""
        clusterer = ReviewClusterer(min_cluster_size=8, min_samples=2)
        attempts: list[tuple[int | None, int | None]] = []

        monkeypatch.setattr(clusterer, "_reduce_dimensions", lambda embeddings: np.array(embeddings))

        def fake_cluster(
            reduced: np.ndarray,
            *,
            min_cluster_size: int | None = None,
            min_samples: int | None = None,
        ) -> np.ndarray:
            attempts.append((min_cluster_size, min_samples))
            if min_cluster_size is None or min_cluster_size >= 7:
                return np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
            return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        monkeypatch.setattr(clusterer, "_cluster", fake_cluster)
        monkeypatch.setattr(clusterer, "_find_medoid", lambda embeddings, indices: indices[0])
        monkeypatch.setattr(clusterer, "_extract_keyphrases", lambda texts, indices: [texts[indices[0]]])

        clusters, labels = clusterer.cluster_reviews(
            embeddings=[[0.1, 0.2]] * 12,
            texts=[f"review-{i}" for i in range(12)],
            target_clusters=3,
        )

        assert len(clusters) == 3
        assert set(labels) == {0, 1, 2}
        assert any(min_cluster_size == 6 for min_cluster_size, _ in attempts)

    def test_clusterer_enforces_minimum_clusters_when_retries_stall(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should split the largest group when HDBSCAN never reaches the target."""
        clusterer = ReviewClusterer(min_cluster_size=8, min_samples=2)

        monkeypatch.setattr(clusterer, "_reduce_dimensions", lambda embeddings: np.array(embeddings))
        monkeypatch.setattr(
            clusterer,
            "_cluster",
            lambda reduced, *, min_cluster_size=None, min_samples=None: np.array(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ),
        )
        monkeypatch.setattr(clusterer, "_find_medoid", lambda embeddings, indices: indices[0])
        monkeypatch.setattr(clusterer, "_extract_keyphrases", lambda texts, indices: [texts[indices[0]]])

        clusters, labels = clusterer.cluster_reviews(
            embeddings=[[float(i), float(i % 3)] for i in range(12)],
            texts=[f"review-{i}" for i in range(12)],
            target_clusters=3,
        )

        assert len(clusters) == 3
        assert set(labels) == {0, 1, 2}
        assert sorted(len(cluster.review_indices) for cluster in clusters) == [3, 3, 6]

    def test_clusterer_rejects_mixed_embedding_dimensions(self) -> None:
        """Should fail fast when embeddings with different dimensions are mixed."""
        clusterer = ReviewClusterer(min_cluster_size=2)

        with pytest.raises(ClusteringError, match="Embedding dimensions were inconsistent"):
            clusterer.cluster_reviews(
                embeddings=[[0.1, 0.2], [0.3, 0.4, 0.5]],
                texts=["review-1", "review-2"],
                target_clusters=3,
            )


class TestHuggingFaceEmbeddingNormalization:
    """Normalization tests for Hugging Face hosted embedding responses."""

    def test_normalizes_sentence_vectors(self) -> None:
        payload = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        normalized = _normalize_huggingface_embeddings(payload)

        assert normalized == payload

    def test_averages_token_level_embeddings(self) -> None:
        payload = [
            [
                [1.0, 3.0],
                [3.0, 5.0],
            ]
        ]

        normalized = _normalize_huggingface_embeddings(payload)

        assert normalized == [[2.0, 4.0]]

    def test_rejects_invalid_payloads(self) -> None:
        with pytest.raises(ClusteringError):
            _normalize_huggingface_embeddings({"embedding": [0.1, 0.2]})
