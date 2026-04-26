"""Clustering pipeline: UMAP + HDBSCAN for review grouping."""

from dataclasses import dataclass

import numpy as np
import structlog

from app.core.exceptions import ClusteringError

logger = structlog.get_logger()


@dataclass(frozen=True)
class Cluster:
    """A cluster of reviews with metadata."""

    id: int
    review_indices: list[int]
    medoid_index: int
    keyphrases: list[str]


class ReviewClusterer:
    """UMAP + HDBSCAN clustering for review embeddings."""

    _keyword_model: object | None = None

    def __init__(
        self,
        n_components: int = 15,
        min_cluster_size: int = 8,
        min_samples: int = 2,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.random_state = random_state

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensions with UMAP."""
        try:
            import umap
        except ImportError as e:
            raise ClusteringError(
                "umap-learn not installed. Install with: uv pip install umap-learn"
            ) from e

        reducer = umap.UMAP(
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_random_state(),
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
        )
        return reducer.fit_transform(embeddings)  # type: ignore[no-any-return]

    def _cluster(
        self,
        reduced: np.ndarray,
        *,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
    ) -> np.ndarray:
        """Run HDBSCAN on reduced embeddings."""
        try:
            import hdbscan
        except ImportError as e:
            raise ClusteringError(
                "hdbscan not installed. Install with: uv pip install hdbscan"
            ) from e

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size or min(self.min_cluster_size, len(reduced)),
            min_samples=self.min_samples if min_samples is None else min_samples,
            metric="euclidean",
        )
        return clusterer.fit_predict(reduced)  # type: ignore[no-any-return]

    def _count_clusters(self, labels: np.ndarray) -> int:
        """Count non-noise clusters in an HDBSCAN label array."""
        return len(set(labels) - {-1})

    def _partition_indices(
        self,
        reduced: np.ndarray,
        indices: np.ndarray,
        parts: int = 2,
    ) -> list[np.ndarray]:
        """Split a group along its highest-variance axis."""
        if len(indices) < parts:
            return [indices]

        points = reduced[indices]
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        axis = int(np.argmax(np.var(points, axis=0)))
        ordered = indices[np.argsort(points[:, axis], kind="stable")]
        return [segment for segment in np.array_split(ordered, parts) if len(segment) > 0]

    def _enforce_minimum_clusters(
        self,
        reduced: np.ndarray,
        labels: np.ndarray,
        target_clusters: int | None,
    ) -> np.ndarray:
        """Create deterministic fallback partitions when HDBSCAN under-produces clusters."""
        if target_clusters is None:
            return labels

        cluster_count = self._count_clusters(labels)
        if cluster_count >= target_clusters:
            return labels

        groups = [
            np.flatnonzero(labels == label)
            for label in sorted(set(labels) - {-1})
        ]
        noise_group = np.flatnonzero(labels == -1)
        if noise_group.size > 0:
            groups.append(noise_group)
        if not groups:
            groups = [np.arange(len(labels), dtype=int)]

        groups.sort(key=len, reverse=True)

        while len(groups) < target_clusters:
            splittable = [index for index, group in enumerate(groups) if len(group) >= 2]
            if not splittable:
                break

            split_index = max(splittable, key=lambda index: len(groups[index]))
            group = groups.pop(split_index)
            partitions = self._partition_indices(reduced, group)
            if len(partitions) < 2:
                groups.insert(split_index, group)
                break
            groups.extend(partitions)
            groups.sort(key=len, reverse=True)

        if len(groups) < target_clusters:
            return labels

        enforced = np.full(labels.shape, -1, dtype=int)
        for cluster_id, group in enumerate(groups):
            enforced[group] = cluster_id

        logger.info(
            "Applied fallback cluster partitioning",
            original_clusters=cluster_count,
            enforced_clusters=len(groups),
        )
        return enforced

    def _cluster_with_target(
        self,
        reduced: np.ndarray,
        target_clusters: int | None,
    ) -> np.ndarray:
        """Retry clustering with smaller cluster sizes until the target is met."""
        labels = self._cluster(reduced)
        if target_clusters is None:
            return labels

        best_labels = labels
        best_count = self._count_clusters(labels)
        if best_count >= target_clusters:
            return labels

        starting_size = min(self.min_cluster_size, len(reduced))
        min_cluster_sizes = range(starting_size - 1, 3, -1)
        min_samples_candidates = [self.min_samples]
        if self.min_samples != 1:
            min_samples_candidates.append(1)

        for min_samples in min_samples_candidates:
            for min_cluster_size in min_cluster_sizes:
                candidate = self._cluster(
                    reduced,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                cluster_count = self._count_clusters(candidate)
                logger.info(
                    "Retrying clustering with relaxed settings",
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    n_clusters=cluster_count,
                )
                if cluster_count > best_count:
                    best_labels = candidate
                    best_count = cluster_count
                if cluster_count >= target_clusters:
                    return candidate

            return self._enforce_minimum_clusters(reduced, best_labels, target_clusters)

    def _find_medoid(self, embeddings: np.ndarray, indices: list[int]) -> int:
        """Find the medoid (closest to centroid) of a cluster."""
        cluster_embeddings = embeddings[indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        medoid_local_idx = int(np.argmin(distances))
        return indices[medoid_local_idx]

    def _extract_keyphrases(
        self, texts: list[str], indices: list[int]
    ) -> list[str]:
        """Extract keyphrases for a cluster using KeyBERT."""
        try:
            from keybert import KeyBERT
        except ImportError as e:
            raise ClusteringError(
                "keybert not installed. Install with: uv pip install keybert"
            ) from e

        cluster_texts = [texts[i] for i in indices]
        combined = " ".join(cluster_texts)

        if ReviewClusterer._keyword_model is None:
            ReviewClusterer._keyword_model = KeyBERT()
        kw_model = ReviewClusterer._keyword_model
        keywords = kw_model.extract_keywords(
            combined,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=8,
        )
        return [kw[0] for kw in keywords]

    def random_random_state(self) -> int:
        """Return the random state for reproducibility."""
        return self.random_state

    def cluster_reviews(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        target_clusters: int | None = None,
    ) -> tuple[list[Cluster], np.ndarray]:
        """Cluster reviews and return cluster assignments.

        Args:
            embeddings: List of embedding vectors
            texts: Original review texts

        Returns:
            Tuple of (clusters, labels_array)

        Raises:
            ClusteringError: If clustering fails
        """
        if len(embeddings) < self.min_cluster_size:
            raise ClusteringError(
                f"Not enough reviews to cluster: {len(embeddings)} < {self.min_cluster_size}"
            )

        embeddings_array = np.array(embeddings)

        # Dimensionality reduction
        logger.info("Running UMAP", n_samples=len(embeddings), n_components=self.n_components)
        reduced = self._reduce_dimensions(embeddings_array)

        # Clustering
        logger.info("Running HDBSCAN", min_cluster_size=self.min_cluster_size)
        labels = self._cluster_with_target(reduced, target_clusters)

        # Build clusters
        unique_labels = sorted(set(labels) - {-1})  # -1 is noise
        clusters: list[Cluster] = []

        for label in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            medoid = self._find_medoid(embeddings_array, indices)
            keyphrases = self._extract_keyphrases(texts, indices)

            clusters.append(
                Cluster(
                    id=int(label),
                    review_indices=indices,
                    medoid_index=medoid,
                    keyphrases=keyphrases,
                )
            )

            clusters.sort(key=lambda cluster: len(cluster.review_indices), reverse=True)

        noise_count = int(np.sum(labels == -1))
        noise_ratio = noise_count / len(labels) if labels.size > 0 else 0

        logger.info(
            "Clustering complete",
            n_clusters=len(clusters),
            noise_ratio=round(noise_ratio, 2),
            total_reviews=len(embeddings),
        )

        return clusters, labels

    def get_noise_indices(self, labels: np.ndarray) -> list[int]:
        """Get indices of reviews classified as noise."""
        return [i for i, label in enumerate(labels) if label == -1]
