"""On-disk embedding cache keyed by SHA1 of text and cache namespace."""

import hashlib
import json
from pathlib import Path
import re

import structlog

from app.config import get_settings

logger = structlog.get_logger()


def _hash_text(text: str) -> str:
    """Generate SHA1 hash of text for cache key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Disk-based cache for text embeddings."""

    def __init__(self, cache_dir: Path | None = None, namespace: str = "default") -> None:
        settings = get_settings()
        normalized_namespace = re.sub(r"[^a-zA-Z0-9._-]+", "_", namespace).strip("._") or "default"
        base_cache_dir = cache_dir or settings.data_dir / "embeddings"
        self.cache_dir = base_cache_dir / normalized_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text_hash: str) -> Path:
        """Get the cache file path for a given hash."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = text_hash[:2]
        return self.cache_dir / subdir / f"{text_hash}.json"

    def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text, or None if not cached."""
        text_hash = _hash_text(text)
        cache_path = self._cache_path(text_hash)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return data["embedding"]  # type: ignore[no-any-return]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Cache read failed", hash=text_hash, error=str(e))
            return None

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding for text."""
        text_hash = _hash_text(text)
        cache_path = self._cache_path(text_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "hash": text_hash,
                        "embedding": embedding,
                        "dimension": len(embedding),
                    },
                    f,
                )
        except OSError as e:
            logger.warning("Cache write failed", hash=text_hash, error=str(e))

    def get_batch(self, texts: list[str]) -> tuple[list[list[float]], list[int]]:
        """Get cached embeddings for a batch.

        Returns:
            Tuple of (cached_embeddings, missing_indices)
        """
        cached: list[list[float]] = []
        missing: list[int] = []

        for i, text in enumerate(texts):
            emb = self.get(text)
            if emb is not None:
                cached.append(emb)
            else:
                missing.append(i)

        return cached, missing

    def set_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Cache embeddings for a batch of texts."""
        for text, embedding in zip(texts, embeddings, strict=True):
            self.set(text, embedding)

    def hit_rate(self, texts: list[str]) -> float:
        """Calculate cache hit rate for a batch of texts."""
        if not texts:
            return 0.0
        hits = sum(1 for t in texts if self.get(t) is not None)
        return hits / len(texts)
