"""Embedding providers for review text."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.clustering.cache import EmbeddingCache
from app.config import get_settings
from app.core.exceptions import ClusteringError

logger = structlog.get_logger()

# Type alias for embeddings
EmbeddingVector = list[float]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a list of texts into vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API embedding provider.

    Optional fallback when users explicitly choose hosted embeddings.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.embedding_model
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ClusteringError(
                "OPENAI_API_KEY not configured. Required for embeddings. "
                "Set it in your .env or switch EMBEDDING_BACKEND to huggingface-local"
            )
        self.client = AsyncOpenAI(api_key=api_key)
        self.cache = EmbeddingCache()

    @property
    def dimension(self) -> int:
        if "3-small" in self.model:
            return 1536
        if "3-large" in self.model:
            return 3072
        return 1536

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []

        cached_embeddings: list[EmbeddingVector | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(text)
                continue
            cached_embeddings[index] = cached

        if not missing_texts:
            return [embedding for embedding in cached_embeddings if embedding is not None]

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=missing_texts,
            )
            new_embeddings = [item.embedding for item in response.data]
            for index, text, embedding in zip(missing_indices, missing_texts, new_embeddings, strict=True):
                self.cache.set(text, embedding)
                cached_embeddings[index] = embedding
            logger.info(
                "Embedding batch complete",
                provider="openai",
                total=len(texts),
                cached=len(texts) - len(missing_texts),
                computed=len(missing_texts),
            )
            return [embedding for embedding in cached_embeddings if embedding is not None]
        except Exception as e:
            raise ClusteringError(
                f"OpenAI embedding failed: {e}",
                context={"model": self.model, "text_count": len(texts)},
            ) from e


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider backed by Hugging Face."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model: Any = None
        self._dimension = 1024  # Qwen/Qwen3-Embedding-0.6B default
        self.cache = EmbeddingCache()

    def _load_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local embedding model", model=self.model_name)
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError as e:
                raise ClusteringError(
                    "sentence-transformers not installed. "
                    "Install with: uv pip install sentence-transformers"
                ) from e
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []

        cached_embeddings: list[EmbeddingVector | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(text)
                continue
            cached_embeddings[index] = cached

        if not missing_texts:
            logger.info("Embedding cache hit", provider="huggingface-local", total=len(texts))
            return [embedding for embedding in cached_embeddings if embedding is not None]

        model = await asyncio.to_thread(self._load_model)

        # Run in thread pool since sentence-transformers is CPU-bound
        embeddings = await asyncio.to_thread(
            lambda: model.encode(
                missing_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )

        computed_embeddings = [emb.tolist() for emb in embeddings]
        for index, text, embedding in zip(missing_indices, missing_texts, computed_embeddings, strict=True):
            self.cache.set(text, embedding)
            cached_embeddings[index] = embedding

        logger.info(
            "Embedding batch complete",
            provider="huggingface-local",
            total=len(texts),
            cached=len(texts) - len(missing_texts),
            computed=len(missing_texts),
        )
        return [embedding for embedding in cached_embeddings if embedding is not None]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get the appropriate embedding provider."""
    settings = get_settings()
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingProvider()
    return LocalEmbeddingProvider()
