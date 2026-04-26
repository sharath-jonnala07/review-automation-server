"""Embedding providers for review text."""

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
import threading
from typing import Any

import httpx
import numpy as np
import structlog
from openai import AsyncOpenAI

from app.clustering.cache import EmbeddingCache
from app.config import get_settings
from app.core.exceptions import ClusteringError

logger = structlog.get_logger()

# Type alias for embeddings
EmbeddingVector = list[float]
HOSTED_EMBEDDING_BATCH_SIZE = 32


def _embedding_cache_namespace(provider: str, model_name: str) -> str:
    """Return a provider/model-specific namespace for embedding cache files."""
    return f"{provider}:{model_name}"


def _normalize_huggingface_embeddings(payload: object) -> list[EmbeddingVector]:
    """Normalize Hugging Face inference payloads into sentence vectors."""
    if not isinstance(payload, list):
        raise ClusteringError("Hugging Face embeddings response was not a list")

    normalized: list[EmbeddingVector] = []
    for item in payload:
        if not isinstance(item, list) or not item:
            raise ClusteringError("Hugging Face embeddings item was not a non-empty list")

        if isinstance(item[0], (int, float)):
            normalized.append([float(value) for value in item])
            continue

        if isinstance(item[0], list):
            token_matrix = np.array(item, dtype=float)
            if token_matrix.ndim != 2:
                raise ClusteringError("Hugging Face token embeddings had an invalid shape")
            normalized.append(token_matrix.mean(axis=0).tolist())
            continue

        raise ClusteringError("Hugging Face embeddings response shape was not recognized")

    return normalized


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
        self.cache = EmbeddingCache(namespace=_embedding_cache_namespace("openai", self.model))

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
        self._model_lock = threading.Lock()
        self.cache = EmbeddingCache(
            namespace=_embedding_cache_namespace("huggingface-local", self.model_name)
        )

    def _load_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            with self._model_lock:
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


class HuggingFaceAPIEmbeddingProvider(EmbeddingProvider):
    """Hosted Hugging Face inference API embedding provider."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.api_key = api_key or settings.huggingface_api_key
        self.api_url = (api_url or settings.huggingface_api_url).rstrip("/")
        if not self.api_key:
            raise ClusteringError(
                "HUGGINGFACE_API_KEY not configured. Required for hosted Hugging Face embeddings. "
                "Set it in your .env or switch EMBEDDING_BACKEND to huggingface-local"
            )
        self.cache = EmbeddingCache(
            namespace=_embedding_cache_namespace("huggingface-api", self.model_name)
        )
        self._dimension = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a single hosted batch against Hugging Face."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_url}/{self.model_name}/pipeline/feature-extraction",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": texts,
                    "options": {"wait_for_model": True},
                },
            )
            response.raise_for_status()
            return _normalize_huggingface_embeddings(response.json())

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
            logger.info("Embedding cache hit", provider="huggingface-api", total=len(texts))
            return [embedding for embedding in cached_embeddings if embedding is not None]

        try:
            for batch_number, start in enumerate(
                range(0, len(missing_texts), HOSTED_EMBEDDING_BATCH_SIZE),
                start=1,
            ):
                batch_texts = missing_texts[start : start + HOSTED_EMBEDDING_BATCH_SIZE]
                batch_indices = missing_indices[start : start + HOSTED_EMBEDDING_BATCH_SIZE]
                logger.info(
                    "Embedding batch started",
                    provider="huggingface-api",
                    batch=batch_number,
                    batch_size=len(batch_texts),
                    total_missing=len(missing_texts),
                )
                new_embeddings = await self._embed_batch(batch_texts)

                if len(new_embeddings) != len(batch_texts):
                    raise ClusteringError(
                        "Hugging Face embeddings response count did not match the request",
                        context={"expected": len(batch_texts), "received": len(new_embeddings)},
                    )

                self._dimension = len(new_embeddings[0]) if new_embeddings else self._dimension
                for index, text, embedding in zip(batch_indices, batch_texts, new_embeddings, strict=True):
                    self.cache.set(text, embedding)
                    cached_embeddings[index] = embedding

            logger.info(
                "Embedding batch complete",
                provider="huggingface-api",
                total=len(texts),
                cached=len(texts) - len(missing_texts),
                computed=len(missing_texts),
            )
            return [embedding for embedding in cached_embeddings if embedding is not None]
        except httpx.HTTPStatusError as exc:
            raise ClusteringError(
                f"Hugging Face embedding failed: {exc.response.status_code} {exc.response.text}",
                context={"model": self.model_name, "text_count": len(texts)},
            ) from exc
        except httpx.HTTPError as exc:
            raise ClusteringError(
                f"Hugging Face embedding failed: {exc}",
                context={"model": self.model_name, "text_count": len(texts)},
            ) from exc


@lru_cache(maxsize=4)
def _get_openai_provider(model: str, api_key: str) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(model=model, api_key=api_key)


@lru_cache(maxsize=4)
def _get_local_provider(model_name: str) -> LocalEmbeddingProvider:
    return LocalEmbeddingProvider(model_name=model_name)


@lru_cache(maxsize=4)
def _get_huggingface_api_provider(
    model_name: str,
    api_key: str,
    api_url: str,
) -> HuggingFaceAPIEmbeddingProvider:
    return HuggingFaceAPIEmbeddingProvider(
        model_name=model_name,
        api_key=api_key,
        api_url=api_url,
    )


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get the appropriate embedding provider."""
    settings = get_settings()
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            return OpenAIEmbeddingProvider()
        return _get_openai_provider(settings.embedding_model, settings.openai_api_key)
    if settings.embedding_backend == "huggingface-api":
        if not settings.huggingface_api_key:
            return HuggingFaceAPIEmbeddingProvider()
        return _get_huggingface_api_provider(
            settings.embedding_model,
            settings.huggingface_api_key,
            settings.huggingface_api_url,
        )
    return _get_local_provider(settings.embedding_model)
