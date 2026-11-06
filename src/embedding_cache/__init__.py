"""SQLite disk cache for embeddings keyed by content hash, model id and dimension."""

from __future__ import annotations

from embedding_cache.cache import (
    CacheStats,
    EmbedderPort,
    EmbeddingCache,
    EmbeddingCacheError,
    content_key,
)

__all__ = [
    "CacheStats",
    "EmbedderPort",
    "EmbeddingCache",
    "EmbeddingCacheError",
    "__version__",
    "content_key",
]

__version__ = "0.1.0"
