"""SQLite-backed cache for embedding vectors.

Only content hashes, vectors and timestamps are stored; the source text never reaches the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import sqlite3
import struct
import time
import unicodedata
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SCHEMA_VERSION = "1"
MAX_MODEL_ID_CHARS = 256
MAX_DIMENSION = 65_536
DEFAULT_MAX_TEXT_CHARS = 1_000_000
MAX_MESSAGE_CHARS = 200

ErrorCode = Literal["embedder_contract", "schema", "invalid_argument", "input_too_large"]

_CREATE_META = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
_CREATE_EMBEDDINGS = (
    "CREATE TABLE IF NOT EXISTS embeddings ("
    "key TEXT PRIMARY KEY, model_id TEXT, dimension INTEGER, "
    "vector BLOB, created_at REAL, accessed_at REAL)"
)
_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_embeddings_accessed ON embeddings (accessed_at, key)"
)
_SELECT_KEYS = "SELECT key, vector FROM embeddings WHERE key IN (SELECT value FROM json_each(?))"
_TOUCH_KEYS = "UPDATE embeddings SET accessed_at = ? WHERE key IN (SELECT value FROM json_each(?))"
_INSERT = (
    "INSERT OR REPLACE INTO embeddings "
    "(key, model_id, dimension, vector, created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?)"
)
_EVICT = (
    "DELETE FROM embeddings WHERE key IN "
    "(SELECT key FROM embeddings ORDER BY accessed_at ASC, key ASC LIMIT ?)"
)


class EmbeddingCacheError(Exception):
    """Base error of the library.

    ``code`` is a stable machine-readable identifier. Messages never contain input text.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message[:MAX_MESSAGE_CHARS])
        self.code: ErrorCode = code


class EmbedderPort(Protocol):
    """Synchronous embedder the cache wraps."""

    model_id: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector of ``dimension`` floats per text, in order."""


@dataclass(frozen=True)
class CacheStats:
    """Counters of one cache instance; ``entries`` is read from the database."""

    hits: int
    misses: int
    entries: int
    evictions: int
    corrupt: int


def _check_identity(model_id: object, dimension: object) -> None:
    if not isinstance(model_id, str):
        raise EmbeddingCacheError("invalid_argument", "model_id must be a string")
    if "\x00" in model_id:
        raise EmbeddingCacheError("invalid_argument", "model_id must not contain NUL")
    if len(model_id) > MAX_MODEL_ID_CHARS:
        raise EmbeddingCacheError(
            "invalid_argument",
            f"model_id must be at most {MAX_MODEL_ID_CHARS} characters",
        )
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise EmbeddingCacheError("invalid_argument", "dimension must be an integer")
    if not 1 <= dimension <= MAX_DIMENSION:
        raise EmbeddingCacheError(
            "invalid_argument",
            f"dimension must be between 1 and {MAX_DIMENSION}",
        )


def _check_text(text: object, limit: int) -> None:
    if not isinstance(text, str):
        raise EmbeddingCacheError("invalid_argument", "text must be a string")
    if len(text) > limit:
        raise EmbeddingCacheError("input_too_large", f"text exceeds {limit} characters")


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EmbeddingCacheError("invalid_argument", f"{name} must be a positive integer")


def _digest(text: str, model_id: str, dimension: int, normalize: bool) -> str:
    if normalize:
        text = unicodedata.normalize("NFC", text).rstrip()
    try:
        payload = (
            model_id.encode("utf-8")
            + b"\x00"
            + str(dimension).encode("ascii")
            + b"\x00"
            + text.encode("utf-8")
        )
    except UnicodeEncodeError as exc:
        raise EmbeddingCacheError("invalid_argument", "text is not valid Unicode") from exc
    return hashlib.sha256(payload).hexdigest()


def content_key(text: str, model_id: str, dimension: int, *, normalize: bool = True) -> str:
    """Return the cache key: sha256 hex of ``model_id NUL dimension NUL text``.

    When ``normalize`` is true the text is NFC-normalized and right-stripped of whitespace.

    Raises:
        EmbeddingCacheError: ``invalid_argument`` for bad types, a NUL in ``model_id``, a model id
            over 256 characters or an invalid dimension; ``input_too_large`` for text over
            1,000,000 characters.
    """
    _check_text(text, DEFAULT_MAX_TEXT_CHARS)
    _check_identity(model_id, dimension)
    return _digest(text, model_id, dimension, normalize)


def _open(target: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(target, isolation_level=None)
    except sqlite3.Error as exc:
        raise EmbeddingCacheError("schema", "cannot open database") from exc
    try:
        conn.execute(_CREATE_META)
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is not None and row[0] != SCHEMA_VERSION:
            raise EmbeddingCacheError("schema", "unsupported schema version")
        conn.execute(_CREATE_EMBEDDINGS)
        conn.execute(_CREATE_INDEX)
        if row is None:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
    except sqlite3.Error as exc:
        conn.close()
        raise EmbeddingCacheError("schema", "invalid database") from exc
    except EmbeddingCacheError:
        conn.close()
        raise
    return conn


class EmbeddingCache:
    """Disk cache of embeddings in SQLite.

    Only misses are sent to the embedder, in batches of ``batch_size``. Each batch is validated
    and written in one transaction. Source text is never stored. One process only: concurrency
    relies on SQLite file locking.
    """

    def __init__(
        self,
        path: Path | str,
        embedder: EmbedderPort,
        *,
        batch_size: int = 64,
        max_entries: int = 1_000_000,
        max_text_chars: int = 1_000_000,
        clock: Callable[[], float] = time.time,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        """Open (or create) the cache file; ``":memory:"`` keeps it in memory.

        Raises:
            EmbeddingCacheError: ``invalid_argument`` for bad arguments, ``schema`` when the file
                is not a compatible cache database.
        """
        _positive_int("batch_size", batch_size)
        _positive_int("max_entries", max_entries)
        _positive_int("max_text_chars", max_text_chars)
        try:
            model_id = embedder.model_id
            dimension = embedder.dimension
        except AttributeError as exc:
            raise EmbeddingCacheError(
                "invalid_argument",
                "embedder must define model_id and dimension",
            ) from exc
        _check_identity(model_id, dimension)
        target = str(path) if isinstance(path, Path) else path
        if not isinstance(target, str) or not target:
            raise EmbeddingCacheError("invalid_argument", "path must be a non-empty path")
        self._embedder = embedder
        self._model_id = model_id
        self._dimension = dimension
        self._fmt = f"<{dimension}d"
        self._batch_size = batch_size
        self._max_entries = max_entries
        self._max_text_chars = max_text_chars
        self._clock = clock
        self._on_event = on_event
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._corrupt = 0
        self._closed = False
        self._conn = _open(target)

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the database; calling it again is a no-op."""
        if not self._closed:
            self._closed = True
            self._conn.close()

    def _require_open(self) -> None:
        if self._closed:
            raise EmbeddingCacheError("invalid_argument", "cache is closed")

    def _check_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise EmbeddingCacheError("invalid_argument", "texts must be a sequence of strings")
        try:
            items = list(texts)
        except TypeError as exc:
            raise EmbeddingCacheError(
                "invalid_argument",
                "texts must be a sequence of strings",
            ) from exc
        for item in items:
            _check_text(item, self._max_text_chars)
        return items

    def _decode(self, blob: object) -> list[float] | None:
        if not isinstance(blob, bytes) or len(blob) != 8 * self._dimension:
            return None
        return list(struct.unpack(self._fmt, blob))

    def _lookup(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for key, blob in self._conn.execute(_SELECT_KEYS, (json.dumps(keys),)):
            vector = self._decode(blob)
            if vector is None:
                self._corrupt += 1
            else:
                found[key] = vector
        return found

    def _touch(self, keys: list[str], now: float) -> None:
        self._conn.execute(_TOUCH_KEYS, (now, json.dumps(keys)))

    def _coerce(self, row: Any) -> list[float]:
        try:
            values = list(row)
        except TypeError as exc:
            raise EmbeddingCacheError(
                "embedder_contract",
                "embedder returned a vector that is not a sequence",
            ) from exc
        if len(values) != self._dimension:
            raise EmbeddingCacheError(
                "embedder_contract",
                f"embedder returned a vector of length {len(values)}, expected {self._dimension}",
            )
        floats: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise EmbeddingCacheError(
                    "embedder_contract",
                    "embedder returned a non-numeric vector element",
                )
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise EmbeddingCacheError(
                    "embedder_contract",
                    "embedder returned an unrepresentable vector element",
                ) from exc
            if not math.isfinite(number):
                raise EmbeddingCacheError(
                    "embedder_contract",
                    "embedder returned a non-finite vector element",
                )
            floats.append(number)
        return floats

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        result: Any = self._embedder.embed(batch)
        try:
            rows = list(result)
        except TypeError as exc:
            raise EmbeddingCacheError(
                "embedder_contract",
                "embedder did not return a sequence of vectors",
            ) from exc
        if len(rows) != len(batch):
            raise EmbeddingCacheError(
                "embedder_contract",
                f"embedder returned {len(rows)} vectors for {len(batch)} texts",
            )
        return [self._coerce(row) for row in rows]

    def _store(self, keys: list[str], vectors: list[list[float]], now: float) -> None:
        rows = [
            (key, self._model_id, self._dimension, struct.pack(self._fmt, *vector), now, now)
            for key, vector in zip(keys, vectors, strict=True)
        ]
        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.executemany(_INSERT, rows)
            count = int(cursor.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            evicted = 0
            if count > self._max_entries:
                cursor.execute(_EVICT, (count - self._max_entries,))
                evicted = max(cursor.rowcount, 0)
            cursor.execute("COMMIT")
        except BaseException:
            with suppress(sqlite3.Error):
                cursor.execute("ROLLBACK")
            raise
        self._evictions += evicted

    def _emit(self, size: int, hits: int, misses: int) -> None:
        logger.debug("embed call: size=%d hits=%d misses=%d", size, hits, misses)
        if self._on_event is None:
            return
        payload: dict[str, object] = {"type": "batch", "size": size, "hits": hits, "misses": misses}
        try:
            self._on_event(payload)
        except Exception as exc:
            logger.warning("on_event callback failed: %s", type(exc).__name__)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in input order, embedding only the misses.

        Duplicate texts (same key) are embedded once. Misses go to the embedder in batches of
        ``batch_size``; each batch is validated and stored in one transaction, so an error in a
        later batch keeps the earlier ones. Counters and the ``on_event`` payload count unique
        texts.

        Raises:
            EmbeddingCacheError: ``invalid_argument``, ``input_too_large`` or
                ``embedder_contract``. Exceptions raised by the embedder propagate unchanged.
        """
        self._require_open()
        items = self._check_texts(texts)
        if not items:
            return []
        now = float(self._clock())
        keys = [_digest(text, self._model_id, self._dimension, True) for text in items]
        unique: dict[str, str] = {}
        for key, text in zip(keys, items, strict=True):
            unique.setdefault(key, text)
        found = self._lookup(list(unique))
        hit_count = len(found)
        self._hits += hit_count
        if found:
            self._touch(list(found), now)
        missing = [(key, text) for key, text in unique.items() if key not in found]
        for start in range(0, len(missing), self._batch_size):
            chunk = missing[start : start + self._batch_size]
            chunk_keys = [key for key, _ in chunk]
            vectors = self._embed_batch([text for _, text in chunk])
            self._store(chunk_keys, vectors, now)
            self._misses += len(chunk)
            found.update(zip(chunk_keys, vectors, strict=True))
        self._emit(len(unique), hit_count, len(missing))
        return [list(found[key]) for key in keys]

    def get(self, text: str) -> list[float] | None:
        """Return the cached vector for ``text`` or None; never calls the embedder."""
        self._require_open()
        _check_text(text, self._max_text_chars)
        key = _digest(text, self._model_id, self._dimension, True)
        found = self._lookup([key])
        if key in found:
            self._hits += 1
            self._touch([key], float(self._clock()))
            return list(found[key])
        self._misses += 1
        return None

    def prune(self, older_than: float) -> int:
        """Delete entries whose ``accessed_at`` is below ``older_than``; return the count."""
        self._require_open()
        if (
            isinstance(older_than, bool)
            or not isinstance(older_than, (int, float))
            or not math.isfinite(older_than)
        ):
            raise EmbeddingCacheError("invalid_argument", "older_than must be a finite number")
        cursor = self._conn.execute(
            "DELETE FROM embeddings WHERE accessed_at < ?",
            (float(older_than),),
        )
        return max(cursor.rowcount, 0)

    def stats(self) -> CacheStats:
        """Return a snapshot of the counters and the current number of entries."""
        self._require_open()
        entries = int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            entries=entries,
            evictions=self._evictions,
            corrupt=self._corrupt,
        )
