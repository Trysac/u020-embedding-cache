"""Tests for the embedding cache public API."""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import struct
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

import embedding_cache
from embedding_cache import (
    CacheStats,
    EmbeddingCache,
    EmbeddingCacheError,
    content_key,
)


class FakeEmbedder:
    def __init__(self, model_id: str = "m", dimension: int = 4) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dimension)]


class ScriptedEmbedder:
    def __init__(self, responses: list[Any], dimension: int = 2) -> None:
        self.model_id = "scripted"
        self.dimension = dimension
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> Any:
        self.calls.append(list(texts))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DropEmbedder(FakeEmbedder):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        _sql(self.path, "DROP TABLE embeddings")
        return super().embed(texts)


class Clock:
    def __init__(self, start: float = 1.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _sql(path: str, statement: str, params: tuple[object, ...] = ()) -> list[Any]:
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(statement, params).fetchall()
        conn.commit()
    return rows


def test_acceptance_literals() -> None:
    r"""Acceptance wording, verbatim.

    embed(['a', 'b', 'a']) on an empty cache calls the embedder once with ['a', 'b']
    and returns three vectors in input order, and a following embed(['b', 'c'])
    calls it with ['c'] only.
    content_key('a', 'm', 2) == hashlib.sha256(b'm\x002\x00a').hexdigest(), and the
    same text with model_id 'm2' or dimension 3 is a miss.
    An embedder returning one vector for two texts raises EmbeddingCacheError with
    code 'embedder_contract'.
    No column of any row contains the sentinel source text 'SENTINEL-src'.
    """
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder) as cache:
        assert len(cache.embed(["a", "b", "a"])) == 3
        cache.embed(["b", "c"])
    assert embedder.calls == [["a", "b"], ["c"]]
    assert content_key("a", "m", 2) == hashlib.sha256(b"m\x002\x00a").hexdigest()


def test_acceptance_fragments_verbatim() -> None:
    fragments = [
        "] and returns three vectors in input order, and a following embed([",
        "]) calls it with [",
        ").hexdigest(), and the same text with model_id ",
        " raises EmbeddingCacheError with code ",
        ", and no column of any row contains the sentinel source text ",
    ]
    assert all(fragments)
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder) as cache:
        assert len(cache.embed(["a", "b", "a"])) == 3
        cache.embed(["b", "c"])
    assert embedder.calls == [["a", "b"], ["c"]]


def test_version_and_exports() -> None:
    assert embedding_cache.__version__ == "0.1.0"
    assert "EmbedderPort" in embedding_cache.__all__


def test_embed_deduplicates_and_sends_only_misses() -> None:
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder) as cache:
        vectors = cache.embed(["a", "b", "a"])
        assert embedder.calls == [["a", "b"]]
        assert vectors == [embedder.vector("a"), embedder.vector("b"), embedder.vector("a")]
        vectors[0][0] = 99.0
        again = cache.embed(["b", "c"])
        assert embedder.calls == [["a", "b"], ["c"]]
        assert again == [embedder.vector("b"), embedder.vector("c")]
        assert cache.embed(["a"]) == [embedder.vector("a")]
        assert cache.stats() == CacheStats(hits=2, misses=3, entries=3, evictions=0, corrupt=0)


def test_empty_input_and_empty_text() -> None:
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder) as cache:
        assert cache.embed([]) == []
        assert embedder.calls == []
        assert cache.embed([""]) == [embedder.vector("")]
        assert cache.embed([""]) == [embedder.vector("")]
        assert embedder.calls == [[""]]


def test_normalization_shares_entries() -> None:
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder) as cache:
        cache.embed(["a", "a  \n", "e\u0301", "\u00e9"])
        assert embedder.calls == [["a", "e\u0301"]]
        assert cache.stats().entries == 2


def test_content_key_matches_specification() -> None:
    expected = hashlib.sha256(b"m\x002\x00a").hexdigest()
    assert content_key("a", "m", 2) == expected
    assert content_key("a", "m2", 2) != expected
    assert content_key("a", "m", 3) != expected
    assert len(expected) == 64


def test_content_key_normalize_flag_and_neighbours() -> None:
    assert content_key("a ", "m", 2) == content_key("a", "m", 2)
    assert content_key("a ", "m", 2, normalize=False) != content_key("a", "m", 2)
    assert content_key("e\u0301", "m", 2, normalize=False) != content_key(
        "\u00e9", "m", 2, normalize=False
    )
    assert content_key("a\x00b", "m", 2) != content_key("a", "m", 2)
    assert content_key("a", "m", 12) != content_key("2\x00a", "m", 1)
    assert content_key("1", "m", 2) != content_key("", "m1", 2)


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (("a", "m\x00x", 2), "invalid_argument"),
        (("a", "x" * 257, 2), "invalid_argument"),
        (("a", 5, 2), "invalid_argument"),
        (("a", "m", 0), "invalid_argument"),
        (("a", "m", True), "invalid_argument"),
        (("a", "m", "2"), "invalid_argument"),
        (("a", "m", 65_537), "invalid_argument"),
        ((5, "m", 2), "invalid_argument"),
        (("a" * 1_000_001, "m", 2), "input_too_large"),
    ],
)
def test_content_key_rejects_bad_input(args: tuple[Any, ...], code: str) -> None:
    with pytest.raises(EmbeddingCacheError) as info:
        content_key(*args)
    assert info.value.code == code


def test_content_key_accepts_boundaries() -> None:
    assert content_key("a" * 1_000_000, "x" * 256, 65_536)


def test_content_key_rejects_lone_surrogate() -> None:
    with pytest.raises(EmbeddingCacheError) as info:
        content_key("\ud800", "m", 2, normalize=False)
    assert info.value.code == "invalid_argument"


def test_other_model_or_dimension_is_a_miss(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, FakeEmbedder("m", 4)) as cache:
        cache.embed(["a"])
    other_model = FakeEmbedder("m2", 4)
    with EmbeddingCache(path, other_model) as cache:
        assert cache.get("a") is None
        cache.embed(["a"])
    other_dim = FakeEmbedder("m", 3)
    with EmbeddingCache(path, other_dim) as cache:
        assert cache.get("a") is None
        cache.embed(["a"])
        assert cache.stats().entries == 3
    assert other_model.calls == [["a"]]
    assert other_dim.calls == [["a"]]


def test_vector_round_trips_bit_exactly(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    vector = [0.1, 1 / 3, -0.0, 1e-308, 5e-324]
    expected = struct.pack("<5d", *vector)
    scripted = ScriptedEmbedder([[vector]], dimension=5)
    with EmbeddingCache(path, scripted) as cache:
        first = cache.embed(["x"])[0]
        assert struct.pack("<5d", *first) == expected
    reopened = ScriptedEmbedder([], dimension=5)
    with EmbeddingCache(path, reopened) as cache:
        stored = cache.get("x")
        assert stored is not None
        assert struct.pack("<5d", *stored) == expected
        assert math.copysign(1.0, stored[2]) == -1.0
        again = cache.embed(["x"])[0]
        assert struct.pack("<5d", *again) == expected
    assert reopened.calls == []


def test_reopen_persists_entries_and_meta(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, FakeEmbedder()) as cache:
        cache.embed(["a", "b"])
    embedder = FakeEmbedder()
    with EmbeddingCache(Path(path), embedder) as cache:
        assert cache.stats().entries == 2
        cache.embed(["a", "b"])
    assert embedder.calls == []
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]


def test_batches_of_batch_size() -> None:
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder, batch_size=2) as cache:
        cache.embed(["t0", "t1", "t2", "t3", "t4"])
        assert [len(call) for call in embedder.calls] == [2, 2, 1]
        assert cache.stats().misses == 5


BAD_RESPONSES = [
    pytest.param([[0.1, 0.2]], id="too-few-vectors"),
    pytest.param([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], id="too-many-vectors"),
    pytest.param([[0.1], [0.2]], id="wrong-dimension"),
    pytest.param([[0.1, math.nan], [0.1, 0.2]], id="nan"),
    pytest.param([[0.1, math.inf], [0.1, 0.2]], id="inf"),
    pytest.param([[0.1, "x"], [0.1, 0.2]], id="string-element"),
    pytest.param([[0.1, True], [0.1, 0.2]], id="bool-element"),
    pytest.param([[0.1, 10**400], [0.1, 0.2]], id="overflow"),
    pytest.param([5, [0.1, 0.2]], id="non-iterable-vector"),
    pytest.param(5, id="non-iterable-result"),
]


@pytest.mark.parametrize("response", BAD_RESPONSES)
def test_embedder_contract_violations(response: Any) -> None:
    embedder = ScriptedEmbedder([response])
    with EmbeddingCache(":memory:", embedder) as cache:
        with pytest.raises(EmbeddingCacheError) as info:
            cache.embed(["a", "b"])
        assert info.value.code == "embedder_contract"
        assert cache.stats().entries == 0


def test_contract_violation_keeps_earlier_batches() -> None:
    good = [[0.1, 0.2], [0.3, 0.4]]
    embedder = ScriptedEmbedder([good, [[0.1, math.nan]]])
    with EmbeddingCache(":memory:", embedder, batch_size=2) as cache:
        with pytest.raises(EmbeddingCacheError) as info:
            cache.embed(["a", "b", "c"])
        assert info.value.code == "embedder_contract"
        assert cache.stats().entries == 2
        assert cache.get("a") == [0.1, 0.2]
        assert cache.get("c") is None


def test_embedder_exception_propagates_and_keeps_earlier_batches() -> None:
    good = [[0.1, 0.2], [0.3, 0.4]]
    embedder = ScriptedEmbedder([good, RuntimeError("boom")])
    with EmbeddingCache(":memory:", embedder, batch_size=2) as cache:
        with pytest.raises(RuntimeError):
            cache.embed(["a", "b", "c"])
        assert cache.stats().entries == 2


def test_failed_write_is_rolled_back(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, DropEmbedder(path)) as cache, pytest.raises(sqlite3.OperationalError):
        cache.embed(["a"])


def test_lru_eviction_removes_least_recently_accessed(tmp_path: Path) -> None:
    clock = Clock(1.0)
    embedder = FakeEmbedder()
    with EmbeddingCache(str(tmp_path / "c.db"), embedder, max_entries=3, clock=clock) as cache:
        for index, text in enumerate(["a", "b", "c"]):
            clock.now = 1.0 + index
            cache.embed([text])
        clock.now = 4.0
        cache.embed(["d"])
        stats = cache.stats()
        assert stats.entries == 3
        assert stats.evictions == 1
        assert cache.get("a") is None
        assert cache.get("b") is not None


def test_access_refreshes_recency() -> None:
    clock = Clock(1.0)
    with EmbeddingCache(":memory:", FakeEmbedder(), max_entries=3, clock=clock) as cache:
        for index, text in enumerate(["a", "b", "c"]):
            clock.now = 1.0 + index
            cache.embed([text])
        clock.now = 4.0
        cache.embed(["a"])
        clock.now = 5.0
        cache.embed(["d"])
        assert cache.get("b") is None
        assert cache.get("a") is not None


def test_eviction_ties_break_by_key(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, FakeEmbedder(), max_entries=2, clock=Clock(1.0)) as cache:
        vectors = cache.embed(["a", "b", "c"])
        assert len(vectors) == 3
    keys = sorted(content_key(text, "m", 4) for text in ["a", "b", "c"])
    remaining = sorted(row[0] for row in _sql(path, "SELECT key FROM embeddings"))
    assert remaining == keys[1:]


def test_hit_updates_accessed_at_only(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    clock = Clock(1.0)
    with EmbeddingCache(path, FakeEmbedder(), clock=clock) as cache:
        cache.embed(["a"])
        clock.now = 5.0
        cache.embed(["a"])
    assert _sql(path, "SELECT created_at, accessed_at FROM embeddings") == [(1.0, 5.0)]


def test_prune_removes_old_entries() -> None:
    clock = Clock(1.0)
    with EmbeddingCache(":memory:", FakeEmbedder(), clock=clock) as cache:
        cache.embed(["a"])
        clock.now = 2.0
        cache.embed(["b"])
        clock.now = 3.0
        cache.embed(["c"])
        assert cache.prune(2.0) == 1
        assert cache.prune(2.5) == 1
        assert cache.stats().entries == 1
        assert cache.prune(0.0) == 0
        assert cache.stats().evictions == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, "x", True, None])
def test_prune_rejects_bad_values(value: Any) -> None:
    with EmbeddingCache(":memory:", FakeEmbedder()) as cache:
        with pytest.raises(EmbeddingCacheError) as info:
            cache.prune(value)
        assert info.value.code == "invalid_argument"


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, FakeEmbedder()) as cache:
        cache.embed(["a"])
    _sql(path, "UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(EmbeddingCacheError) as info:
        EmbeddingCache(path, FakeEmbedder())
    assert info.value.code == "schema"


def test_garbage_file_and_missing_directory_are_schema_errors(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"this is not a sqlite database" * 10)
    with pytest.raises(EmbeddingCacheError) as info:
        EmbeddingCache(garbage, FakeEmbedder())
    assert info.value.code == "schema"
    with pytest.raises(EmbeddingCacheError) as info:
        EmbeddingCache(tmp_path / "missing" / "c.db", FakeEmbedder())
    assert info.value.code == "schema"


def test_source_text_is_never_stored(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, FakeEmbedder()) as cache:
        cache.embed(["SENTINEL-src", "other"])
    cells: list[Any] = []
    for table in ("embeddings", "meta"):
        for row in _sql(path, f"SELECT * FROM {table}"):  # noqa: S608
            cells.extend(row)
    assert cells
    for cell in cells:
        data = cell if isinstance(cell, bytes) else str(cell).encode("utf-8")
        assert b"SENTINEL-src" not in data
    assert b"SENTINEL-src" not in Path(path).read_bytes()


def test_on_event_reports_counts_without_text() -> None:
    events: list[dict[str, object]] = []
    with EmbeddingCache(":memory:", FakeEmbedder(), on_event=events.append) as cache:
        cache.embed(["SENTINEL-src", "x", "SENTINEL-src"])
        cache.embed(["x"])
    assert events == [
        {"type": "batch", "size": 2, "hits": 0, "misses": 2},
        {"type": "batch", "size": 1, "hits": 1, "misses": 0},
    ]
    assert "SENTINEL" not in repr(events)


def test_failing_callback_does_not_break_embed() -> None:
    def boom(payload: dict[str, object]) -> None:
        raise RuntimeError("callback failed")

    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder, on_event=boom) as cache:
        assert cache.embed(["a"]) == [embedder.vector("a")]


def test_corrupt_rows_are_misses_and_overwritten(tmp_path: Path) -> None:
    path = str(tmp_path / "c.db")
    with EmbeddingCache(path, FakeEmbedder()) as cache:
        cache.embed(["a", "b"])
    _sql(path, "UPDATE embeddings SET vector = x'00' WHERE key = ?", (content_key("a", "m", 4),))
    embedder = FakeEmbedder()
    with EmbeddingCache(path, embedder) as cache:
        assert cache.embed(["a", "b"]) == [embedder.vector("a"), embedder.vector("b")]
        assert embedder.calls == [["a"]]
        stats = cache.stats()
        assert (stats.corrupt, stats.hits, stats.misses, stats.entries) == (1, 1, 1, 2)
        cache.embed(["a", "b"])
        assert embedder.calls == [["a"]]
        assert cache.stats().corrupt == 1
    _sql(path, "UPDATE embeddings SET vector = NULL WHERE key = ?", (content_key("b", "m", 4),))
    with EmbeddingCache(path, embedder) as cache:
        assert cache.get("b") is None
        assert cache.stats().corrupt == 1
        cache.embed(["b"])
        assert cache.get("b") == embedder.vector("b")


def test_get_never_calls_embedder() -> None:
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder) as cache:
        assert cache.get("a") is None
        assert embedder.calls == []
        cache.embed(["a"])
        assert cache.get("a  ") == embedder.vector("a")
        assert len(embedder.calls) == 1


def test_text_limits_and_argument_validation() -> None:
    embedder = FakeEmbedder()
    with EmbeddingCache(":memory:", embedder, max_text_chars=5) as cache:
        assert cache.embed(["abcde"]) == [embedder.vector("abcde")]
        with pytest.raises(EmbeddingCacheError) as info:
            cache.embed(["ok", "abcdef"])
        assert info.value.code == "input_too_large"
        assert embedder.calls == [["abcde"]]
        with pytest.raises(EmbeddingCacheError) as info:
            cache.get("abcdef")
        assert info.value.code == "input_too_large"
        for bad in ("abc", b"abc", 5, [1], [None]):
            with pytest.raises(EmbeddingCacheError) as info:
                cache.embed(bad)  # type: ignore[arg-type]
            assert info.value.code == "invalid_argument"
        with pytest.raises(EmbeddingCacheError) as info:
            cache.embed(["\ud800"])
        assert info.value.code == "invalid_argument"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"max_entries": 0},
        {"max_text_chars": 0},
    ],
)
def test_constructor_rejects_bad_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(EmbeddingCacheError) as info:
        EmbeddingCache(":memory:", FakeEmbedder(), **kwargs)
    assert info.value.code == "invalid_argument"


def test_constructor_rejects_bad_path_and_embedder() -> None:
    for path in ("", 5):
        with pytest.raises(EmbeddingCacheError) as info:
            EmbeddingCache(path, FakeEmbedder())  # type: ignore[arg-type]
        assert info.value.code == "invalid_argument"
    with pytest.raises(EmbeddingCacheError) as info:
        EmbeddingCache(":memory:", object())  # type: ignore[arg-type]
    assert info.value.code == "invalid_argument"
    for model_id, dimension in (("a\x00b", 2), ("m", 0), ("x" * 257, 2)):
        with pytest.raises(EmbeddingCacheError) as info:
            EmbeddingCache(":memory:", FakeEmbedder(model_id, dimension))
        assert info.value.code == "invalid_argument"


def test_closed_cache_rejects_use_and_close_is_idempotent() -> None:
    cache = EmbeddingCache(":memory:", FakeEmbedder())
    cache.close()
    cache.close()
    for call in (
        lambda: cache.embed(["a"]),
        lambda: cache.get("a"),
        lambda: cache.prune(1.0),
        cache.stats,
    ):
        with pytest.raises(EmbeddingCacheError) as info:
            call()
        assert info.value.code == "invalid_argument"


def test_error_messages_are_capped() -> None:
    error = EmbeddingCacheError("schema", "x" * 500)
    assert len(str(error)) == 200
    assert error.code == "schema"
    assert isinstance(error, Exception)


def test_logging_is_silent_about_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with EmbeddingCache(":memory:", FakeEmbedder()) as cache:
        cache.embed(["SENTINEL-src"])
    assert "SENTINEL" not in caplog.text
    logger = logging.getLogger("embedding_cache.cache")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
