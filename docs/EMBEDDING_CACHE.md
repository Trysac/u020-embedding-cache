# Embedding cache

`embedding_cache` is a dependency-free SQLite cache for embeddings. It uses only `sqlite3`,
`hashlib` and `struct` from the standard library.

## Usage

```python
from embedding_cache import EmbeddingCache


class Embedder:
    model_id = "demo"
    dimension = 2

    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


with EmbeddingCache(":memory:", Embedder()) as cache:
    print(cache.embed(["a", "bb", "a"]))
```

## Public API

- `EmbedderPort`: protocol with `model_id`, `dimension` and `embed(texts)`.
- `content_key(text, model_id, dimension, *, normalize=True)`: `sha256` hex over
  `model_id NUL dimension NUL text`. With `normalize` the text is NFC-normalized and
  right-stripped of whitespace first. `model_id` must not contain NUL and has at most 256
  characters.
- `EmbeddingCache(path, embedder, *, batch_size=64, max_entries=1_000_000,
  max_text_chars=1_000_000, clock=time.time, on_event=None)`: `embed`, `get`, `prune`,
  `stats`, `close` and context-manager support. `":memory:"` is supported.
- `CacheStats`: frozen dataclass with `hits`, `misses`, `entries`, `evictions`, `corrupt`.
- `EmbeddingCacheError` with `code` in `embedder_contract`, `schema`, `invalid_argument`,
  `input_too_large`.

## Behaviour

- Duplicate texts (same key) in one call are embedded once; only misses reach the embedder,
  in batches of `batch_size`. The embedder receives the first original text seen for a key.
- Each batch is validated (count, dimension, `math.isfinite`) and written in one transaction.
  Earlier batches stay stored if a later one fails; embedder exceptions propagate.
- Vectors are stored as little-endian float64 (`struct` format `<{dimension}d`), so values
  round-trip bit-exactly, including the sign of `-0.0`.
- A row whose BLOB length is not `8 * dimension` counts in `stats.corrupt`, is treated as a
  miss and overwritten.
- Hits refresh `accessed_at` with a single statement per call. When the row count exceeds
  `max_entries`, the rows with the oldest `accessed_at` (ties by key) are deleted.
- `hits`, `misses` and the `on_event` payload count unique texts per call. The payload is
  `{"type": "batch", "size", "hits", "misses"}` and carries no text. Exceptions raised by the
  callback are swallowed and logged by exception type only.
- Logging uses `logging.getLogger(__name__)` with a `NullHandler`; input text is never logged.

## Privacy

The file holds keys (hashes), model ids, dimensions, vectors and timestamps. The source text is
never stored, so the file can sit next to sensitive corpora. Vectors can still leak information
about the text they encode, and a hash of a guessable text can be confirmed by guessing, so
treat the file as sensitive.

## Concurrency and scope

One process only; coordination relies on SQLite file locking. Out of scope: async embedders,
multi-process write coordination, vector search and compression.

## Compared with LangChain `CacheBackedEmbeddings`

That class requires LangChain and a key-value store abstraction. This library is a single
dependency-free module.
