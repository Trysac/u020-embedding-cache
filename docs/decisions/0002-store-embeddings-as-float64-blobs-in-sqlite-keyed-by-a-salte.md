# Store embeddings as float64 BLOBs in SQLite keyed by a salted content hash

Status: accepted (generated, pending human review)

## Context

Re-embedding unchanged chunks wastes money and time. Caches keyed by text alone return vectors from the wrong model after an upgrade, and caches that keep the source text expose sensitive corpora. The library must also reject malformed embedder output and stay dependency-free.

## Decision

Use the standard-library sqlite3 module with an embeddings table keyed by sha256(model_id NUL dimension NUL normalized text), so model or dimension changes miss. Only the hash, vector and timestamps are stored. Vectors are packed as little-endian float64 for bit-exact round trips. Misses are embedded in batches, each validated (count, dimension, finiteness) and written in one transaction. Eviction is LRU by accessed_at with key tie-breaks. Lookups and hit refreshes use json_each so each is a single statement regardless of size. Errors use one base class with stable codes.

## Consequences

Zero runtime dependencies and a portable cache file that never contains source text, though vectors may still leak information about it. Float64 storage doubles size compared with float32 but guarantees exact round trips. Only a single process is supported, using SQLite file locking. json_each requires SQLite with JSON1, which current builds include.
