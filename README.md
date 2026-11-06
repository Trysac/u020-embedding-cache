# embedding-cache

[![PyPI version](https://img.shields.io/pypi/v/trysac-embedding-cache)](https://pypi.org/project/trysac-embedding-cache/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-embedding-cache)](LICENSE)
[![CI](https://github.com/Trysac/u020-embedding-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u020-embedding-cache/actions/workflows/ci.yml)

SQLite disk cache for embeddings keyed by content hash, model id and dimension, with batched misses, bit-exact float64 storage and LRU eviction

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-embedding-cache
# or
uv add trysac-embedding-cache
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import embedding_cache

# normalize() trims surrounding whitespace and lower-cases the text.
print(embedding_cache.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    embedding_cache.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
