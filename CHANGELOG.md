# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `EmbeddingCache`, `content_key`, `EmbedderPort`, `CacheStats` and `EmbeddingCacheError`: a dependency-free SQLite embedding cache with batched misses, bit-exact float64 storage, validation of embedder output and LRU eviction.
- Initial project structure from the factory utility template.
