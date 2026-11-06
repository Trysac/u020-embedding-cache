# Project record

## Objective

SQLite disk cache for embeddings keyed by content hash, model id and dimension, with batched misses, bit-exact float64 storage and LRU eviction

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested.
- Out of scope: anything that needs runtime dependencies or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 484897 / 55870 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.6053 |

## Factory run u020 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2049 | 45024 | 0 | 4246 | 0.2276 |
| fix-1 | claude-sonnet-5-5 | no | 6 | 621 | 6070 | 39223 | 0.0253 |
| fix-2 | claude-sonnet-5-5 | no | 10 | 1554 | 16164 | 87508 | 0.0647 |
| fix-3 | claude-sonnet-5-5 | no | 8 | 1062 | 15824 | 62830 | 0.0565 |
| fix-4 | claude-sonnet-5-5 | no | 8 | 804 | 8339 | 57339 | 0.0346 |
| audit | claude-sonnet-5-5 | no | 10 | 2530 | 34068 | 101079 | 0.1206 |
| docs | claude-haiku-5-5 | no | 130581 | 4275 | 0 | 0 | 0.0760 |

Total: $0.6053. Prices: Anthropic list prices.
