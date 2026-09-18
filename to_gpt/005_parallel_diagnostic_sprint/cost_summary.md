# Task 005 cost summary

## Paid calls

| lane | scientific calls | provider attempts | retries | parse repairs | estimated USD |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 0 | 0 |
| B q2 | 1000 | 1000 | 0 | 0 | 1.382600 |
| C | 0 | 0 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 0 | 0 |
| **Total** | **1000** | **1000** | **0** | **0** | **1.382600** |

## Lane B split

| model | scientific | input tokens | output tokens | estimated USD |
|---|---:|---:|---:|---:|
| GPT `gpt-5.6-sol` | 500 | 141982 | 8500 | 0.737928 |
| Claude `claude-sonnet-5` | 500 | 289826 | 6502 | 0.644672 |

Pricing from `config/models.yaml`: GPT $4/$20 per million; Claude $2/$10 per million.

## Runtime

- Lane B wall-clock: 418.9 s
- Lane A local analysis: ~78 s (no API)
- Stopped reason: none
