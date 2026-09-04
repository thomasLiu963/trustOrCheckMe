# V2 experiment namespace

The frozen V2 protocol is `EXPERIMENT_V2.md`; runtime configuration is
`config/experiment_v2.yaml`.

Zero-cost preparation:

```bash
uv run python -m src.cli prepare-v2-sample --size 200
uv run python -m src.cli audit-v1
uv run python -m src.cli import-v1-reuse --sample v2a
uv run python -m src.cli inspect-v2-prompts --sample v2a --limit 5
uv run python -m src.cli plan-v2 --size 200
uv run python -m src.cli run-v2-answers --sample v2a --dry-run
uv run python -m src.cli run-v2-confidence --sample v2a --dry-run
uv run python -m src.cli run-v2-decisions --sample v2a --dry-run
```

Paid commands require the explicit `--yes` flag. Do not run them until the
request plan, current provider pricing, prompts, and one-question smoke-test
budget have been reviewed.

V2 data is stored under `results/v2/`; paper-oriented outputs are generated
under `paper_outputs/v2/`. Neither location may overwrite V1 artifacts.
