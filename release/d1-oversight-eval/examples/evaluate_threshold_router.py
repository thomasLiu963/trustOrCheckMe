#!/usr/bin/env python3
"""Offline example: evaluate a threshold router on frozen D1 labels.

This does not call a model. It substitutes a new *decision rule* that uses
only released columns, then scores coverage, leakage, escape, and loss
against the frozen correctness labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.constants import GPT_CLAUDE, LAMBDA_GRID
from analysis.io_csv import filter_routing, load_routing
from analysis.metrics import coverage, escape_rate, leakage, loss


def threshold_policy(displayed_confidence: float | None, lambda_cost: float) -> int:
    """VERIFY iff implied displayed risk exceeds λ. Hidden rows always USE."""
    if displayed_confidence is None:
        return 0
    return int((1.0 - displayed_confidence) > lambda_cost)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="code", choices=("mmlu", "code"))
    parser.add_argument("--model", default="openai_gpt56_sol", choices=GPT_CLAUDE)
    parser.add_argument("--condition", default="true_q_visible")
    parser.add_argument("--lambda-cost", type=float, default=0.20)
    args = parser.parse_args()

    rows = filter_routing(
        load_routing(ROOT / "data" / "item_level_routing.csv"),
        task=args.task,
        model=args.model,
        condition=args.condition,
    )
    y = [int(r.incorrect) for r in rows]
    v_obs = [int(r.verify) for r in rows]
    v_hat = [threshold_policy(r.displayed_confidence, args.lambda_cost) for r in rows]

    def summarize(verify: list[int], label: str) -> None:
        n = len(verify)
        cov = sum(verify) / n
        leak = sum((1 - v) * yi for v, yi in zip(verify, y)) / n
        err = sum(y) / n
        esc = leak / err if err else float("nan")
        print(
            f"{label:24s} coverage={cov:.3f} leakage={leak:.3f} "
            f"escape={esc:.3f} loss={loss(leak, cov, args.lambda_cost):.3f}"
        )

    print(f"{args.task} {args.model} {args.condition}  λ={args.lambda_cost}  n={len(rows)}")
    summarize(v_obs, "observed D1 router")
    summarize(v_hat, "threshold on displayed q")
    print("Other λ on the frozen grid for the threshold rule:")
    for lam in LAMBDA_GRID:
        v = [threshold_policy(r.displayed_confidence, lam) for r in rows]
        cov = sum(v) / len(v)
        leak = sum((1 - vi) * yi for vi, yi in zip(v, y)) / len(v)
        print(f"  λ={lam:<6} loss={loss(leak, cov, lam):.4f} coverage={cov:.3f} leakage={leak:.3f}")


if __name__ == "__main__":
    main()
