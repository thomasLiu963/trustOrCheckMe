#!/usr/bin/env python3
"""Checkpoint A analysis of existing V2-B data.

Read-only with respect to experiment checkpoints. Does not call models or
modify raw records. Run from the repository root:

    python to_gpt/checkpoint_A/analyze_checkpoint_A.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.calibration import fit_isotonic  # noqa: E402
from src.v2_scoring import (  # noqa: E402
    confidence_policy_action,
    validate_factor_completeness,
)

CHECKPOINT = REPO_ROOT / "results" / "v2" / "raw" / "v2.sqlite3"
SAMPLE_PATH = REPO_ROOT / "data" / "samples" / "mmlu_pro_v2b.jsonl"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

PRIMARY_FAMILY = "v2_owner_match_v1"
ROBUSTNESS_FAMILY = "v2_owner_match_paraphrase_v1"
SEED = 20260904
N_BOOT = 5000
N_FOLDS = 5
ERROR_COSTS = (2.0, 5.0, 10.0, 20.0)
OWNERS = ("ai_system", "human")
MODELS = (
    "anthropic_sonnet5",
    "google_gemini38_flash",
    "openai_gpt56_sol",
    "xai_grok420_nonreasoning",
)
MODEL_LABEL = {
    "anthropic_sonnet5": "Claude Sonnet 5",
    "google_gemini38_flash": "Gemini 3.8 Flash",
    "openai_gpt56_sol": "GPT-5.6 Sol",
    "xai_grok420_nonreasoning": "Grok 4.20",
}
MODEL_SHORT = {
    "anthropic_sonnet5": "Claude",
    "google_gemini38_flash": "Gemini",
    "openai_gpt56_sol": "GPT",
    "xai_grok420_nonreasoning": "Grok",
}
MODEL_COLOR = {
    "anthropic_sonnet5": "#D55E00",
    "google_gemini38_flash": "#009E73",
    "openai_gpt56_sol": "#0072B2",
    "xai_grok420_nonreasoning": "#CC79A7",
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Any) -> float:
    usable = [number for value in values if (number := _finite(value)) is not None]
    return float(np.mean(usable)) if usable else float("nan")


def _print(*args: Any) -> None:
    print(*args, flush=True)


def logistic_model() -> LogisticRegression:
    # sklearn 1.8+: C=np.inf is the unregularized (former penalty=None) setting.
    return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)


def _percentile_ci(samples: list[float], level: float = 0.95) -> tuple[float, float]:
    usable = sorted(x for x in samples if math.isfinite(x))
    if not usable:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    arr = np.asarray(usable, dtype=float)
    lo = float(np.quantile(arr, alpha, method="linear"))
    hi = float(np.quantile(arr, 1.0 - alpha, method="linear"))
    return lo, hi


def cell_seed(*parts: Any) -> int:
    material = "\0".join(str(p) for p in (SEED, *parts)).encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16) % (2**31 - 1)


def load_json_records(stage: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(CHECKPOINT)
    try:
        rows = connection.execute(
            """
            SELECT record_json FROM requests
            WHERE stage = ? AND status = 'success' AND record_json IS NOT NULL
            ORDER BY example_id, model_alias, request_key
            """,
            (stage,),
        ).fetchall()
        run_counts = connection.execute(
            """
            SELECT run_id, stage, status, COUNT(*)
            FROM requests
            GROUP BY run_id, stage, status
            ORDER BY run_id, stage, status
            """
        ).fetchall()
        status_counts = connection.execute(
            """
            SELECT stage, status, COUNT(*)
            FROM requests
            GROUP BY stage, status
            ORDER BY stage, status
            """
        ).fetchall()
    finally:
        connection.close()
    records = [json.loads(row[0]) for row in rows]
    return records, run_counts, status_counts


def load_v2b_ids() -> list[str]:
    ids = []
    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(json.loads(line)["example_id"])
    return ids


def realized_cost(verified: bool, is_correct: bool, error_cost: float) -> float:
    if verified:
        return 1.0
    return 0.0 if is_correct else float(error_cost)


def fractional_verify_weights(q: np.ndarray, n_verify: float) -> np.ndarray:
    """Expected inclusion weights verifying the lowest-q items, fractionally at ties.

    All items with q strictly below the cutoff receive weight 1. Items tied at
    the cutoff receive the unique weight that makes the weights sum to n_verify.
    Items above the cutoff receive weight 0.
    """
    n = len(q)
    weights = np.zeros(n, dtype=float)
    if n == 0 or n_verify <= 0:
        return weights
    if n_verify >= n:
        weights[:] = 1.0
        return weights
    order = np.argsort(q, kind="mergesort")
    remaining = float(n_verify)
    i = 0
    qs = q[order]
    while remaining > 1e-12 and i < n:
        j = i + 1
        while j < n and qs[j] == qs[i]:
            j += 1
        group = order[i:j]
        take = min(remaining, float(len(group)))
        weights[group] = take / float(len(group))
        remaining -= take
        i = j
    return weights


def rates_from_weights(
    weights: np.ndarray, wrong: np.ndarray
) -> dict[str, float]:
    n = len(weights)
    n_wrong = float(wrong.sum())
    n_correct = float((~wrong).sum())
    expected_verify = float(weights.sum())
    expected_wrong_v = float((weights * wrong).sum())
    expected_correct_v = float((weights * (~wrong)).sum())
    catch = expected_wrong_v / n_wrong if n_wrong else float("nan")
    correct_v = expected_correct_v / n_correct if n_correct else float("nan")
    precision = expected_wrong_v / expected_verify if expected_verify > 1e-12 else float("nan")
    return {
        "verify_rate": expected_verify / n if n else float("nan"),
        "error_catch_rate": catch,
        "correct_verify_rate": correct_v,
        "precision": precision,
        "wrong_unverified_rate": (1.0 - catch) if math.isfinite(catch) else float("nan"),
        "expected_n_verify": expected_verify,
    }


def hidden_rates(verified: np.ndarray, wrong: np.ndarray) -> dict[str, float]:
    weights = verified.astype(float)
    return rates_from_weights(weights, wrong)


def confidence_rates_at_budget(
    q: np.ndarray, wrong: np.ndarray, n_verify: float
) -> dict[str, float]:
    return rates_from_weights(fractional_verify_weights(q, n_verify), wrong)


def catch_curve(q: np.ndarray, wrong: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage (fraction verified) vs expected error-catch rate, with tie interpolation."""
    n = len(q)
    n_wrong = float(wrong.sum())
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(q, kind="mergesort")
    qs = q[order]
    ws = wrong[order]
    cover = [0.0]
    catch = [0.0]
    caught = 0.0
    seen = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and qs[j] == qs[i]:
            j += 1
        caught += float(ws[i:j].sum())
        seen += j - i
        cover.append(seen / n)
        catch.append(caught / n_wrong if n_wrong else float("nan"))
        i = j
    return np.asarray(cover), np.asarray(catch)


def error_catch_fractional(q: np.ndarray, wrong: np.ndarray, n_verify: float) -> float:
    """Expected fraction of wrong items verified under the fractional cutoff policy."""
    n = len(q)
    n_wrong = float(wrong.sum())
    if n_wrong == 0:
        return float("nan")
    if n_verify <= 0:
        return 0.0
    if n_verify >= n:
        return 1.0
    order = np.argsort(q, kind="mergesort")
    qs = q[order]
    ws = wrong[order]
    remaining = float(n_verify)
    caught = 0.0
    i = 0
    while remaining > 1e-12 and i < n:
        j = i + 1
        while j < n and qs[j] == qs[i]:
            j += 1
        group_n = j - i
        take = min(remaining, float(group_n))
        caught += take / group_n * float(ws[i:j].sum())
        remaining -= take
        i = j
    return caught / n_wrong


def hidden_error_catch(verified: np.ndarray, wrong: np.ndarray) -> float:
    n_wrong = float(wrong.sum())
    if n_wrong == 0:
        return float("nan")
    return float((verified & wrong).sum()) / n_wrong


def bootstrap_catch_difference(
    q: np.ndarray,
    wrong: np.ndarray,
    hidden: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int,
) -> dict[str, float]:
    n = len(q)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    n_valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w_b = wrong[idx]
        if not w_b.any():
            continue
        h_b = hidden[idx]
        hid = hidden_error_catch(h_b, w_b)
        conf = error_catch_fractional(q[idx], w_b, float(h_b.sum()))
        diffs[n_valid] = hid - conf
        n_valid += 1
    usable = diffs[:n_valid].tolist()
    lo, hi = _percentile_ci(usable)
    return {
        "catch_diff": float(np.mean(diffs[:n_valid])) if n_valid else float("nan"),
        "catch_diff_ci_lower": lo,
        "catch_diff_ci_upper": hi,
        "n_valid_resamples": float(n_valid),
    }


def bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int, seed: int, question_ids: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Question-level bootstrap of a mean. If question_ids given, cluster first."""
    if question_ids is None:
        units = values.astype(float)
    else:
        series = pd.Series(values.astype(float), index=question_ids)
        units = series.groupby(level=0).mean().to_numpy()
    n = len(units)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(units))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = units[idx].mean(axis=1)
    lo, hi = _percentile_ci(samples.tolist())
    return estimate, lo, hi


def bootstrap_paired_mean_diff(
    left: np.ndarray, right: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    n = len(left)
    diff = left.astype(float) - right.astype(float)
    estimate = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = diff[idx].mean(axis=1)
    lo, hi = _percentile_ci(samples.tolist())
    return estimate, lo, hi


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if not y.any():
        return float("nan")
    try:
        return float(average_precision_score(y, scores))
    except ValueError:
        return float("nan")


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    if len(np.unique(y)) < 2:
        return float(log_loss(y, p, labels=[0, 1]))
    return float(log_loss(y, p))


def fit_platt(q_train: np.ndarray, y_train: np.ndarray, q_test: np.ndarray) -> np.ndarray:
    """Logistic calibration of correctness probability from raw q."""
    if len(np.unique(y_train)) < 2 or len(np.unique(q_train)) < 2:
        rate = float(np.mean(y_train))
        return np.full(len(q_test), rate, dtype=float)
    clf = logistic_model()
    clf.fit(q_train.reshape(-1, 1), y_train.astype(int))
    return clf.predict_proba(q_test.reshape(-1, 1))[:, 1]


def fit_logistic_oos(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
) -> np.ndarray:
    """Out-of-sample predicted P(y=1) from unregularized logistic regression."""
    n = len(y)
    pred = np.full(n, float("nan"))
    for test_idx in folds:
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_idx = np.flatnonzero(~test_mask)
        y_train = y[train_idx]
        rate = float(np.mean(y_train))
        if len(np.unique(y_train)) < 2:
            pred[test_idx] = rate
            continue
        x_train = X[train_idx]
        # Drop near-constant columns to avoid singular fits.
        usable = []
        for col in range(x_train.shape[1]):
            if np.nanstd(x_train[:, col]) > 1e-12:
                usable.append(col)
        if not usable:
            pred[test_idx] = rate
            continue
        clf = logistic_model()
        clf.fit(x_train[:, usable], y_train.astype(int))
        pred[test_idx] = clf.predict_proba(X[test_idx][:, usable])[:, 1]
    return pred


def make_stratified_folds(
    y: np.ndarray, n_splits: int = N_FOLDS, seed: int = SEED
) -> list[np.ndarray]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(y), 1))
    return [test_idx.copy() for _, test_idx in splitter.split(dummy, y.astype(int))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
        }
    )
    return plt


def validate_and_load() -> tuple[pd.DataFrame, dict[str, Any]]:
    print("Loading checkpoint:", CHECKPOINT, flush=True)
    answers, run_counts, status_counts = load_json_records("answer")
    confidences, _, _ = load_json_records("confidence")
    decisions, _, _ = load_json_records("verification")
    sample_ids = load_v2b_ids()

    answer_keys = [(r["example_id"], r["model_id"]) for r in answers]
    conf_keys = [(r["example_id"], r["model_id"]) for r in confidences]
    ans_map = {(r["example_id"], r["model_id"]): r for r in answers}
    conf_map = {(r["example_id"], r["model_id"]): r for r in confidences}

    primary = [r for r in decisions if r.get("prompt_family") == PRIMARY_FAMILY]
    robustness = [r for r in decisions if r.get("prompt_family") == ROBUSTNESS_FAMILY]
    other_family = [
        r
        for r in decisions
        if r.get("prompt_family") not in {PRIMARY_FAMILY, ROBUSTNESS_FAMILY}
    ]

    cell_keys = [
        (
            r["example_id"],
            r["model_id"],
            r["decision_owner"],
            r["confidence_visibility"],
            float(r["error_cost"]),
        )
        for r in primary
    ]
    completeness = validate_factor_completeness(primary)

    frozen_answer_mismatch = 0
    q_mismatch = 0
    missing_answer = 0
    missing_conf = 0
    joined_rows = []
    for decision in primary:
        key = (decision["example_id"], decision["model_id"])
        answer = ans_map.get(key)
        conf = conf_map.get(key)
        if answer is None:
            missing_answer += 1
            continue
        if conf is None:
            missing_conf += 1
            continue
        if (
            decision["frozen_answer_label"] != answer["answer_label"]
            or decision["frozen_answer_label"] != conf["frozen_answer_label"]
        ):
            frozen_answer_mismatch += 1
        if decision["probability_correct"] != conf["probability_correct"]:
            q_mismatch += 1
        joined_rows.append(
            {
                "example_id": decision["example_id"],
                "model_id": decision["model_id"],
                "decision_owner": decision["decision_owner"],
                "confidence_visibility": decision["confidence_visibility"],
                "error_cost": float(decision["error_cost"]),
                "verification_cost": float(decision["verification_cost"]),
                "action": decision["action"],
                "verified": decision["action"] == "VERIFY_FIRST",
                "probability_correct": float(decision["probability_correct"]),
                "frozen_answer_label": decision["frozen_answer_label"],
                "is_correct": bool(answer["is_correct"]),
                "wrong": not bool(answer["is_correct"]),
                "category": answer.get("category"),
                "prompt_family": decision["prompt_family"],
                "run_id": decision["run_id"],
                "answer_run_id": answer["run_id"],
                "confidence_run_id": conf["run_id"],
                "request_key": decision["request_key"],
            }
        )

    df = pd.DataFrame(joined_rows)
    df["raw_threshold_verify"] = [
        confidence_policy_action(q, L, 1.0) == "VERIFY_FIRST"
        for q, L in zip(df["probability_correct"], df["error_cost"])
    ]
    df["realized_cost"] = [
        realized_cost(v, c, L)
        for v, c, L in zip(df["verified"], df["is_correct"], df["error_cost"])
    ]

    # Frozen Stage-3 field consistency across the 16 cells.
    n_inconsistent_answer = 0
    n_inconsistent_q = 0
    for _, group in df.groupby(["example_id", "model_id"]):
        if group["frozen_answer_label"].nunique() != 1:
            n_inconsistent_answer += 1
        if group["probability_correct"].nunique() != 1:
            n_inconsistent_q += 1

    unique_q_by_model = {
        model: sorted(
            df.loc[df["model_id"] == model, "probability_correct"].drop_duplicates()
        )
        for model in MODELS
    }
    q_value_counts = {
        model: Counter(
            df.loc[df["model_id"] == model]
            .drop_duplicates(["example_id"])["probability_correct"]
            .tolist()
        )
        for model in MODELS
    }

    answer_ids = {r["example_id"] for r in answers}
    sample_set = set(sample_ids)
    validation = {
        "checkpoint": str(CHECKPOINT),
        "sample_file": str(SAMPLE_PATH),
        "status_counts": [list(row) for row in status_counts],
        "run_counts": [list(row) for row in run_counts],
        "n_answer_records": len(answers),
        "n_confidence_records": len(confidences),
        "n_verification_records_all": len(decisions),
        "n_primary_stage3": len(primary),
        "n_robustness_stage3": len(robustness),
        "n_other_prompt_family": len(other_family),
        "n_unique_answer_keys": len(set(answer_keys)),
        "n_duplicate_answer_keys": len(answer_keys) - len(set(answer_keys)),
        "n_unique_confidence_keys": len(set(conf_keys)),
        "n_duplicate_confidence_keys": len(conf_keys) - len(set(conf_keys)),
        "n_unique_primary_cells": len(set(cell_keys)),
        "n_duplicate_primary_cells": len(cell_keys) - len(set(cell_keys)),
        "models": list(MODELS),
        "n_models": len(MODELS),
        "n_questions": len(sample_ids),
        "questions_per_model": {
            model: int(df.loc[df["model_id"] == model, "example_id"].nunique())
            for model in MODELS
        },
        "error_costs": list(ERROR_COSTS),
        "owners": list(OWNERS),
        "visibility": ["hidden", "visible"],
        "prompt_family_analyzed": PRIMARY_FAMILY,
        "prompt_family_excluded": ROBUSTNESS_FAMILY,
        "factor_completeness_issue_count": len(completeness),
        "factor_completeness_issues": completeness[:20],
        "missing_answer_joins": missing_answer,
        "missing_confidence_joins": missing_conf,
        "stage3_vs_stage1_answer_mismatches": frozen_answer_mismatch,
        "stage3_vs_stage2_confidence_mismatches": q_mismatch,
        "pairs_with_inconsistent_frozen_answer_across_cells": n_inconsistent_answer,
        "pairs_with_inconsistent_confidence_across_cells": n_inconsistent_q,
        "answer_ids_match_v2b_sample": answer_ids == sample_set,
        "n_answer_ids_not_in_sample": len(answer_ids - sample_set),
        "n_sample_ids_without_answers": len(sample_set - answer_ids),
        "n_refused_primary": int(sum(1 for r in primary if r.get("refused"))),
        "verification_cost_values": sorted(set(df["verification_cost"])),
        "actions": dict(Counter(df["action"])),
        "primary_run_ids": dict(Counter(df["run_id"])),
        "answer_run_ids": dict(Counter(r["run_id"] for r in answers)),
        "n_unique_confidence_values": {
            model: len(vals) for model, vals in unique_q_by_model.items()
        },
        "most_common_confidence_values": {
            model: Counter(counts).most_common(8)
            for model, counts in q_value_counts.items()
        },
        "notes": [
            "Primary analysis uses prompt family v2_owner_match_v1 only.",
            "6400 paraphrase-robustness Stage-3 rows are excluded from Checkpoint A.",
            "One overlapping V2-A cell was completed during v2b-core (19201 V2-B "
            "run_id rows rather than 19200). The scientific cell grid is complete "
            "and unique; this is a run-id bookkeeping detail, not a missing/extra cell.",
            "Frozen Stage-1 answers and Stage-2 confidences are unique on "
            "(example_id, model_id) and are copied consistently into every Stage-3 cell.",
        ],
    }
    write_json(OUT_DIR / "data_validation.json", validation)
    print(
        f"Primary Stage-3 rows: {len(df):,} | models: {df['model_id'].nunique()} | "
        f"questions: {df['example_id'].nunique()} | completeness issues: {len(completeness)}",
        flush=True,
    )
    return df, validation


def question_frame(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per question for a model, with frozen answer/confidence."""
    sub = df.loc[df["model_id"] == model]
    base = (
        sub.drop_duplicates("example_id")
        .sort_values("example_id")
        .loc[:, ["example_id", "is_correct", "wrong", "probability_correct", "category"]]
        .reset_index(drop=True)
    )
    for owner in OWNERS:
        for vis in ("hidden", "visible"):
            for L in ERROR_COSTS:
                mask = (
                    (sub["decision_owner"] == owner)
                    & (sub["confidence_visibility"] == vis)
                    & (sub["error_cost"] == L)
                )
                piece = sub.loc[mask, ["example_id", "verified", "realized_cost"]].rename(
                    columns={
                        "verified": f"{vis}_{owner}_{int(L)}_verified",
                        "realized_cost": f"{vis}_{owner}_{int(L)}_cost",
                    }
                )
                base = base.merge(piece, on="example_id", how="left")
    return base


def part1_equal_budget(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    print("Part 1: equal-budget comparison", flush=True)
    cell_rows: list[dict[str, Any]] = []
    qframes: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        qf = question_frame(df, model)
        qframes[model] = qf
        q = qf["probability_correct"].to_numpy(dtype=float)
        wrong = qf["wrong"].to_numpy(dtype=bool)
        n = len(qf)
        n_wrong = int(wrong.sum())
        n_correct = n - n_wrong
        print(f"  {model}: n={n} wrong={n_wrong}", flush=True)
        for owner in OWNERS:
            for L in ERROR_COSTS:
                hid = qf[f"hidden_{owner}_{int(L)}_verified"].to_numpy(dtype=bool)
                vis = qf[f"visible_{owner}_{int(L)}_verified"].to_numpy(dtype=bool)
                hid_cost = qf[f"hidden_{owner}_{int(L)}_cost"].to_numpy(dtype=float)
                vis_cost = qf[f"visible_{owner}_{int(L)}_cost"].to_numpy(dtype=float)
                hid_rates = hidden_rates(hid, wrong)
                conf_rates = confidence_rates_at_budget(q, wrong, float(hid.sum()))
                vis_rates = hidden_rates(vis, wrong)
                boot = bootstrap_catch_difference(
                    q, wrong, hid, seed=cell_seed("catch", model, owner, L)
                )
                # Point estimate from the original sample, CI from bootstrap.
                catch_diff = (
                    hid_rates["error_catch_rate"] - conf_rates["error_catch_rate"]
                )
                cost_diff, cost_lo, cost_hi = bootstrap_paired_mean_diff(
                    vis_cost,
                    hid_cost,
                    n_boot=N_BOOT,
                    seed=cell_seed("cost", model, owner, L),
                )
                raw_thr = np.array(
                    [
                        confidence_policy_action(qi, L, 1.0) == "VERIFY_FIRST"
                        for qi in q
                    ],
                    dtype=bool,
                )
                cell_rows.append(
                    {
                        "model_id": model,
                        "model_label": MODEL_LABEL[model],
                        "decision_owner": owner,
                        "error_cost": L,
                        "n": n,
                        "n_wrong": n_wrong,
                        "n_correct": n_correct,
                        "hidden_verify_budget": hid_rates["verify_rate"],
                        "hidden_n_verify": int(hid.sum()),
                        "hidden_error_catch_rate": hid_rates["error_catch_rate"],
                        "hidden_correct_verify_rate": hid_rates["correct_verify_rate"],
                        "hidden_precision": hid_rates["precision"],
                        "hidden_wrong_unverified_rate": hid_rates["wrong_unverified_rate"],
                        "conf_error_catch_rate": conf_rates["error_catch_rate"],
                        "conf_correct_verify_rate": conf_rates["correct_verify_rate"],
                        "conf_precision": conf_rates["precision"],
                        "catch_diff_hidden_minus_conf": catch_diff,
                        "catch_diff_ci_lower": boot["catch_diff_ci_lower"],
                        "catch_diff_ci_upper": boot["catch_diff_ci_upper"],
                        "catch_diff_n_valid_resamples": boot["n_valid_resamples"],
                        "visible_verify_rate": vis_rates["verify_rate"],
                        "visible_error_catch_rate": vis_rates["error_catch_rate"],
                        "visible_wrong_unverified_rate": vis_rates["wrong_unverified_rate"],
                        "hidden_mean_cost": float(np.mean(hid_cost)),
                        "visible_mean_cost": float(np.mean(vis_cost)),
                        "cost_diff_visible_minus_hidden": cost_diff,
                        "cost_diff_ci_lower": cost_lo,
                        "cost_diff_ci_upper": cost_hi,
                        "wrong_unverified_diff_visible_minus_hidden": (
                            vis_rates["wrong_unverified_rate"]
                            - hid_rates["wrong_unverified_rate"]
                        ),
                        "verify_rate_diff_visible_minus_hidden": (
                            vis_rates["verify_rate"] - hid_rates["verify_rate"]
                        ),
                        "visible_vs_raw_threshold_agreement": float(
                            np.mean(vis == raw_thr)
                        ),
                        "hidden_vs_raw_threshold_agreement": float(
                            np.mean(hid == raw_thr)
                        ),
                        "raw_threshold_verify_rate": float(np.mean(raw_thr)),
                    }
                )
    return cell_rows, qframes


def pooled_summaries(
    cell_rows: list[dict[str, Any]], qframes: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    """Model-level and authority-separated averages with question-level bootstrap.

    Cells that share questions are not treated as independent samples. Each
    bootstrap replicate resamples questions, recomputes every cell, then
    averages the cell-level catch differences.
    """
    print("Part 1 pooled summaries", flush=True)
    summaries: list[dict[str, Any]] = []

    def summarize(model: str, owners: tuple[str, ...], tag: str) -> dict[str, Any]:
        qf = qframes[model]
        q = qf["probability_correct"].to_numpy(dtype=float)
        wrong = qf["wrong"].to_numpy(dtype=bool)
        n = len(qf)
        cells = [
            (owner, L)
            for owner in owners
            for L in ERROR_COSTS
        ]

        hidden_arrays = {
            (owner, L): qf[f"hidden_{owner}_{int(L)}_verified"].to_numpy(dtype=bool)
            for owner in owners
            for L in ERROR_COSTS
        }

        def cell_diffs(idx: np.ndarray) -> list[float]:
            q_b, w_b = q[idx], wrong[idx]
            diffs = []
            for owner, L in cells:
                hid = hidden_arrays[(owner, L)][idx]
                hid_catch = hidden_error_catch(hid, w_b)
                conf_catch = error_catch_fractional(q_b, w_b, float(hid.sum()))
                if math.isfinite(hid_catch) and math.isfinite(conf_catch):
                    diffs.append(hid_catch - conf_catch)
            return diffs

        observed = cell_diffs(np.arange(n))
        estimate = _mean(observed)
        rng = np.random.default_rng(cell_seed("pooled", model, tag))
        samples = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            diffs = cell_diffs(idx)
            if diffs:
                samples.append(_mean(diffs))
        lo, hi = _percentile_ci(samples)
        subset = [
            row
            for row in cell_rows
            if row["model_id"] == model and row["decision_owner"] in owners
        ]
        return {
            "model_id": model,
            "model_label": MODEL_LABEL[model],
            "scope": tag,
            "n_questions": n,
            "n_wrong": int(wrong.sum()),
            "n_cells_averaged": len(cells),
            "mean_hidden_verify_budget": _mean(r["hidden_verify_budget"] for r in subset),
            "mean_hidden_error_catch_rate": _mean(
                r["hidden_error_catch_rate"] for r in subset
            ),
            "mean_conf_error_catch_rate": _mean(
                r["conf_error_catch_rate"] for r in subset
            ),
            "mean_catch_diff": estimate,
            "catch_diff_ci_lower": lo,
            "catch_diff_ci_upper": hi,
            "n_cells_with_ci_excluding_zero": sum(
                1
                for r in subset
                if r["catch_diff_ci_lower"] > 0 or r["catch_diff_ci_upper"] < 0
            ),
            "n_cells_positive_diff": sum(
                1 for r in subset if r["catch_diff_hidden_minus_conf"] > 0
            ),
            "n_cells_negative_diff": sum(
                1 for r in subset if r["catch_diff_hidden_minus_conf"] < 0
            ),
            "accuracy": float(np.mean(~wrong)),
        }

    for model in MODELS:
        summaries.append(summarize(model, OWNERS, "model_pooled_over_owner_and_L"))
        for owner in OWNERS:
            summaries.append(summarize(model, (owner,), f"authority_{owner}"))
    return summaries


def add_confidence_quality(summaries: list[dict[str, Any]], qframes: dict[str, pd.DataFrame]) -> None:
    by_model = {
        row["model_id"]: row
        for row in summaries
        if row["scope"] == "model_pooled_over_owner_and_L"
    }
    for model, qf in qframes.items():
        y_correct = qf["is_correct"].to_numpy(dtype=int)
        y_wrong = qf["wrong"].to_numpy(dtype=int)
        q = qf["probability_correct"].to_numpy(dtype=float)
        row = by_model[model]
        row["brier_correctness"] = float(brier_score_loss(y_correct, q))
        row["auroc_correctness"] = safe_auroc(y_correct, q)
        row["auroc_wrong_using_one_minus_q"] = safe_auroc(y_wrong, 1.0 - q)
        row["auprc_wrong_using_one_minus_q"] = safe_auprc(y_wrong, 1.0 - q)
        row["mean_q_correct"] = float(np.mean(q[y_correct.astype(bool)])) if y_correct.any() else float("nan")
        row["mean_q_wrong"] = float(np.mean(q[y_wrong.astype(bool)])) if y_wrong.any() else float("nan")


def part2_calibration(
    qframes: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    print("Part 2: cross-fitted calibration", flush=True)
    calibrated_frames: dict[str, pd.DataFrame] = {}
    results: list[dict[str, Any]] = []
    for model in MODELS:
        qf = qframes[model].copy()
        q = qf["probability_correct"].to_numpy(dtype=float)
        y = qf["is_correct"].to_numpy(dtype=int)
        wrong = qf["wrong"].to_numpy(dtype=bool)
        folds = make_stratified_folds(y, n_splits=N_FOLDS, seed=cell_seed("folds", model))
        p_iso = np.full(len(q), float("nan"))
        p_platt = np.full(len(q), float("nan"))
        fold_id = np.full(len(q), -1, dtype=int)
        for fold_number, test_idx in enumerate(folds):
            test_mask = np.zeros(len(q), dtype=bool)
            test_mask[test_idx] = True
            train_idx = np.flatnonzero(~test_mask)
            fold_id[test_idx] = fold_number
            iso = fit_isotonic(q[train_idx].tolist(), y[train_idx].tolist())
            p_iso[test_idx] = np.array(
                [iso.predict_one(float(v)) for v in q[test_idx]], dtype=float
            )
            p_platt[test_idx] = fit_platt(q[train_idx], y[train_idx], q[test_idx])
        qf["p_isotonic"] = p_iso
        qf["p_platt"] = p_platt
        qf["cv_fold"] = fold_id
        calibrated_frames[model] = qf

        def policy_metrics(
            verify: np.ndarray, L: float
        ) -> dict[str, float]:
            costs = np.array(
                [realized_cost(bool(v), (not w), L) for v, w in zip(verify, wrong)],
                dtype=float,
            )
            rates = hidden_rates(verify.astype(bool), wrong)
            mean_cost, lo, hi = bootstrap_mean_ci(
                costs, n_boot=N_BOOT, seed=cell_seed("calcost", model, L, float(verify.mean()))
            )
            return {
                "verification_rate": rates["verify_rate"],
                "wrong_unverified_rate": rates["wrong_unverified_rate"],
                "correct_verify_rate": rates["correct_verify_rate"],
                "error_catch_rate": rates["error_catch_rate"],
                "mean_realized_cost": float(np.mean(costs)),
                "mean_cost_ci_lower": lo,
                "mean_cost_ci_upper": hi,
            }

        for L in ERROR_COSTS:
            raw_v = np.array(
                [confidence_policy_action(qi, L, 1.0) == "VERIFY_FIRST" for qi in q],
                dtype=bool,
            )
            iso_v = np.array(
                [confidence_policy_action(pi, L, 1.0) == "VERIFY_FIRST" for pi in p_iso],
                dtype=bool,
            )
            platt_v = np.array(
                [
                    confidence_policy_action(pi, L, 1.0) == "VERIFY_FIRST"
                    for pi in p_platt
                ],
                dtype=bool,
            )
            policies = {
                "raw_confidence_threshold": raw_v,
                "isotonic_calibrated_threshold": iso_v,
                "platt_calibrated_threshold": platt_v,
            }
            for owner in OWNERS:
                policies[f"hidden_{owner}"] = qf[
                    f"hidden_{owner}_{int(L)}_verified"
                ].to_numpy(dtype=bool)
                policies[f"visible_{owner}"] = qf[
                    f"visible_{owner}_{int(L)}_verified"
                ].to_numpy(dtype=bool)
            # Pooled-owner behavioral policies: verify if either owner verifies?
            # No: average metrics across owners by stacking would double-count.
            # Report the mean of the two owner-specific policies as pooled cost.
            for name, verify in policies.items():
                metrics = policy_metrics(verify, L)
                owner = "na"
                visibility = "na"
                kind = name
                if name.startswith("hidden_"):
                    owner = name.split("_", 1)[1]
                    visibility = "hidden"
                    kind = "actual_hidden"
                elif name.startswith("visible_"):
                    owner = name.split("_", 1)[1]
                    visibility = "visible"
                    kind = "actual_visible"
                results.append(
                    {
                        "model_id": model,
                        "model_label": MODEL_LABEL[model],
                        "error_cost": L,
                        "decision_owner": owner,
                        "policy": kind if kind.startswith("actual") else name,
                        "policy_detail": name,
                        "visibility": visibility,
                        **metrics,
                        "n": len(qf),
                        "n_wrong": int(wrong.sum()),
                    }
                )
            # Owner-pooled actual policies: per-question mean of the two owners.
            for vis in ("hidden", "visible"):
                v_ai = qf[f"{vis}_ai_system_{int(L)}_verified"].to_numpy(dtype=bool)
                v_hu = qf[f"{vis}_human_{int(L)}_verified"].to_numpy(dtype=bool)
                # Mean cost across owners, not a single fused action.
                cost_ai = np.array(
                    [realized_cost(bool(v), (not w), L) for v, w in zip(v_ai, wrong)]
                )
                cost_hu = np.array(
                    [realized_cost(bool(v), (not w), L) for v, w in zip(v_hu, wrong)]
                )
                mean_cost = 0.5 * (cost_ai + cost_hu)
                est, lo, hi = bootstrap_mean_ci(
                    mean_cost, n_boot=N_BOOT, seed=cell_seed("poolcost", model, vis, L)
                )
                results.append(
                    {
                        "model_id": model,
                        "model_label": MODEL_LABEL[model],
                        "error_cost": L,
                        "decision_owner": "pooled_owners",
                        "policy": f"actual_{vis}",
                        "policy_detail": f"{vis}_mean_of_owners",
                        "visibility": vis,
                        "verification_rate": 0.5 * (v_ai.mean() + v_hu.mean()),
                        "wrong_unverified_rate": 0.5
                        * (
                            hidden_rates(v_ai, wrong)["wrong_unverified_rate"]
                            + hidden_rates(v_hu, wrong)["wrong_unverified_rate"]
                        ),
                        "correct_verify_rate": 0.5
                        * (
                            hidden_rates(v_ai, wrong)["correct_verify_rate"]
                            + hidden_rates(v_hu, wrong)["correct_verify_rate"]
                        ),
                        "error_catch_rate": 0.5
                        * (
                            hidden_rates(v_ai, wrong)["error_catch_rate"]
                            + hidden_rates(v_hu, wrong)["error_catch_rate"]
                        ),
                        "mean_realized_cost": float(np.mean(mean_cost)),
                        "mean_cost_ci_lower": lo,
                        "mean_cost_ci_upper": hi,
                        "n": len(qf),
                        "n_wrong": int(wrong.sum()),
                    }
                )

        # Discrimination vs calibration diagnostics (model-level, not per L).
        results.append(
            {
                "model_id": model,
                "model_label": MODEL_LABEL[model],
                "error_cost": "",
                "decision_owner": "na",
                "policy": "score_quality",
                "policy_detail": "raw_q",
                "visibility": "na",
                "brier": float(brier_score_loss(y, q)),
                "auroc": safe_auroc(y, q),
                "ece_10bin": expected_calibration_error(q, y),
                "n": len(qf),
                "n_wrong": int(wrong.sum()),
            }
        )
        results.append(
            {
                "model_id": model,
                "model_label": MODEL_LABEL[model],
                "error_cost": "",
                "decision_owner": "na",
                "policy": "score_quality",
                "policy_detail": "isotonic_cv",
                "visibility": "na",
                "brier": float(brier_score_loss(y, p_iso)),
                "auroc": safe_auroc(y, p_iso),
                "ece_10bin": expected_calibration_error(p_iso, y),
                "n": len(qf),
                "n_wrong": int(wrong.sum()),
            }
        )
        results.append(
            {
                "model_id": model,
                "model_label": MODEL_LABEL[model],
                "error_cost": "",
                "decision_owner": "na",
                "policy": "score_quality",
                "policy_detail": "platt_cv",
                "visibility": "na",
                "brier": float(brier_score_loss(y, p_platt)),
                "auroc": safe_auroc(y, p_platt),
                "ece_10bin": expected_calibration_error(p_platt, y),
                "n": len(qf),
                "n_wrong": int(wrong.sum()),
            }
        )
    return results, calibrated_frames


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probabilities, outcomes):
        index = min(n_bins - 1, int(float(p) * n_bins))
        bins[index].append((float(p), float(y)))
    total = sum(len(items) for items in bins)
    if total == 0:
        return float("nan")
    ece = 0.0
    for items in bins:
        if not items:
            continue
        mean_p = sum(p for p, _ in items) / len(items)
        mean_y = sum(y for _, y in items) / len(items)
        ece += len(items) / total * abs(mean_p - mean_y)
    return ece


def part3_residual(
    qframes: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("Part 3: residual information in hidden action", flush=True)
    model_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for model in MODELS:
        qf = qframes[model]
        q = qf["probability_correct"].to_numpy(dtype=float)
        y_wrong = qf["wrong"].to_numpy(dtype=int)
        folds = make_stratified_folds(
            qf["is_correct"].to_numpy(dtype=int),
            n_splits=N_FOLDS,
            seed=cell_seed("folds", model),
        )
        for owner in OWNERS:
            for L in ERROR_COSTS:
                action = qf[f"hidden_{owner}_{int(L)}_verified"].to_numpy(dtype=float)
                X_a = q.reshape(-1, 1)
                X_b = np.column_stack([q, action])
                p_a = fit_logistic_oos(X_a, y_wrong, folds)
                p_b = fit_logistic_oos(X_b, y_wrong, folds)
                metrics = compare_oos(y_wrong, p_a, p_b, seed=cell_seed("resid", model, owner, L))
                model_rows.append(
                    {
                        "model_id": model,
                        "model_label": MODEL_LABEL[model],
                        "decision_owner": owner,
                        "error_cost": L,
                        "n": len(qf),
                        "n_wrong": int(y_wrong.sum()),
                        "hidden_verify_rate": float(action.mean()),
                        "hidden_action_nunique": int(np.unique(action).size),
                        **metrics,
                    }
                )
                bin_rows.extend(
                    residual_bins(qf, model, owner, L, q, y_wrong, action.astype(bool))
                )
    return model_rows, bin_rows


def _fast_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _fast_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _fast_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[y.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compare_oos(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, *, seed: int
) -> dict[str, float]:
    mask = np.isfinite(p_a) & np.isfinite(p_b)
    y_m = y[mask].astype(float)
    a = p_a[mask]
    b = p_b[mask]
    log_a = _fast_log_loss(y_m, a)
    log_b = _fast_log_loss(y_m, b)
    brier_a = _fast_brier(y_m, a)
    brier_b = _fast_brier(y_m, b)
    auc_a = _fast_auroc(y_m, a)
    auc_b = _fast_auroc(y_m, b)
    n = len(y_m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    yy = y_m[idx]
    aa = a[idx]
    bb = b[idx]
    mixed = yy.min(axis=1) != yy.max(axis=1)

    def _batch_log(y_mat: np.ndarray, p_mat: np.ndarray) -> np.ndarray:
        p_clip = np.clip(p_mat, 1e-6, 1.0 - 1e-6)
        return -np.mean(
            y_mat * np.log(p_clip) + (1.0 - y_mat) * np.log(1.0 - p_clip), axis=1
        )

    d_log = (_batch_log(yy, aa) - _batch_log(yy, bb))[mixed]
    d_brier = (((aa - yy) ** 2).mean(axis=1) - ((bb - yy) ** 2).mean(axis=1))[mixed]
    d_auc: list[float] = []
    for row in idx[::5]:
        y_row = y_m[row]
        if y_row.min() == y_row.max():
            continue
        d_auc.append(_fast_auroc(y_row, b[row]) - _fast_auroc(y_row, a[row]))
    log_lo, log_hi = _percentile_ci(d_log.tolist())
    brier_lo, brier_hi = _percentile_ci(d_brier.tolist())
    auc_lo, auc_hi = _percentile_ci(d_auc)
    return {
        "log_loss_conf_only": log_a,
        "log_loss_conf_plus_hidden": log_b,
        "delta_log_loss_A_minus_B": log_a - log_b,
        "delta_log_loss_ci_lower": log_lo,
        "delta_log_loss_ci_upper": log_hi,
        "brier_conf_only": brier_a,
        "brier_conf_plus_hidden": brier_b,
        "delta_brier_A_minus_B": brier_a - brier_b,
        "delta_brier_ci_lower": brier_lo,
        "delta_brier_ci_upper": brier_hi,
        "auroc_conf_only": auc_a,
        "auroc_conf_plus_hidden": auc_b,
        "delta_auroc_B_minus_A": auc_b - auc_a,
        "delta_auroc_ci_lower": auc_lo,
        "delta_auroc_ci_upper": auc_hi,
        "positive_delta_means": "positive log-loss/Brier delta: hidden action reduces loss; positive AUROC delta: hidden action improves ranking",
    }


def residual_bins(
    qf: pd.DataFrame,
    model: str,
    owner: str,
    L: float,
    q: np.ndarray,
    y_wrong: np.ndarray,
    action: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    # Exact confidence groups with both actions and n>=10.
    frame = pd.DataFrame({"q": q, "wrong": y_wrong.astype(bool), "verify": action})
    for q_value, group in frame.groupby("q"):
        n = len(group)
        n_v = int(group["verify"].sum())
        n_u = n - n_v
        if n < 10 or n_v == 0 or n_u == 0:
            continue
        err_v = float(group.loc[group["verify"], "wrong"].mean())
        err_u = float(group.loc[~group["verify"], "wrong"].mean())
        rows.append(
            {
                "model_id": model,
                "decision_owner": owner,
                "error_cost": L,
                "bin_type": "exact_q",
                "bin_label": f"q={q_value}",
                "q_value": float(q_value),
                "n": n,
                "n_verify": n_v,
                "n_use": n_u,
                "error_rate_verify": err_v,
                "error_rate_use": err_u,
                "error_rate_diff_verify_minus_use": err_v - err_u,
            }
        )
    # Quantile bins, dropping duplicate edges caused by mass points.
    try:
        frame["qbin"] = pd.qcut(frame["q"], q=5, duplicates="drop")
    except ValueError:
        return rows
    for bin_label, group in frame.groupby("qbin", observed=True):
        n = len(group)
        n_v = int(group["verify"].sum())
        n_u = n - n_v
        if n < 10 or n_v == 0 or n_u == 0:
            continue
        err_v = float(group.loc[group["verify"], "wrong"].mean())
        err_u = float(group.loc[~group["verify"], "wrong"].mean())
        rows.append(
            {
                "model_id": model,
                "decision_owner": owner,
                "error_cost": L,
                "bin_type": "quantile",
                "bin_label": str(bin_label),
                "q_value": float(group["q"].mean()),
                "n": n,
                "n_verify": n_v,
                "n_use": n_u,
                "error_rate_verify": err_v,
                "error_rate_use": err_u,
                "error_rate_diff_verify_minus_use": err_v - err_u,
            }
        )
    return rows


def part4_correlation(cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
    print("Part 4: exploratory signal-gap vs visibility-cost scatter", flush=True)
    xs = np.array([r["catch_diff_hidden_minus_conf"] for r in cell_rows], dtype=float)
    ys = np.array([r["cost_diff_visible_minus_hidden"] for r in cell_rows], dtype=float)
    overall = {
        "n_cells": len(cell_rows),
        "pearson_r_descriptive_only": float(pd.Series(xs).corr(pd.Series(ys))),
        "spearman_r_descriptive_only": float(pd.Series(xs).corr(pd.Series(ys), method="spearman")),
        "warning": (
            "These 32 cells are not independent. Each model contributes 8 cells "
            "that reuse the same 500 questions. Do not treat r as a confirmatory test."
        ),
        "within_model": {},
    }
    for model in MODELS:
        sub = [r for r in cell_rows if r["model_id"] == model]
        x = pd.Series([r["catch_diff_hidden_minus_conf"] for r in sub])
        y = pd.Series([r["cost_diff_visible_minus_hidden"] for r in sub])
        overall["within_model"][model] = {
            "n_cells": len(sub),
            "pearson_r": float(x.corr(y)),
            "spearman_r": float(x.corr(y, method="spearman")),
        }
    return overall


def plot_risk_coverage(qframes: dict[str, pd.DataFrame], cell_rows: list[dict[str, Any]]) -> None:
    plt = pyplot()
    owner_marker = {"ai_system": "o", "human": "s"}
    L_edge = {2.0: 0.35, 5.0: 0.55, 10.0: 0.75, 20.0: 1.0}
    for model, qf in qframes.items():
        q = qf["probability_correct"].to_numpy(dtype=float)
        wrong = qf["wrong"].to_numpy(dtype=bool)
        cover, catch = catch_curve(q, wrong)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls=":", label="Random checking")
        ax.plot(
            cover,
            catch,
            color=MODEL_COLOR[model],
            lw=2,
            label="Confidence ranking (low q first)",
        )
        for owner in OWNERS:
            for L in ERROR_COSTS:
                hid = hidden_rates(
                    qf[f"hidden_{owner}_{int(L)}_verified"].to_numpy(dtype=bool), wrong
                )
                vis = hidden_rates(
                    qf[f"visible_{owner}_{int(L)}_verified"].to_numpy(dtype=bool), wrong
                )
                ax.scatter(
                    hid["verify_rate"],
                    hid["error_catch_rate"],
                    marker=owner_marker[owner],
                    s=55,
                    facecolors=MODEL_COLOR[model],
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=L_edge[L],
                    zorder=3,
                    label=f"Hidden {owner} L={int(L)}" if owner == "ai_system" else None,
                )
                ax.scatter(
                    vis["verify_rate"],
                    vis["error_catch_rate"],
                    marker=owner_marker[owner],
                    s=55,
                    facecolors="none",
                    edgecolors=MODEL_COLOR[model],
                    linewidths=1.1,
                    alpha=L_edge[L],
                    zorder=3,
                )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Fraction of answers verified")
        ax.set_ylabel("Fraction of wrong answers caught")
        ax.set_title(f"{MODEL_LABEL[model]}: can hidden checking beat confidence ranking?")
        handles, labels = ax.get_legend_handles_labels()
        # Keep the curve legends plus a small manual note.
        ax.legend(handles[:3], labels[:3], loc="lower right")
        ax.text(
            0.02,
            0.98,
            "Filled = hidden policy\nOpen = visible policy\nSquare = human owner, circle = AI owner\nDarker = higher L",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"risk_coverage_{model}.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"risk_coverage_{model}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_calibration(calibrated: dict[str, pd.DataFrame]) -> None:
    plt = pyplot()
    for model, qf in calibrated.items():
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        y = qf["is_correct"].to_numpy(dtype=float)
        specs = [
            (axes[0], qf["probability_correct"].to_numpy(dtype=float), "Raw Stage-2 confidence"),
            (axes[1], qf["p_isotonic"].to_numpy(dtype=float), "Cross-fitted isotonic p"),
        ]
        for ax, p, title in specs:
            # Reliability using 10 equal-width bins plus jittered scatter of bin means.
            bins = np.linspace(0, 1, 11)
            xs, ys, ns = [], [], []
            for i in range(10):
                if i < 9:
                    mask = (p >= bins[i]) & (p < bins[i + 1])
                else:
                    mask = (p >= bins[i]) & (p <= bins[i + 1])
                if not mask.any():
                    continue
                xs.append(float(p[mask].mean()))
                ys.append(float(y[mask].mean()))
                ns.append(int(mask.sum()))
            ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls=":")
            ax.plot(xs, ys, "-o", color=MODEL_COLOR[model], lw=1.6)
            for x, yy, n in zip(xs, ys, ns):
                ax.text(x, yy, str(n), fontsize=7, ha="left", va="bottom", color="#333333")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean reported / calibrated probability")
            ax.set_ylabel("Observed accuracy")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"{MODEL_LABEL[model]} calibration (numbers = bin count)", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"calibration_{model}.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"calibration_{model}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        q = qf["probability_correct"].to_numpy(dtype=float)
        wrong = qf["wrong"].to_numpy(dtype=bool)
        ax.hist(q[~wrong], bins=20, range=(0, 1), alpha=0.55, label="Correct answers", color="#4C78A8")
        ax.hist(q[wrong], bins=20, range=(0, 1), alpha=0.7, label="Wrong answers", color="#E45756")
        ax.set_xlabel("Stage-2 reported confidence")
        ax.set_ylabel("Number of questions")
        ax.set_title(f"{MODEL_LABEL[model]}: confidence by correctness")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"confidence_hist_{model}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_signal_gap(cell_rows: list[dict[str, Any]]) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    owner_marker = {"ai_system": "o", "human": "s"}
    for model in MODELS:
        for owner in OWNERS:
            sub = [r for r in cell_rows if r["model_id"] == model and r["decision_owner"] == owner]
            xs = [r["catch_diff_hidden_minus_conf"] for r in sub]
            ys = [r["cost_diff_visible_minus_hidden"] for r in sub]
            ax.scatter(
                xs,
                ys,
                marker=owner_marker[owner],
                s=60,
                color=MODEL_COLOR[model],
                label=f"{MODEL_SHORT[model]} / {owner}",
                zorder=3,
            )
            for r in sub:
                ax.annotate(
                    f"L={int(r['error_cost'])}",
                    (r["catch_diff_hidden_minus_conf"], r["cost_diff_visible_minus_hidden"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=MODEL_COLOR[model],
                )
    ax.axhline(0, color="#888888", lw=1)
    ax.axvline(0, color="#888888", lw=1)
    ax.set_xlabel("Signal gap: hidden catch − confidence-only catch (same budget)\npositive = hidden ranking is better")
    ax.set_ylabel("Visibility effect on mean cost: visible − hidden\npositive = showing confidence hurt (higher cost)")
    ax.set_title("Exploratory only: 32 cells are not independent samples")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "signal_gap_vs_visibility.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "signal_gap_vs_visibility.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cost_policies(cal_rows: list[dict[str, Any]]) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), sharex=True)
    wanted = {
        "raw_confidence_threshold": ("Raw-q threshold", "--", "o"),
        "isotonic_calibrated_threshold": ("Isotonic threshold", "-", "D"),
        "actual_hidden": ("Hidden model policy", "-", "s"),
        "actual_visible": ("Visible model policy", ":", "^"),
    }
    for ax, model in zip(axes.flat, MODELS):
        for policy, (label, ls, marker) in wanted.items():
            sub = [
                r
                for r in cal_rows
                if r["model_id"] == model
                and r["policy"] == policy
                and r.get("decision_owner") in {"na", "pooled_owners"}
                and r.get("error_cost") != ""
            ]
            sub = sorted(sub, key=lambda r: float(r["error_cost"]))
            if policy.startswith("actual"):
                # Prefer pooled-owner rows.
                sub = [r for r in sub if r["decision_owner"] == "pooled_owners"]
            xs = [float(r["error_cost"]) for r in sub]
            ys = [r["mean_realized_cost"] for r in sub]
            ax.plot(xs, ys, ls=ls, marker=marker, label=label, lw=1.8)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xticks(list(ERROR_COSTS))
        ax.set_ylabel("Mean realized cost")
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Error cost L")
    axes[1, 1].set_xlabel("Error cost L")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Cost under the actual (1-p)×L > 1 rule vs the model's Stage-3 policy")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "cost_policies_by_L.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_overview(resid_rows: list[dict[str, Any]]) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model in MODELS:
        sub = [r for r in resid_rows if r["model_id"] == model]
        xs = [r["error_cost"] + (-0.25 if r["decision_owner"] == "ai_system" else 0.25) for r in sub]
        ys = [r["delta_log_loss_A_minus_B"] for r in sub]
        ylo = [r["delta_log_loss_ci_lower"] for r in sub]
        yhi = [r["delta_log_loss_ci_upper"] for r in sub]
        ax.errorbar(
            xs,
            ys,
            yerr=np.array(
                [
                    np.clip(np.array(ys) - np.array(ylo), 0, None),
                    np.clip(np.array(yhi) - np.array(ys), 0, None),
                ]
            ),
            fmt="o" if True else "s",
            color=MODEL_COLOR[model],
            label=MODEL_SHORT[model],
            capsize=3,
        )
    ax.axhline(0, color="#888888", lw=1)
    ax.set_xticks(list(ERROR_COSTS))
    ax.set_xlabel("Error cost L")
    ax.set_ylabel("Log-loss improvement from adding hidden action\n(positive = hidden action helps)")
    ax.set_title("Does hidden VERIFY/USE add predictive information beyond q?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_logloss_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_model_summary(
    cell_rows: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    resid_rows: list[dict[str, Any]],
    qframes: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    out = []
    for model in MODELS:
        pool = next(
            r
            for r in pooled
            if r["model_id"] == model and r["scope"] == "model_pooled_over_owner_and_L"
        )
        cells = [r for r in cell_rows if r["model_id"] == model]
        resid = [r for r in resid_rows if r["model_id"] == model]
        quality = [
            r
            for r in cal_rows
            if r["model_id"] == model and r["policy"] == "score_quality"
        ]
        raw_q = next(r for r in quality if r["policy_detail"] == "raw_q")
        iso_q = next(r for r in quality if r["policy_detail"] == "isotonic_cv")
        qf = qframes[model]
        vis_agree = _mean(r["visible_vs_raw_threshold_agreement"] for r in cells)
        hid_agree = _mean(r["hidden_vs_raw_threshold_agreement"] for r in cells)
        out.append(
            {
                "model_id": model,
                "model_label": MODEL_LABEL[model],
                "n_questions": pool["n_questions"],
                "n_wrong": pool["n_wrong"],
                "accuracy": pool["accuracy"],
                "brier_raw": raw_q["brier"],
                "auroc_raw": raw_q["auroc"],
                "ece_raw_10bin": raw_q["ece_10bin"],
                "brier_isotonic_cv": iso_q["brier"],
                "auroc_isotonic_cv": iso_q["auroc"],
                "ece_isotonic_cv_10bin": iso_q["ece_10bin"],
                "auroc_wrong_one_minus_q": pool["auroc_wrong_using_one_minus_q"],
                "auprc_wrong_one_minus_q": pool["auprc_wrong_using_one_minus_q"],
                "mean_q_correct": pool["mean_q_correct"],
                "mean_q_wrong": pool["mean_q_wrong"],
                "mean_hidden_verify_budget": pool["mean_hidden_verify_budget"],
                "mean_hidden_error_catch_rate": pool["mean_hidden_error_catch_rate"],
                "mean_conf_error_catch_rate": pool["mean_conf_error_catch_rate"],
                "mean_catch_diff_hidden_minus_conf": pool["mean_catch_diff"],
                "catch_diff_ci_lower": pool["catch_diff_ci_lower"],
                "catch_diff_ci_upper": pool["catch_diff_ci_upper"],
                "n_cells_catch_diff_ci_excludes_zero": pool["n_cells_with_ci_excluding_zero"],
                "n_cells_hidden_better": pool["n_cells_positive_diff"],
                "n_cells_conf_better": pool["n_cells_negative_diff"],
                "mean_visible_minus_hidden_cost": _mean(
                    r["cost_diff_visible_minus_hidden"] for r in cells
                ),
                "mean_visible_vs_raw_threshold_agreement": vis_agree,
                "mean_hidden_vs_raw_threshold_agreement": hid_agree,
                "mean_delta_log_loss_hidden_action": _mean(
                    r["delta_log_loss_A_minus_B"] for r in resid
                ),
                "n_resid_cells_logloss_ci_gt0": sum(
                    1 for r in resid if r["delta_log_loss_ci_lower"] > 0
                ),
                "n_resid_cells_logloss_ci_lt0": sum(
                    1 for r in resid if r["delta_log_loss_ci_upper"] < 0
                ),
            }
        )
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, validation = validate_and_load()
    cell_rows, qframes = part1_equal_budget(df)
    pooled = pooled_summaries(cell_rows, qframes)
    add_confidence_quality(pooled, qframes)
    cal_rows, calibrated = part2_calibration(qframes)
    resid_rows, bin_rows = part3_residual(qframes)
    corr = part4_correlation(cell_rows)
    model_summary = build_model_summary(cell_rows, pooled, cal_rows, resid_rows, qframes)

    print("Writing tables and figures", flush=True)
    write_csv(OUT_DIR / "checkpoint_A_cell_results.csv", cell_rows)
    write_csv(OUT_DIR / "checkpoint_A_model_summary.csv", model_summary)
    write_csv(OUT_DIR / "checkpoint_A_calibration_results.csv", cal_rows)
    write_csv(OUT_DIR / "checkpoint_A_residual_results.csv", resid_rows)
    write_csv(OUT_DIR / "checkpoint_A_residual_bins.csv", bin_rows)
    write_csv(OUT_DIR / "checkpoint_A_pooled_catch_summaries.csv", pooled)
    write_json(OUT_DIR / "part4_descriptive_correlation.json", corr)

    plot_risk_coverage(qframes, cell_rows)
    plot_calibration(calibrated)
    plot_signal_gap(cell_rows)
    plot_cost_policies(cal_rows)
    plot_residual_overview(resid_rows)

    write_json(
        OUT_DIR / "run_metadata.json",
        {
            "seed": SEED,
            "n_bootstrap": N_BOOT,
            "n_folds": N_FOLDS,
            "tie_handling": (
                "Fractional/expected inclusion at the cutoff confidence value. "
                "All strictly lower-q items are verified. Items tied at the cutoff "
                "share equal weight so the expected number verified equals the "
                "hidden policy's integer verification count."
            ),
            "calibration": (
                "Per-model 5-fold StratifiedKFold on Stage-1 correctness, seed "
                f"{SEED} hashed with model id. Primary calibrator is pool-adjacent-"
                "violators isotonic regression from src.calibration. Robustness "
                "calibrator is unregularized logistic / Platt scaling of raw q."
            ),
            "residual_auroc_bootstrap": (
                "Log-loss and Brier deltas use 5000 question-level replicates. "
                "AUROC deltas use every 5th replicate (1000 samples) because the "
                "rank statistic is slower; the point estimate still uses the full sample."
            ),
            "excluded": (
                f"{validation['n_robustness_stage3']} paraphrase Stage-3 rows "
                f"({ROBUSTNESS_FAMILY}) were excluded from Checkpoint A."
            ),
        },
    )
    print("Wrote outputs to", OUT_DIR, flush=True)
    print("Figures in", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
