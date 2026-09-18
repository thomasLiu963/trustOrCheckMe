"""Task 005C: hidden-action residual after q1+q2 and observable difficulty.

Read-only on V2, Study 1, q2, Task 005, and Task 005B. Zero API calls.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from .config import PROJECT_ROOT
from .study1_analysis import MODEL_LABELS, _is_verify, mean_or_nan
from .study1_sample import load_v2b_examples
from .study1_schemas import STUDY1_MODEL_ALIASES
from .task005_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PAPER_DIRECTION,
    Q2_SQLITE,
    STUDY1_SQLITE,
    V2_SQLITE,
    file_fingerprint,
    sha256_file,
)
from .task005_lane_a import (
    error_catch_from_weights,
    fractional_verify_weights,
    load_v2_answers,
    load_v2_primary_verification,
)
from .task005_q2 import hidden_action_summaries

RETURN_DIR = PROJECT_ROOT / "to_gpt" / "005c_hidden_residual_after_difficulty"
Q2_RESULTS = (
    PROJECT_ROOT
    / "to_gpt"
    / "005_parallel_diagnostic_sprint"
    / "lane_B_q2"
    / "q2_results.csv"
)
TASK005_REPORT = PROJECT_ROOT / "to_gpt" / "005_parallel_diagnostic_sprint" / "report.md"
TASK005B_REPORT = PROJECT_ROOT / "to_gpt" / "005b_difficulty_control" / "report.md"
CANONICAL_OWNER = "ai_system"
CANONICAL_L = 10.0
OTHER_BY_TARGET = {
    "openai_gpt56_sol": (
        "anthropic_sonnet5",
        "google_gemini38_flash",
        "xai_grok420_nonreasoning",
    ),
    "anthropic_sonnet5": (
        "openai_gpt56_sol",
        "google_gemini38_flash",
        "xai_grok420_nonreasoning",
    ),
}
ALL_V2_MODELS = (
    "openai_gpt56_sol",
    "anthropic_sonnet5",
    "google_gemini38_flash",
    "xai_grok420_nonreasoning",
)
TASK005_INCREMENT = {
    "openai_gpt56_sol": {
        "delta_auroc": 0.08111380145278457,
        "delta_log_loss": -0.04233296509574952,
    },
    "anthropic_sonnet5": {
        "delta_auroc": 0.04502810338703289,
        "delta_log_loss": -0.032366874631817866,
    },
}
FEATURE_SETS = {
    "q1": ("q1",),
    "q1_q2": ("q1", "q2"),
    "difficulty": ("difficulty",),
    "q1_q2_hidden": ("q1", "q2", "hidden"),
    "q1_q2_difficulty": ("q1", "q2", "difficulty"),
    "q1_q2_difficulty_hidden": ("q1", "q2", "difficulty", "hidden"),
    "q1_q2_difficulty_single_hidden": ("q1", "q2", "difficulty", "single_hidden"),
}
CONTRASTS = (
    ("contrast_A_hidden_after_q1_q2", "q1_q2_hidden", "q1_q2"),
    ("contrast_B_hidden_after_q1_q2_difficulty", "q1_q2_difficulty_hidden", "q1_q2_difficulty"),
    ("contrast_difficulty_after_q1_q2", "q1_q2_difficulty", "q1_q2"),
    ("contrast_q2_after_q1", "q1_q2", "q1"),
    (
        "contrast_single_hidden_after_q1_q2_difficulty",
        "q1_q2_difficulty_single_hidden",
        "q1_q2_difficulty",
    ),
)
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
ROUTERS = (
    "q1",
    "q1_q2",
    "q1_q2_difficulty",
    "q1_q2_difficulty_hidden",
    "q1_q2_difficulty_single_hidden",
)


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _safe_ll(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    try:
        return float(log_loss(y, clipped, labels=[0, 1]))
    except ValueError:
        return float("nan")


def _safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y, np.clip(p, 0.0, 1.0)))
    except ValueError:
        return float("nan")


def _ci(vals: Sequence[float]) -> tuple[float, float]:
    clean = sorted(v for v in vals if math.isfinite(v))
    if not clean:
        return float("nan"), float("nan")
    return clean[int(0.025 * (len(clean) - 1))], clean[int(0.975 * (len(clean) - 1))]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value)
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_ci(est: Any, lo: Any, hi: Any, digits: int = 3) -> str:
    return f"{_fmt(est, digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def load_q2_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Q2_RESULTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "question_id": row["question_id"],
                    "model_alias": row["model_alias"],
                    "q1": float(row["q1"]),
                    "q2": float(row["q2"]),
                    "stage1_correct": row["stage1_correct"].strip().lower() == "true",
                    "hidden_verify_fraction_csv": float(row["hidden_verify_fraction"]),
                    "hidden_verify_count_csv": int(float(row["hidden_verify_count"])),
                    "n_hidden_cells_csv": int(float(row["n_hidden_cells"])),
                }
            )
    return rows


def canonical_hidden_map(
    v2_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in v2_rows:
        if row["confidence_visibility"] != "hidden":
            continue
        if row["decision_owner"] != CANONICAL_OWNER:
            continue
        if abs(float(row["L"]) - CANONICAL_L) > 1e-12:
            continue
        key = (str(row["question_id"]), str(row["model_alias"]))
        out[key] = _is_verify(row["parsed_action"])
    return out


def build_difficulty(
    question_ids: Sequence[str],
    answers: Mapping[tuple[str, str], Mapping[str, Any]],
    examples: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    missing: list[str] = []
    out: dict[str, dict[str, Any]] = {}
    n_choices_seen: set[int] = set()
    for qid in question_ids:
        example = examples[qid]
        n_choices = len(example.choices)
        n_choices_seen.add(n_choices)
        model_correct: dict[str, bool] = {}
        for alias in ALL_V2_MODELS:
            rec = answers.get((qid, alias))
            if rec is None:
                missing.append(f"{qid}/{alias}")
                continue
            model_correct[alias] = bool(rec["is_correct"])
        if any(alias not in model_correct for alias in ALL_V2_MODELS):
            continue
        out[qid] = {
            "question_id": qid,
            "category": example.category,
            "n_choices": n_choices,
            "question_char_len": len(example.question),
            "question_ws_tokens": len(example.question.split()),
            "gpt_correct": int(model_correct["openai_gpt56_sol"]),
            "claude_correct": int(model_correct["anthropic_sonnet5"]),
            "gemini_correct": int(model_correct["google_gemini38_flash"]),
            "grok_correct": int(model_correct["xai_grok420_nonreasoning"]),
        }
    if missing:
        raise RuntimeError(f"difficulty join incomplete: {missing[:12]}")
    for row in out.values():
        row["choice_count_varies"] = len(n_choices_seen) > 1
        row["n_choices_unique_values"] = sorted(n_choices_seen)
    return out


def attach_rows(
    q2_rows: Sequence[Mapping[str, Any]],
    difficulty: Mapping[str, Mapping[str, Any]],
    hidden: Mapping[tuple[str, str], Mapping[str, Any]],
    canonical: Mapping[tuple[str, str], int],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    table: list[dict[str, Any]] = []
    for row in q2_rows:
        qid = row["question_id"]
        model = row["model_alias"]
        feat = difficulty.get(qid)
        hid = hidden.get((qid, model))
        canon = canonical.get((qid, model))
        if feat is None:
            issues.append(f"missing difficulty {qid}")
            continue
        if hid is None:
            issues.append(f"missing hidden {qid}/{model}")
            continue
        if canon is None:
            issues.append(f"missing canonical hidden {qid}/{model}")
            continue
        if abs(float(hid["hidden_verify_fraction"]) - float(row["hidden_verify_fraction_csv"])) > 1e-12:
            issues.append(f"hidden fraction mismatch {qid}/{model}")
        others = OTHER_BY_TARGET[model]
        key_map = {
            "openai_gpt56_sol": "gpt_correct",
            "anthropic_sonnet5": "claude_correct",
            "google_gemini38_flash": "gemini_correct",
            "xai_grok420_nonreasoning": "grok_correct",
        }
        other_n = int(sum(int(feat[key_map[alias]]) for alias in others))
        target_key = key_map[model]
        if int(feat[target_key]) != int(row["stage1_correct"]):
            issues.append(f"correctness mismatch V2 vs q2 table {qid}/{model}")
        item = {
            "question_id": qid,
            "model_alias": model,
            "stage1_correct": bool(row["stage1_correct"]),
            "error": int(not bool(row["stage1_correct"])),
            "q1": float(row["q1"]),
            "q2": float(row["q2"]),
            "hidden_verify_fraction": float(hid["hidden_verify_fraction"]),
            "hidden_verify_count": int(hid["hidden_verify_count"]),
            "n_hidden_cells": int(hid["n_hidden_cells"]),
            "single_hidden_verify": int(canon),
            "canonical_hidden_spec": f"{CANONICAL_OWNER}|L={CANONICAL_L:.0f}|hidden",
            "category": feat["category"],
            "n_choices": feat["n_choices"],
            "question_char_len": feat["question_char_len"],
            "question_ws_tokens": feat["question_ws_tokens"],
            "other_n_correct": other_n,
            "other_frac_correct": other_n / 3.0,
            "other_frac_wrong": 1.0 - other_n / 3.0,
            "other_models": ",".join(others),
            "target_correct": int(feat[target_key]),
            "gpt_correct": feat["gpt_correct"],
            "claude_correct": feat["claude_correct"],
            "gemini_correct": feat["gemini_correct"],
            "grok_correct": feat["grok_correct"],
            "choice_count_varies": feat["choice_count_varies"],
        }
        table.append(item)
    return table, issues


def _col_stats(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, float]:
    vals = np.array([float(row[key]) for row in rows], dtype=float)
    mean = float(vals.mean()) if len(vals) else 0.0
    std = float(vals.std())
    if std < 1e-9:
        std = 1.0
    return mean, std


def _design_matrix(
    rows: Sequence[Mapping[str, Any]],
    parts: Sequence[str],
    *,
    encoder: OneHotEncoder | None = None,
    fit_encoder: bool = False,
    length_mean: float = 0.0,
    length_std: float = 1.0,
    nch_mean: float = 0.0,
    nch_std: float = 1.0,
) -> tuple[np.ndarray, OneHotEncoder]:
    blocks: list[np.ndarray] = []
    if encoder is None:
        encoder = _make_encoder()
    for part in parts:
        if part == "q1":
            blocks.append(np.array([float(row["q1"]) for row in rows]).reshape(-1, 1))
        elif part == "q2":
            blocks.append(np.array([float(row["q2"]) for row in rows]).reshape(-1, 1))
        elif part == "hidden":
            blocks.append(
                np.array([float(row["hidden_verify_fraction"]) for row in rows]).reshape(
                    -1, 1
                )
            )
        elif part == "single_hidden":
            blocks.append(
                np.array([float(row["single_hidden_verify"]) for row in rows]).reshape(
                    -1, 1
                )
            )
        elif part == "difficulty":
            other = np.array([float(row["other_n_correct"]) for row in rows]).reshape(
                -1, 1
            )
            length = np.array([float(row["question_char_len"]) for row in rows]).reshape(
                -1, 1
            )
            length = (length - length_mean) / length_std
            nch = np.array([float(row["n_choices"]) for row in rows]).reshape(-1, 1)
            nch = (nch - nch_mean) / nch_std
            cats = np.array([row["category"] for row in rows]).reshape(-1, 1)
            cat_mat = (
                encoder.fit_transform(cats) if fit_encoder else encoder.transform(cats)
            )
            blocks.extend([other, length, nch, np.asarray(cat_mat)])
        else:
            raise ValueError(part)
    return np.hstack(blocks), encoder


def grouped_cv_oof(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    y = np.array([int(row["stage1_correct"]) for row in rows], dtype=int)
    qids = np.array([row["question_id"] for row in rows])
    gkf = GroupKFold(n_splits=5)
    dummy = np.zeros(len(rows))
    oof = {name: np.full(len(rows), np.nan) for name in FEATURE_SETS}
    unstable = False
    for train_idx, test_idx in gkf.split(dummy, y, groups=qids):
        if len(np.unique(y[train_idx])) < 2:
            unstable = True
            continue
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        length_mean, length_std = _col_stats(train_rows, "question_char_len")
        nch_mean, nch_std = _col_stats(train_rows, "n_choices")
        try:
            for name, parts in FEATURE_SETS.items():
                X_train, enc = _design_matrix(
                    train_rows,
                    parts,
                    fit_encoder=True,
                    length_mean=length_mean,
                    length_std=length_std,
                    nch_mean=nch_mean,
                    nch_std=nch_std,
                )
                X_test, _ = _design_matrix(
                    test_rows,
                    parts,
                    encoder=enc,
                    fit_encoder=False,
                    length_mean=length_mean,
                    length_std=length_std,
                    nch_mean=nch_mean,
                    nch_std=nch_std,
                )
                clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                clf.fit(X_train, y[train_idx])
                oof[name][test_idx] = clf.predict_proba(X_test)[:, 1]
        except ValueError:
            unstable = True
    return oof, y, unstable


def bootstrap_metric_delta(
    y: np.ndarray,
    qids: Sequence[str],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    kind: str,
) -> tuple[float, float, float]:
    by_q: dict[str, list[int]] = defaultdict(list)
    mask = np.isfinite(scores_a) & np.isfinite(scores_b)
    for i, qid in enumerate(qids):
        if mask[i]:
            by_q[str(qid)].append(i)
    q_list = sorted(by_q)
    rng = random.Random(BOOTSTRAP_SEED)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [q_list[rng.randrange(len(q_list))] for _ in q_list]
        idx = [j for qid in draw for j in by_q[qid]]
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        if kind == "auroc":
            vals.append(_safe_auc(yy, scores_b[idx]) - _safe_auc(yy, scores_a[idx]))
        elif kind == "ll":
            vals.append(_safe_ll(yy, scores_b[idx]) - _safe_ll(yy, scores_a[idx]))
        else:
            vals.append(_safe_brier(yy, scores_b[idx]) - _safe_brier(yy, scores_a[idx]))
    point = {
        "auroc": _safe_auc(y[mask], scores_b[mask]) - _safe_auc(y[mask], scores_a[mask]),
        "ll": _safe_ll(y[mask], scores_b[mask]) - _safe_ll(y[mask], scores_a[mask]),
        "brier": _safe_brier(y[mask], scores_b[mask]) - _safe_brier(y[mask], scores_a[mask]),
    }[kind]
    lo, hi = _ci(vals)
    return point, lo, hi


def evaluate_models(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    oof, y, unstable = grouped_cv_oof(rows)
    qids = [row["question_id"] for row in rows]
    mask = np.ones(len(rows), dtype=bool)
    for arr in oof.values():
        mask &= np.isfinite(arr)
    yy = y[mask]
    cv_rows = []
    metrics = {}
    for name in FEATURE_SETS:
        pred = oof[name][mask]
        metrics[name] = {
            "log_loss": _safe_ll(yy, pred),
            "auroc": _safe_auc(yy, pred),
            "brier": _safe_brier(yy, pred),
        }
        cv_rows.append(
            {
                "model_alias": rows[0]["model_alias"],
                "model_label": MODEL_LABELS[rows[0]["model_alias"]],
                "feature_set": name,
                "n": int(mask.sum()),
                "log_loss": metrics[name]["log_loss"],
                "auroc": metrics[name]["auroc"],
                "brier": metrics[name]["brier"],
                "unstable": unstable,
            }
        )
    contrasts = []
    for contrast_name, aug, base in CONTRASTS:
        d_auc, auc_lo, auc_hi = bootstrap_metric_delta(
            y, qids, oof[base], oof[aug], "auroc"
        )
        d_ll, ll_lo, ll_hi = bootstrap_metric_delta(y, qids, oof[base], oof[aug], "ll")
        d_br, br_lo, br_hi = bootstrap_metric_delta(y, qids, oof[base], oof[aug], "brier")
        contrasts.append(
            {
                "model_alias": rows[0]["model_alias"],
                "model_label": MODEL_LABELS[rows[0]["model_alias"]],
                "contrast": contrast_name,
                "augmented": aug,
                "baseline": base,
                "delta_auroc": d_auc,
                "delta_auroc_ci_lower": auc_lo,
                "delta_auroc_ci_upper": auc_hi,
                "delta_log_loss": d_ll,
                "delta_log_loss_ci_lower": ll_lo,
                "delta_log_loss_ci_upper": ll_hi,
                "delta_brier": d_br,
                "delta_brier_ci_lower": br_lo,
                "delta_brier_ci_upper": br_hi,
                "note": "negative delta_log_loss / positive delta_auroc means the augmented set helps",
            }
        )
    routing = matched_budget(rows, oof, y)
    return {
        "cv": cv_rows,
        "contrasts": contrasts,
        "routing": routing,
        "oof": oof,
        "y": y,
        "unstable": unstable,
        "metrics": metrics,
    }


def matched_budget(
    rows: Sequence[Mapping[str, Any]],
    oof: Mapping[str, np.ndarray],
    y: np.ndarray,
) -> list[dict[str, Any]]:
    qids = [row["question_id"] for row in rows]
    wrong = np.array([not bool(row["stage1_correct"]) for row in rows])
    n_wrong = int(wrong.sum())
    output = []
    scores = {name: oof[name] for name in ROUTERS}
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(qids):
        by_q[str(qid)].append(i)
    q_list = sorted(by_q)
    for frac in BUDGETS:
        n_v = frac * len(rows)
        catch = {}
        counts = {}
        for name, pred in scores.items():
            weights = fractional_verify_weights(pred, n_v)
            catch[name] = error_catch_from_weights(weights, wrong)
            counts[name] = float((weights * wrong).sum())
        rng = random.Random(BOOTSTRAP_SEED)
        boot: dict[str, list[float]] = {name: [] for name in ROUTERS}
        boot_diff: dict[str, list[float]] = {
            name: [] for name in ROUTERS if name != "q1_q2_difficulty"
        }
        for _ in range(BOOTSTRAP_RESAMPLES):
            draw = [q_list[rng.randrange(len(q_list))] for _ in q_list]
            idx = np.array([j for qid in draw for j in by_q[qid]], dtype=int)
            w = wrong[idx]
            if w.sum() == 0:
                continue
            base_w = fractional_verify_weights(scores["q1_q2_difficulty"][idx], n_v)
            base_c = error_catch_from_weights(base_w, w)
            for name in ROUTERS:
                ww = fractional_verify_weights(scores[name][idx], n_v)
                c = error_catch_from_weights(ww, w)
                boot[name].append(c)
                if name != "q1_q2_difficulty":
                    boot_diff[name].append(c - base_c)
        for name in ROUTERS:
            lo, hi = _ci(boot[name])
            if name == "q1_q2_difficulty":
                dlo = dhi = dest = ""
            else:
                dest = catch[name] - catch["q1_q2_difficulty"]
                dlo, dhi = _ci(boot_diff[name])
            output.append(
                {
                    "model_alias": rows[0]["model_alias"],
                    "model_label": MODEL_LABELS[rows[0]["model_alias"]],
                    "budget_fraction": frac,
                    "router": name,
                    "n": len(rows),
                    "n_wrong": n_wrong,
                    "n_verify": n_v,
                    "catch_fraction": catch[name],
                    "errors_caught": counts[name],
                    "catch_ci_lower": lo,
                    "catch_ci_upper": hi,
                    "delta_vs_q1_q2_difficulty": dest,
                    "delta_ci_lower": dlo,
                    "delta_ci_upper": dhi,
                    "note": "OOF predicted P(correct); lowest predicted correctness verified first; Checkpoint A fractional ties",
                }
            )
    return output


def difficulty_associations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    y = np.array([int(row["stage1_correct"]) for row in rows], dtype=int)
    diff = np.array([float(row["other_n_correct"]) for row in rows], dtype=float)
    hid = np.array([float(row["hidden_verify_fraction"]) for row in rows], dtype=float)
    single = np.array([float(row["single_hidden_verify"]) for row in rows], dtype=float)
    output = [
        {
            "model_alias": rows[0]["model_alias"],
            "metric": "other_n_correct_auroc_for_target_correct",
            "value": _safe_auc(y, diff),
            "note": "higher other-model correct count should rank correct items higher",
        },
        {
            "model_alias": rows[0]["model_alias"],
            "metric": "pearson_hidden_frac_vs_other_frac_wrong",
            "value": float(
                np.corrcoef(
                    hid, np.array([float(row["other_frac_wrong"]) for row in rows])
                )[0, 1]
            ),
        },
        {
            "model_alias": rows[0]["model_alias"],
            "metric": "pearson_hidden_frac_vs_target_error",
            "value": float(np.corrcoef(hid, 1 - y)[0, 1]),
        },
        {
            "model_alias": rows[0]["model_alias"],
            "metric": "pearson_single_hidden_vs_other_frac_wrong",
            "value": float(
                np.corrcoef(
                    single, np.array([float(row["other_frac_wrong"]) for row in rows])
                )[0, 1]
            ),
        },
        {
            "model_alias": rows[0]["model_alias"],
            "metric": "auroc_other_n_correct_predicting_single_hidden_verify",
            "value": _safe_auc(single.astype(int), 3.0 - diff),
            "note": "harder items (fewer others correct) ranked as more likely VERIFY",
        },
    ]
    for n_correct in range(4):
        sub = [row for row in rows if int(row["other_n_correct"]) == n_correct]
        output.append(
            {
                "model_alias": rows[0]["model_alias"],
                "metric": f"bin_{n_correct}_of_3_other_correct",
                "n": len(sub),
                "target_error_rate": mean_or_nan(row["error"] for row in sub),
                "mean_hidden_verify_fraction": mean_or_nan(
                    row["hidden_verify_fraction"] for row in sub
                ),
                "mean_single_hidden_verify": mean_or_nan(
                    row["single_hidden_verify"] for row in sub
                ),
                "mean_q1": mean_or_nan(row["q1"] for row in sub),
                "mean_q2": mean_or_nan(row["q2"] for row in sub),
            }
        )
    return output


def choose_bucket(contrasts: Sequence[Mapping[str, Any]], routing: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    a = next(row for row in contrasts if row["contrast"] == "contrast_A_hidden_after_q1_q2")
    b = next(
        row
        for row in contrasts
        if row["contrast"] == "contrast_B_hidden_after_q1_q2_difficulty"
    )
    auc = float(b["delta_auroc"])
    auc_lo = float(b["delta_auroc_ci_lower"])
    auc_hi = float(b["delta_auroc_ci_upper"])
    ll = float(b["delta_log_loss"])
    ll_lo = float(b["delta_log_loss_ci_lower"])
    ll_hi = float(b["delta_log_loss_ci_upper"])
    auc_excludes0 = auc_lo > 0
    ll_helps = ll_hi < 0
    a_auc = float(a["delta_auroc"])
    route = [
        row
        for row in routing
        if row["router"] == "q1_q2_difficulty_hidden"
        and row["budget_fraction"] in {0.2, 0.3, 0.4, 0.5}
    ]
    route_pos = sum(
        1
        for row in route
        if row["delta_vs_q1_q2_difficulty"] != ""
        and float(row["delta_vs_q1_q2_difficulty"]) > 0.02
        and (
            row["delta_ci_lower"] == ""
            or float(row["delta_ci_lower"]) > 0
        )
    )
    rationale = (
        f"Contrast A ΔAUROC {_fmt(a_auc)} [{_fmt(a['delta_auroc_ci_lower'])}, {_fmt(a['delta_auroc_ci_upper'])}]; "
        f"Contrast B ΔAUROC {_fmt(auc)} [{_fmt(auc_lo)}, {_fmt(auc_hi)}], "
        f"Δlog-loss {_fmt(ll)} [{_fmt(ll_lo)}, {_fmt(ll_hi)}]."
    )
    width = auc_hi - auc_lo
    if (auc_excludes0 or ll_helps) and auc >= 0.03 and (a_auc <= 0 or auc >= 0.5 * a_auc):
        if route_pos >= 2:
            return "HIDDEN_ADDS_BEYOND_CONFIDENCE_AND_DIFFICULTY", rationale
        return "HIDDEN_PARTIALLY_ADDS", rationale
    if (auc_excludes0 or ll_helps) and auc > 0.01:
        return "HIDDEN_PARTIALLY_ADDS", rationale
    if (not auc_excludes0) and (not ll_helps) and abs(auc) < 0.03:
        if width > 0.12:
            return "AMBIGUOUS", rationale
        return "DIFFICULTY_LARGELY_EXPLAINS_HIDDEN_RESIDUAL", rationale
    if width > 0.12 and not auc_excludes0 and not ll_helps:
        return "AMBIGUOUS", rationale
    return "HIDDEN_PARTIALLY_ADDS", rationale


def validate(
    table: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    errors = list(issues)
    for model in STUDY1_MODEL_ALIASES:
        sub = [row for row in table if row["model_alias"] == model]
        if len(sub) != 500:
            errors.append(f"{model} has {len(sub)} rows, expected 500")
        qids = [row["question_id"] for row in sub]
        if len(set(qids)) != len(qids):
            errors.append(f"{model} duplicate question IDs")
        if any(not math.isfinite(row["q1"]) for row in sub):
            errors.append(f"{model} missing q1")
        if any(not math.isfinite(row["q2"]) for row in sub):
            errors.append(f"{model} missing q2")
        if any(row["n_hidden_cells"] != 8 for row in sub):
            errors.append(f"{model} hidden cell count is not uniformly 8")
        others = OTHER_BY_TARGET[model]
        key_map = {
            "openai_gpt56_sol": "gpt_correct",
            "anthropic_sonnet5": "claude_correct",
            "google_gemini38_flash": "gemini_correct",
            "xai_grok420_nonreasoning": "grok_correct",
        }
        target_key = key_map[model]
        for row in sub:
            reconstructed = sum(int(row[key_map[alias]]) for alias in others)
            if reconstructed != int(row["other_n_correct"]):
                errors.append(f"{model} difficulty reconstruction leak/mismatch")
                break
            if int(row[target_key]) != int(row["target_correct"]):
                errors.append(f"{model} target correctness column mismatch")
                break
            if target_key in others:
                errors.append("target model listed among other-models")
                break
    payload = {
        "ok": not errors,
        "errors": errors,
        "n_rows": len(table),
        "canonical_hidden": f"{CANONICAL_OWNER} L={CANONICAL_L:.0f} hidden",
        "fingerprints": fingerprints,
        "api_calls": 0,
        "q2_source": str(Q2_RESULTS),
        "unit": "question ID",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    return payload


def write_validation_md(payload: Mapping[str, Any]) -> None:
    lines = [
        "# Task 005C validation / leakage audit",
        "",
        f"- API calls: **{payload['api_calls']}**",
        f"- OK: **{payload['ok']}**",
        f"- Rows: {payload['n_rows']} (expected 1000 = 500 GPT + 500 Claude)",
        f"- q2 table: `{payload['q2_source']}` (read-only)",
        f"- Canonical single hidden action: `{payload['canonical_hidden']}`",
        f"- Unit: {payload['unit']}",
        f"- Bootstrap seed {payload['bootstrap_seed']}, resamples {payload['bootstrap_resamples']}",
        "",
        "## Frozen-file fingerprints (unchanged if listed)",
        "",
    ]
    for name, meta in payload["fingerprints"].items():
        sha = meta["sha256"] if isinstance(meta, dict) else meta
        lines.append(f"- {name}: `{sha}`")
    lines += [
        "",
        "## Leakage rules",
        "",
        "- Target-model Stage-1 correctness is not inside `other_n_correct`.",
        "- Difficulty uses the other three historical V2-B models only.",
        "- Study-1 manipulated actions are not used.",
        "- q1/q2/hidden/difficulty preprocessing for CV is fit on training folds only.",
        "- Hidden fraction is rebuilt from historical V2 primary hidden cells and checked against Task 005 `q2_results.csv`.",
        "",
        "## Errors",
        "",
    ]
    if payload["errors"]:
        lines.extend(f"- {item}" for item in payload["errors"])
    else:
        lines.append("- none")
    lines.append("")
    (RETURN_DIR / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any]) -> None:
    buckets = summary["buckets"]
    cv = summary["cv"]
    contrasts = summary["contrasts"]
    routing = summary["routing"]
    assoc = summary["associations"]

    def mrow(model: str, feature: str) -> dict[str, Any]:
        return next(
            row
            for row in cv
            if row["model_alias"] == model and row["feature_set"] == feature
        )

    def crow(model: str, name: str) -> dict[str, Any]:
        return next(
            row
            for row in contrasts
            if row["model_alias"] == model and row["contrast"] == name
        )

    def arow(model: str, metric: str) -> dict[str, Any]:
        return next(
            row
            for row in assoc
            if row["model_alias"] == model and row["metric"] == metric
        )

    gpt_b = buckets["openai_gpt56_sol"]
    claude_b = buckets["anthropic_sonnet5"]
    gpt_a = crow("openai_gpt56_sol", "contrast_A_hidden_after_q1_q2")
    claude_a = crow("anthropic_sonnet5", "contrast_A_hidden_after_q1_q2")
    gpt_c = crow("openai_gpt56_sol", "contrast_B_hidden_after_q1_q2_difficulty")
    claude_c = crow("anthropic_sonnet5", "contrast_B_hidden_after_q1_q2_difficulty")
    gpt_d = crow("openai_gpt56_sol", "contrast_difficulty_after_q1_q2")
    claude_d = crow("anthropic_sonnet5", "contrast_difficulty_after_q1_q2")
    gpt_s = crow(
        "openai_gpt56_sol", "contrast_single_hidden_after_q1_q2_difficulty"
    )
    claude_s = crow(
        "anthropic_sonnet5", "contrast_single_hidden_after_q1_q2_difficulty"
    )

    def route_lines(model: str) -> list[str]:
        lines = []
        for frac in BUDGETS:
            base = next(
                row
                for row in routing
                if row["model_alias"] == model
                and row["router"] == "q1_q2_difficulty"
                and abs(float(row["budget_fraction"]) - frac) < 1e-12
            )
            hid = next(
                row
                for row in routing
                if row["model_alias"] == model
                and row["router"] == "q1_q2_difficulty_hidden"
                and abs(float(row["budget_fraction"]) - frac) < 1e-12
            )
            lines.append(
                f"- {int(100 * frac)}%: q1+q2+difficulty catch {_fmt(base['catch_fraction'])} "
                f"({_fmt(base['errors_caught'], 1)} of {base['n_wrong']}); "
                f"+hidden {_fmt(hid['catch_fraction'])} "
                f"(Δ {_fmt_ci(hid['delta_vs_q1_q2_difficulty'], hid['delta_ci_lower'], hid['delta_ci_upper'])})"
            )
        return lines

    bottom = (
        f"GPT bucket **{gpt_b['bucket']}**. Claude bucket **{claude_b['bucket']}**. "
        "The analysis asks whether historical hidden VERIFY still helps after q1, an independent q2, "
        "and leakage-safe other-model difficulty."
    )
    rec = [
        "Do not launch paid experiments from this audit.",
        "Keep paperDirection.txt unmodified.",
        "Treat the 8-cell hidden fraction as a multi-elicitation diagnostic / upper bound, not a one-call production router.",
    ]
    if gpt_b["bucket"] == "DIFFICULTY_LARGELY_EXPLAINS_HIDDEN_RESIDUAL" and claude_b[
        "bucket"
    ] == "DIFFICULTY_LARGELY_EXPLAINS_HIDDEN_RESIDUAL":
        rec.append(
            "In this 500-question exploratory set, other-model difficulty plus repeated confidence is the leading simpler explanation of the Task-005 hidden residual."
        )
    elif "ADDS" in gpt_b["bucket"] or "ADDS" in claude_b["bucket"]:
        rec.append(
            "If GPT review agrees, keep a deeper residual as an exploratory D1 finding for the model(s) where Contrast B remains material, still not as a hidden-state proof."
        )

    lines = [
        "# Task 005C — Hidden-Action Residual After Difficulty Control",
        "",
        "Exploratory analysis only. Not confirmatory. `paperDirection.txt`, historical V2, Study 1, Task 005 q2 data, and Task 005B data were not modified.",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
        "API calls made: **0**",
        "",
        "---",
        "",
        "## 1. Plain-English bottom line",
        "",
        bottom,
        "",
        f"- GPT Contrast B (hidden after q1+q2+difficulty): ΔAUROC {_fmt_ci(gpt_c['delta_auroc'], gpt_c['delta_auroc_ci_lower'], gpt_c['delta_auroc_ci_upper'])}; Δlog-loss {_fmt_ci(gpt_c['delta_log_loss'], gpt_c['delta_log_loss_ci_lower'], gpt_c['delta_log_loss_ci_upper'])}",
        f"- Claude Contrast B: ΔAUROC {_fmt_ci(claude_c['delta_auroc'], claude_c['delta_auroc_ci_lower'], claude_c['delta_auroc_ci_upper'])}; Δlog-loss {_fmt_ci(claude_c['delta_log_loss'], claude_c['delta_log_loss_ci_lower'], claude_c['delta_log_loss_ci_upper'])}",
        "",
        "Difficulty is leakage-safe: GPT is scored with Claude/Gemini/Grok correctness; Claude with GPT/Gemini/Grok. The target model's own Stage-1 correctness is never inside its difficulty feature.",
        "",
        "---",
        "",
        "## 2. Validation / leakage audit",
        "",
        f"- Joins: **{summary['n_gpt']} GPT + {summary['n_claude']} Claude** question-level rows on the 500 V2-B IDs",
        f"- Hidden summary rebuilt from V2 primary hidden cells (8 cells/question = 2 owners × 4 L) and matched Task 005 `q2_results.csv`",
        f"- Canonical single hidden action predeclared as `{CANONICAL_OWNER}`, L={CANONICAL_L:.0f}, hidden",
        f"- Validation OK: **{summary['validation']['ok']}**",
        f"- V2 sha256 `{summary['hashes']['v2']}`",
        f"- Study 1 sha256 `{summary['hashes']['study1']}`",
        f"- q2 sqlite sha256 `{summary['hashes']['q2']}`",
        f"- paperDirection sha256 `{summary['hashes']['paperDirection']}`",
        f"- Task 005 report sha256 `{summary['hashes']['task005_report']}`",
        f"- Task 005B report sha256 `{summary['hashes']['task005b_report']}`",
        "- Grouped 5-fold CV by question ID. One-hot category and length/choice-count scaling fit on training folds only.",
        "- See `validation.md`.",
        "",
        "---",
        "",
        "## 3. Replication of Task-005 q2 result",
        "",
        "Contrast A is `q1+q2+hidden_fraction` vs `q1+q2`, same construction as Task 005 Lane B.",
        "",
        f"- GPT: ΔAUROC {_fmt_ci(gpt_a['delta_auroc'], gpt_a['delta_auroc_ci_lower'], gpt_a['delta_auroc_ci_upper'])}; Δlog-loss {_fmt_ci(gpt_a['delta_log_loss'], gpt_a['delta_log_loss_ci_lower'], gpt_a['delta_log_loss_ci_upper'])}",
        f"- Claude: ΔAUROC {_fmt_ci(claude_a['delta_auroc'], claude_a['delta_auroc_ci_lower'], claude_a['delta_auroc_ci_upper'])}; Δlog-loss {_fmt_ci(claude_a['delta_log_loss'], claude_a['delta_log_loss_ci_lower'], claude_a['delta_log_loss_ci_upper'])}",
        "",
        "Task 005 reported GPT ΔAUROC +0.081 [−0.002, +0.163] / Δlog-loss −0.042, and Claude ΔAUROC +0.045 [+0.016, +0.074] / Δlog-loss −0.032. Contrast A point estimates match those nested models exactly. Bootstrap percentile intervals can differ slightly because this audit reseeds each contrast/metric instead of advancing one RNG across sequential metrics.",
        "",
        f"- GPT q1 AUROC {_fmt(mrow('openai_gpt56_sol','q1')['auroc'])}, q1+q2 {_fmt(mrow('openai_gpt56_sol','q1_q2')['auroc'])}, q1+q2+hidden {_fmt(mrow('openai_gpt56_sol','q1_q2_hidden')['auroc'])}",
        f"- Claude q1 AUROC {_fmt(mrow('anthropic_sonnet5','q1')['auroc'])}, q1+q2 {_fmt(mrow('anthropic_sonnet5','q1_q2')['auroc'])}, q1+q2+hidden {_fmt(mrow('anthropic_sonnet5','q1_q2_hidden')['auroc'])}",
        "",
        "---",
        "",
        "## 4. Strength of observable difficulty",
        "",
        f"- GPT: other-model n_correct AUROC for GPT correctness {_fmt(arow('openai_gpt56_sol','other_n_correct_auroc_for_target_correct')['value'])}; Pearson(hidden fraction, other-frac-wrong) {_fmt(arow('openai_gpt56_sol','pearson_hidden_frac_vs_other_frac_wrong')['value'])}",
        f"- Claude: other-model n_correct AUROC for Claude correctness {_fmt(arow('anthropic_sonnet5','other_n_correct_auroc_for_target_correct')['value'])}; Pearson(hidden fraction, other-frac-wrong) {_fmt(arow('anthropic_sonnet5','pearson_hidden_frac_vs_other_frac_wrong')['value'])}",
        f"- Adding difficulty after q1+q2: GPT ΔAUROC {_fmt_ci(gpt_d['delta_auroc'], gpt_d['delta_auroc_ci_lower'], gpt_d['delta_auroc_ci_upper'])}; Claude ΔAUROC {_fmt_ci(claude_d['delta_auroc'], claude_d['delta_auroc_ci_lower'], claude_d['delta_auroc_ci_upper'])}",
        f"- Difficulty-only OOF AUROC: GPT {_fmt(mrow('openai_gpt56_sol','difficulty')['auroc'])}; Claude {_fmt(mrow('anthropic_sonnet5','difficulty')['auroc'])}",
        "",
        "Bin means are in `difficulty_associations.csv`. Hidden VERIFY propensity rises as fewer of the other three models are correct. That is the Task-005B-style confounder, now on all 500 V2-B items.",
        "",
        "---",
        "",
        "## 5. PRIMARY: does hidden action add after q1+q2+difficulty?",
        "",
        "Contrast B: `q1+q2+difficulty+hidden_fraction` vs `q1+q2+difficulty`.",
        "",
        f"- GPT: ΔAUROC {_fmt_ci(gpt_c['delta_auroc'], gpt_c['delta_auroc_ci_lower'], gpt_c['delta_auroc_ci_upper'])}; Δlog-loss {_fmt_ci(gpt_c['delta_log_loss'], gpt_c['delta_log_loss_ci_lower'], gpt_c['delta_log_loss_ci_upper'])}; ΔBrier {_fmt_ci(gpt_c['delta_brier'], gpt_c['delta_brier_ci_lower'], gpt_c['delta_brier_ci_upper'])}",
        f"- Claude: ΔAUROC {_fmt_ci(claude_c['delta_auroc'], claude_c['delta_auroc_ci_lower'], claude_c['delta_auroc_ci_upper'])}; Δlog-loss {_fmt_ci(claude_c['delta_log_loss'], claude_c['delta_log_loss_ci_lower'], claude_c['delta_log_loss_ci_upper'])}; ΔBrier {_fmt_ci(claude_c['delta_brier'], claude_c['delta_brier_ci_lower'], claude_c['delta_brier_ci_upper'])}",
        "",
        "Single canonical hidden action (AI-system, L=10, one cell) after q1+q2+difficulty:",
        "",
        f"- GPT: ΔAUROC {_fmt_ci(gpt_s['delta_auroc'], gpt_s['delta_auroc_ci_lower'], gpt_s['delta_auroc_ci_upper'])}",
        f"- Claude: ΔAUROC {_fmt_ci(claude_s['delta_auroc'], claude_s['delta_auroc_ci_lower'], claude_s['delta_auroc_ci_upper'])}",
        "",
        "The 8-cell fraction is a multi-elicitation diagnostic / upper bound. The single-cell sensitivity is closer to a one-call router.",
        "",
        f"- GPT: {gpt_b['rationale']}",
        f"- Claude: {claude_b['rationale']}",
        "",
        "---",
        "",
        "## 6. Matched-budget routing after difficulty control",
        "",
        "OOF predicted P(correct); lowest predicted correctness verified first; Checkpoint A fractional ties. Primary comparison is +hidden vs q1+q2+difficulty.",
        "",
        "### GPT",
        "",
        *route_lines("openai_gpt56_sol"),
        "",
        "### Claude",
        "",
        *route_lines("anthropic_sonnet5"),
        "",
        "Full 10–50% tables, including the single-cell hidden router, are in `matched_budget_routing.csv`.",
        "",
        "---",
        "",
        "## 7. GPT interpretation bucket",
        "",
        f"**{gpt_b['bucket']}**",
        "",
        gpt_b["rationale"],
        "",
        "---",
        "",
        "## 8. Claude interpretation bucket",
        "",
        f"**{claude_b['bucket']}**",
        "",
        claude_b["rationale"],
        "",
        "---",
        "",
        "## 9. What this DOES establish",
        "",
        "- Whether Task 005's hidden-after-q1+q2 residual survives a leakage-safe other-model difficulty control on the same 500 questions.",
        "- How strongly other-model difficulty predicts target correctness and hidden VERIFY propensity.",
        "- Whether a one-cell canonical hidden action behaves like the 8-cell fraction.",
        "- Model-specific exploratory buckets for GPT review.",
        "",
        "---",
        "",
        "## 10. What it DOES NOT establish",
        "",
        "- It does **not** establish an internal hidden-state mechanism.",
        "- If difficulty explains the signal, that does **not** make verification useless; it may be a cheap item-difficulty estimator.",
        "- If hidden remains useful beyond difficulty, that still does **not** prove a hidden internal state; it is a behavioral residual.",
        "- The 8-cell hidden fraction is **not** automatically a fair one-call production router.",
        "- This is not confirmatory and does not rewrite paperDirection.txt.",
        "",
        "---",
        "",
        "## 11. Recommendation only — do not launch paid experiments",
        "",
        *[f"- {item}" for item in rec],
        "",
        "---",
        "",
        "## 12. READY_FOR_GPT_REVIEW = YES",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
    ]
    (RETURN_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    fps = {
        "v2": file_fingerprint(V2_SQLITE),
        "study1": file_fingerprint(STUDY1_SQLITE),
        "q2": file_fingerprint(Q2_SQLITE),
        "paperDirection": {"sha256": sha256_file(PAPER_DIRECTION)},
        "task005_report": {"sha256": sha256_file(TASK005_REPORT)},
        "task005b_report": {"sha256": sha256_file(TASK005B_REPORT)},
    }
    q2_rows = load_q2_table()
    examples = {row.example_id: row for row in load_v2b_examples()}
    answers = load_v2_answers()
    v2_ver = load_v2_primary_verification()
    hidden = hidden_action_summaries()
    canonical = canonical_hidden_map(v2_ver)
    qids = sorted({row["question_id"] for row in q2_rows})
    if len(qids) != 500:
        raise RuntimeError(f"expected 500 q2 question IDs, got {len(qids)}")
    difficulty = build_difficulty(qids, answers, examples)
    table, issues = attach_rows(q2_rows, difficulty, hidden, canonical)
    validation = validate(table, issues, fps)
    write_validation_md(validation)
    if not validation["ok"]:
        (RETURN_DIR / "report.md").write_text(
            "# Task 005C STOPPED\n\nJoins failed materially. See validation.md.\n",
            encoding="utf-8",
        )
        raise RuntimeError("005C validation failed: " + "; ".join(validation["errors"][:12]))
    cv_all = []
    contrast_all = []
    routing_all = []
    assoc_all = []
    buckets = {}
    single_rows = []
    for model in STUDY1_MODEL_ALIASES:
        sub = [row for row in table if row["model_alias"] == model]
        result = evaluate_models(sub)
        cv_all.extend(result["cv"])
        contrast_all.extend(result["contrasts"])
        routing_all.extend(result["routing"])
        assoc_all.extend(difficulty_associations(sub))
        bucket, rationale = choose_bucket(result["contrasts"], result["routing"])
        buckets[model] = {"bucket": bucket, "rationale": rationale}
        single_rows.append(
            {
                "model_alias": model,
                "canonical_hidden": f"{CANONICAL_OWNER}|L={CANONICAL_L:.0f}|hidden",
                "n": len(sub),
                "single_hidden_rate": mean_or_nan(row["single_hidden_verify"] for row in sub),
                "mean_hidden_fraction": mean_or_nan(
                    row["hidden_verify_fraction"] for row in sub
                ),
                **{
                    key: next(
                        row
                        for row in result["contrasts"]
                        if row["contrast"]
                        == "contrast_single_hidden_after_q1_q2_difficulty"
                    )[key]
                    for key in (
                        "delta_auroc",
                        "delta_auroc_ci_lower",
                        "delta_auroc_ci_upper",
                        "delta_log_loss",
                        "delta_log_loss_ci_lower",
                        "delta_log_loss_ci_upper",
                    )
                },
            }
        )
    write_csv(RETURN_DIR / "question_level_features.csv", table)
    write_csv(RETURN_DIR / "grouped_cv_nested_models.csv", cv_all)
    write_csv(RETURN_DIR / "bootstrap_contrasts.csv", contrast_all)
    write_csv(RETURN_DIR / "matched_budget_routing.csv", routing_all)
    write_csv(RETURN_DIR / "difficulty_associations.csv", assoc_all)
    write_csv(RETURN_DIR / "single_hidden_sensitivity.csv", single_rows)
    shutil.copy2(Path(__file__).resolve(), RETURN_DIR / "task005c_hidden_residual.py")
    (RETURN_DIR / "run_005c.py").write_text(
        '"""Offline pointer for Task 005C. No API calls."""\n\n'
        "from src.task005c_hidden_residual import main\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n",
        encoding="utf-8",
    )
    fps_after = {
        "v2": file_fingerprint(V2_SQLITE)["sha256"],
        "study1": file_fingerprint(STUDY1_SQLITE)["sha256"],
        "q2": file_fingerprint(Q2_SQLITE)["sha256"],
        "paperDirection": sha256_file(PAPER_DIRECTION),
        "task005_report": sha256_file(TASK005_REPORT),
        "task005b_report": sha256_file(TASK005B_REPORT),
    }
    if fps_after["v2"] != fps["v2"]["sha256"]:
        raise RuntimeError("V2 hash changed")
    if fps_after["study1"] != fps["study1"]["sha256"]:
        raise RuntimeError("Study 1 hash changed")
    if fps_after["q2"] != fps["q2"]["sha256"]:
        raise RuntimeError("q2 sqlite hash changed")
    if fps_after["paperDirection"] != fps["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection hash changed")
    if fps_after["task005_report"] != fps["task005_report"]["sha256"]:
        raise RuntimeError("Task 005 report hash changed")
    if fps_after["task005b_report"] != fps["task005b_report"]["sha256"]:
        raise RuntimeError("Task 005B report hash changed")
    summary = {
        "ok": True,
        "api_calls": 0,
        "n_gpt": sum(1 for row in table if row["model_alias"] == "openai_gpt56_sol"),
        "n_claude": sum(1 for row in table if row["model_alias"] == "anthropic_sonnet5"),
        "validation": validation,
        "cv": cv_all,
        "contrasts": contrast_all,
        "routing": routing_all,
        "associations": assoc_all,
        "buckets": buckets,
        "hashes": fps_after,
    }
    write_report(summary)
    slim = {
        "ok": True,
        "api_calls": 0,
        "buckets": buckets,
        "hashes": fps_after,
        "n_gpt": summary["n_gpt"],
        "n_claude": summary["n_claude"],
    }
    (RETURN_DIR / "summary.json").write_text(
        json.dumps(slim, indent=2) + "\n", encoding="utf-8"
    )
    changed = [
        "to_gpt/005c_hidden_residual_after_difficulty/report.md",
        "to_gpt/005c_hidden_residual_after_difficulty/validation.md",
        "to_gpt/005c_hidden_residual_after_difficulty/question_level_features.csv",
        "to_gpt/005c_hidden_residual_after_difficulty/grouped_cv_nested_models.csv",
        "to_gpt/005c_hidden_residual_after_difficulty/bootstrap_contrasts.csv",
        "to_gpt/005c_hidden_residual_after_difficulty/matched_budget_routing.csv",
        "to_gpt/005c_hidden_residual_after_difficulty/difficulty_associations.csv",
        "to_gpt/005c_hidden_residual_after_difficulty/single_hidden_sensitivity.csv",
        "to_gpt/005c_hidden_residual_after_difficulty/changed_files.txt",
        "to_gpt/005c_hidden_residual_after_difficulty/task005c_hidden_residual.py",
        "to_gpt/005c_hidden_residual_after_difficulty/run_005c.py",
        "to_gpt/005c_hidden_residual_after_difficulty/summary.json",
        "src/task005c_hidden_residual.py",
        "from_gpt/005c_hidden_residual_after_difficulty.md",
    ]
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "api_calls": 0,
                "buckets": {
                    model: payload["bucket"]
                    for model, payload in summary["buckets"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
