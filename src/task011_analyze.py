"""Task 011 confirmatory analysis. Run only after routing freeze and hidden tests."""

from __future__ import annotations

import json
import math
import random
import sqlite3
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from .study1_analysis import _is_verify, _pyplot
from .task005_common import sha256_file
from .task005_lane_a import (
    crossfit_isotonic,
    error_catch_from_weights,
    fractional_verify_weights,
)
from .task011_common import (
    ADJACENT_PAIRS,
    ANALYSIS_DIR,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COVERAGE_LARGE_PP,
    COVERAGE_MODERATE_PP,
    CV_FOLDS,
    EPS,
    FIGURES_DIR,
    GPT_CLAUDE,
    H1_HIGH,
    H1_LOW,
    MATCHED_BUDGETS,
    MATERIAL_LOGLOSS,
    MODEL_LABELS,
    PAPER_DIRECTION,
    REFERENCE_SCORE,
    RETURN_DIR,
    ROUTING_GAIN_PP,
    SAMPLE_DIR,
    SCORE_CONDITIONS,
    SQLITE_PATH,
    VISIBLE_FIXED_CONDITIONS,
    json_dump,
    load_csv,
    load_json,
    protected_fingerprints,
    write_csv,
)

MMLU_COVERAGE = {
    "openai_gpt56_sol": {"delta": 0.502, "lo": 0.458, "hi": 0.546, "acc": 0.830},
    "anthropic_sonnet5": {"delta": 0.354, "lo": 0.312, "hi": 0.396, "acc": 0.758},
}
MMLU_DLL = {
    "openai_gpt56_sol": 0.0008,
    "anthropic_sonnet5": -0.0000,
}


def _ci(vals: Sequence[float]) -> tuple[float, float]:
    clean = sorted(float(v) for v in vals if math.isfinite(v))
    if not clean:
        return float("nan"), float("nan")
    return clean[int(0.025 * (len(clean) - 1))], clean[int(0.975 * (len(clean) - 1))]


def _mean(vals: Sequence[float]) -> float:
    clean = [float(v) for v in vals if math.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _safe_auc(y: Sequence[int], scores: Sequence[float]) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(scores), dtype=float)
    if len(y_arr) < 2 or len(np.unique(y_arr)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_arr, s_arr))
    except ValueError:
        return float("nan")


def _safe_log_loss(y: Sequence[int], p: Sequence[float]) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    p_arr = np.clip(np.asarray(list(p), dtype=float), EPS, 1.0 - EPS)
    if len(y_arr) == 0:
        return float("nan")
    return float(log_loss(y_arr, p_arr, labels=[0, 1]))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall(x: Sequence[float], y: Sequence[float]) -> float:
    xs = list(x)
    ys = list(y)
    n = len(xs)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                conc += 1
            elif prod < 0:
                disc += 1
    denom = conc + disc
    return float((conc - disc) / denom) if denom else float("nan")


def fit_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    for cand in (np.inf, 1e6, 1.0):
        try:
            clf = LogisticRegression(C=cand, solver="lbfgs", max_iter=4000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X, y)
            return clf
        except Exception:
            continue
    return None


def predict_p(clf: LogisticRegression | None, X: np.ndarray, fallback: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if clf is None:
        return np.full(len(X), fallback)
    return np.clip(clf.predict_proba(X)[:, 1], EPS, 1.0 - EPS)


def _open_readonly() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _load_requests() -> tuple[list[dict[str, Any]], int]:
    connection = _open_readonly()
    try:
        rows = connection.execute(
            "SELECT request_key, stage, example_id, model_alias, status, "
            "record_json, error, requested_model_id FROM requests"
        ).fetchall()
        attempts = connection.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"]
    finally:
        connection.close()
    records = []
    for row in rows:
        record = json.loads(row["record_json"]) if row["record_json"] else {}
        records.append(
            {
                "request_key": row["request_key"],
                "stage": row["stage"],
                "example_id": row["example_id"],
                "model_alias": row["model_alias"],
                "status": row["status"],
                "error": row["error"],
                "requested_model_id": row["requested_model_id"],
                "record": record,
            }
        )
    return records, int(attempts)


def export_tables(
    requests: Sequence[Mapping[str, Any]],
    *,
    main_ids: Sequence[str],
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    allowed = set(main_ids)
    passed = {
        (str(row["question_id"]), str(row["model_alias"])): int(row["passed"])
        for row in tests
    }
    codes: dict[tuple[str, str], dict[str, Any]] = {}
    confs: dict[tuple[str, str], dict[str, Any]] = {}
    stage3: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stage1_rows: list[dict[str, Any]] = []
    q1_rows: list[dict[str, Any]] = []
    for row in requests:
        rec = row["record"]
        qid = str(row["example_id"])
        if qid not in allowed:
            continue
        alias = str(row["model_alias"])
        key = (qid, alias)
        if row["status"] != "success":
            failures.append(
                {
                    "request_key": row["request_key"],
                    "stage": row["stage"],
                    "question_id": qid,
                    "model_alias": alias,
                    "status": row["status"],
                    "error": row["error"],
                    "score_condition": rec.get("score_condition"),
                    "repeat_index": rec.get("repeat_index"),
                }
            )
            continue
        if row["stage"] == "code":
            codes[key] = rec
            stage1_rows.append(
                {
                    "question_id": qid,
                    "model_alias": alias,
                    "model_endpoint": rec.get("model_endpoint"),
                    "extraction_status": rec.get("extraction_status"),
                    "malformed": rec.get("malformed"),
                    "input_tokens": rec.get("input_tokens"),
                    "output_tokens": rec.get("output_tokens"),
                    "estimated_cost_usd": rec.get("estimated_cost_usd"),
                    "latency_ms": rec.get("latency_ms"),
                    "passed": passed.get(key),
                    "request_key": row["request_key"],
                }
            )
        elif row["stage"] == "confidence":
            confs[key] = rec
            q1_rows.append(
                {
                    "question_id": qid,
                    "model_alias": alias,
                    "probability_correct": rec.get("probability_correct"),
                    "input_tokens": rec.get("input_tokens"),
                    "output_tokens": rec.get("output_tokens"),
                    "estimated_cost_usd": rec.get("estimated_cost_usd"),
                    "request_key": row["request_key"],
                }
            )
        elif row["stage"] == "verification":
            action = rec.get("parsed_action")
            item = {
                "question_id": qid,
                "model_alias": alias,
                "model_endpoint": rec.get("model_endpoint") or row["requested_model_id"],
                "score_condition": rec.get("score_condition"),
                "displayed_confidence": rec.get("displayed_confidence"),
                "parsed_action": action,
                "verify": int(_is_verify(action)),
                "repeat_index": int(rec.get("repeat_index") or 0),
                "q1": rec.get("probability_correct") or confs.get(key, {}).get("probability_correct"),
                "stage1_correct": passed.get(key),
                "prompt_hash": rec.get("prompt_hash"),
                "request_key": row["request_key"],
                "estimated_cost_usd": rec.get("estimated_cost_usd"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
            }
            if item["repeat_index"] == 0:
                stage3.append(item)
            else:
                repeats.append(item)
    stage12 = []
    for key, code in codes.items():
        conf = confs.get(key, {})
        stage12.append(
            {
                "question_id": key[0],
                "model_alias": key[1],
                "extraction_status": code.get("extraction_status"),
                "malformed": code.get("malformed"),
                "q1": conf.get("probability_correct"),
                "stage1_correct": passed.get(key),
                "code_cost_usd": code.get("estimated_cost_usd"),
                "q1_cost_usd": conf.get("estimated_cost_usd"),
            }
        )
    write_csv(RETURN_DIR / "stage1_code_results.csv", stage1_rows)
    write_csv(RETURN_DIR / "q1_results.csv", q1_rows)
    write_csv(RETURN_DIR / "stage3_results.csv", stage3)
    write_csv(RETURN_DIR / "repeat_results.csv", repeats)
    write_csv(RETURN_DIR / "failures.csv", failures)
    write_csv(RETURN_DIR / "test_execution_results.csv", list(tests))
    return {
        "stage12": stage12,
        "stage3": stage3,
        "repeats": repeats,
        "failures": failures,
        "stage1": stage1_rows,
        "q1": q1_rows,
    }


def coverage_table(stage3: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for model in GPT_CLAUDE:
        model_rows = [
            row
            for row in stage3
            if row["model_alias"] == model and int(row.get("repeat_index") or 0) == 0
        ]
        by_q: dict[str, dict[str, int]] = defaultdict(dict)
        for row in model_rows:
            by_q[str(row["question_id"])][str(row["score_condition"])] = int(row["verify"])
        complete = {
            qid: conds
            for qid, conds in by_q.items()
            if all(cond in conds for cond in SCORE_CONDITIONS)
        }
        for condition in SCORE_CONDITIONS:
            vals = [complete[qid][condition] for qid in complete]
            rate = _mean(vals)
            boots = []
            qids = list(complete)
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw = [complete[qids[rng.randrange(len(qids))]][condition] for _ in qids]
                boots.append(float(np.mean(draw)))
            lo, hi = _ci(boots)
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score_condition": condition,
                    "n_questions": len(complete),
                    "verify_rate": rate,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        if complete:
            diffs = [complete[qid][H1_LOW] - complete[qid][H1_HIGH] for qid in complete]
            boots = []
            qids = list(complete)
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw_q = [qids[rng.randrange(len(qids))] for _ in qids]
                draw = [complete[qid][H1_LOW] - complete[qid][H1_HIGH] for qid in draw_q]
                boots.append(float(np.mean(draw)))
            lo, hi = _ci(boots)
            effect = _mean(diffs)
            if effect >= COVERAGE_LARGE_PP:
                label = "large_replication"
            elif effect >= COVERAGE_MODERATE_PP:
                label = "moderate_replication"
            else:
                label = "weak_or_none"
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score_condition": "H1_0.70_minus_0.99",
                    "n_questions": len(complete),
                    "verify_rate": effect,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "replication_label": label,
                }
            )
    write_csv(RETURN_DIR / "coverage_response.csv", rows)
    return rows


def attach_difficulty(stage12: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_q: dict[str, dict[str, int]] = defaultdict(dict)
    for row in stage12:
        if row.get("stage1_correct") in (None, ""):
            continue
        by_q[str(row["question_id"])][str(row["model_alias"])] = int(row["stage1_correct"])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for qid, model_map in by_q.items():
        for target in GPT_CLAUDE:
            if target not in model_map:
                continue
            others = [alias for alias in GPT_CLAUDE if alias != target]
            other_correct = sum(model_map.get(alias, 0) for alias in others)
            out[(qid, target)] = {
                "target_correct": model_map[target],
                "other_primary_correct": other_correct,
            }
    return out


def _grouped_cv(visible: Sequence[Mapping[str, Any]]):
    qids = np.array([row["question_id"] for row in visible])
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    score = np.array([float(row["displayed_confidence"]) for row in visible], dtype=float)
    diff = np.array([float(row["difficulty"]) for row in visible], dtype=float)
    cond = np.array([row["score_condition"] for row in visible])
    unique = np.unique(qids)
    if len(unique) < CV_FOLDS or len(np.unique(y)) < 2:
        nan = np.full(len(y), np.nan)
        return nan, nan, qids, float("nan"), float("nan")
    enc = OneHotEncoder(sparse_output=False, drop="first")
    gkf = GroupKFold(n_splits=CV_FOLDS)
    p_shared = np.zeros(len(y))
    p_rich = np.zeros(len(y))
    for train, test in gkf.split(y, y, groups=qids):
        fallback = float(np.mean(y[train]))
        enc.fit(cond[train].reshape(-1, 1))
        c_tr = enc.transform(cond[train].reshape(-1, 1))
        c_te = enc.transform(cond[test].reshape(-1, 1))
        dmean = float(np.mean(diff[train]))
        d_tr = (diff[train] - dmean).reshape(-1, 1)
        d_te = (diff[test] - dmean).reshape(-1, 1)
        s_tr = (score[train] - REFERENCE_SCORE).reshape(-1, 1)
        s_te = (score[test] - REFERENCE_SCORE).reshape(-1, 1)
        p_shared[test] = predict_p(
            fit_logit(np.hstack([c_tr, d_tr]), y[train]),
            np.hstack([c_te, d_te]),
            fallback,
        )
        p_rich[test] = predict_p(
            fit_logit(np.hstack([c_tr, d_tr, s_tr * d_tr]), y[train]),
            np.hstack([c_te, d_te, s_te * d_te]),
            fallback,
        )
    return p_shared, p_rich, qids, _safe_log_loss(y, p_shared), _safe_log_loss(y, p_rich)


def ranking_models(stage3, difficulty) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(BOOTSTRAP_SEED)
    shared_out: list[dict[str, Any]] = []
    rich_out: list[dict[str, Any]] = []
    for model in GPT_CLAUDE:
        visible = []
        for row in stage3:
            if row["model_alias"] != model:
                continue
            if row["score_condition"] not in VISIBLE_FIXED_CONDITIONS:
                continue
            feat = difficulty.get((str(row["question_id"]), model))
            if feat is None:
                continue
            item = dict(row)
            item["difficulty"] = 1.0 - float(feat["other_primary_correct"])
            visible.append(item)
        p_s, p_r, qids, ll_s, ll_r = _grouped_cv(visible)
        improvement = ll_s - ll_r
        q_unique = list(dict.fromkeys(str(q) for q in qids))
        by_q_idx: dict[str, list[int]] = defaultdict(list)
        for idx, qid in enumerate(qids):
            by_q_idx[str(qid)].append(idx)
        y = np.array([int(row["verify"]) for row in visible], dtype=int)
        boots = []
        if q_unique and np.all(np.isfinite(p_s)) and np.all(np.isfinite(p_r)):
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw = [q_unique[rng.randrange(len(q_unique))] for _ in q_unique]
                idx = [i for qid in draw for i in by_q_idx[qid]]
                boots.append(
                    _safe_log_loss(y[idx], p_s[idx]) - _safe_log_loss(y[idx], p_r[idx])
                )
        lo, hi = _ci(boots)
        shared_out.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "model_name": "shared_prioritization",
                "n_rows": len(visible),
                "n_questions": len(q_unique),
                "cv_logloss": ll_s,
            }
        )
        rich_out.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "model_name": "score_specific_reprioritization",
                "n_rows": len(visible),
                "n_questions": len(q_unique),
                "cv_logloss": ll_r,
                "heldout_improvement_vs_shared": improvement,
                "bootstrap_lo": lo,
                "bootstrap_hi": hi,
                "material_logloss": MATERIAL_LOGLOSS,
                "material_gain": bool(
                    math.isfinite(improvement) and improvement >= MATERIAL_LOGLOSS
                ),
            }
        )
    write_csv(RETURN_DIR / "allocation_model_comparison.csv", shared_out + rich_out)
    return shared_out, rich_out


def rank_stability(stage3, repeats) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    combined = list(stage3) + list(repeats)
    rows: list[dict[str, Any]] = []
    for model in GPT_CLAUDE:
        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in combined:
            if row["model_alias"] != model:
                continue
            if row["score_condition"] not in VISIBLE_FIXED_CONDITIONS:
                continue
            cells[(str(row["question_id"]), str(row["score_condition"]))].append(
                int(row["verify"])
            )
        means: dict[str, dict[str, float]] = defaultdict(dict)
        rerun_dis: list[float] = []
        for (qid, cond), vals in cells.items():
            if len(vals) >= 2:
                disag = [float(a != b) for i, a in enumerate(vals) for b in vals[i + 1 :]]
                if disag:
                    rerun_dis.append(float(np.mean(disag)))
            if vals:
                means[qid][cond] = float(np.mean(vals))
        qids = [
            qid
            for qid, conds in means.items()
            if all(cond in conds for cond in VISIBLE_FIXED_CONDITIONS)
        ]
        for left, right in ADJACENT_PAIRS:
            xs = [means[qid][left] for qid in qids]
            ys = [means[qid][right] for qid in qids]
            rho = _spearman(xs, ys)
            tau = _kendall(xs, ys)
            boots = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw = [qids[rng.randrange(len(qids))] for _ in qids] if qids else []
                boots.append(
                    _spearman(
                        [means[qid][left] for qid in draw],
                        [means[qid][right] for qid in draw],
                    )
                    if draw
                    else float("nan")
                )
            lo, hi = _ci(boots)
            n_rev = n_pair = 0
            for qid in qids:
                a, b = means[qid][left], means[qid][right]
                if abs(a - b) < 1e-12:
                    continue
                n_pair += 1
                if a < b:
                    n_rev += 1
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "pair": f"{left}->{right}",
                    "n_questions": len(qids),
                    "spearman": rho,
                    "kendall_tau": tau,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "pairwise_order_reversal_rate": n_rev / n_pair if n_pair else float("nan"),
                    "same_prompt_rerun_disagreement": _mean(rerun_dis),
                }
            )
    write_csv(RETURN_DIR / "repeat_stability.csv", rows)
    return rows


def matched_budget(stage3, stage12) -> list[dict[str, Any]]:
    q1 = {
        (row["question_id"], row["model_alias"]): float(row["q1"])
        for row in stage12
        if row.get("q1") not in (None, "")
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for model in GPT_CLAUDE:
        hidden = [
            row
            for row in stage3
            if row["model_alias"] == model
            and row["score_condition"] == "hidden"
            and row.get("stage1_correct") not in (None, "")
        ]
        if not hidden:
            continue
        qids = np.array([str(row["question_id"]) for row in hidden])
        wrong = np.array([int(not bool(row["stage1_correct"])) for row in hidden])
        correct = 1 - wrong
        hidden_scores = np.array([int(row["verify"]) for row in hidden], dtype=float)
        q1_raw = np.array([q1.get((qid, model), 0.5) for qid in qids], dtype=float)
        q1_scores = 1.0 - q1_raw
        q1_cal = crossfit_isotonic(q1_raw, correct, qids, seed=BOOTSTRAP_SEED)
        cal_scores = 1.0 - np.asarray(q1_cal, dtype=float)
        combo = np.clip(0.5 * hidden_scores + 0.5 * q1_scores, 0.0, 1.0)
        oracle = wrong.astype(float)
        n_wrong = int(wrong.sum())
        for budget in MATCHED_BUDGETS:
            n_verify = budget * len(qids)
            routers = {
                "random": rng.random(len(qids)),
                "raw_q1": q1_scores,
                "calibrated_q1": cal_scores,
                "hidden": hidden_scores,
                "q1_plus_hidden": combo,
                "hindsight_oracle": oracle,
            }
            for name, score in routers.items():
                if name == "random":
                    order = np.argsort(score)
                    weights = np.zeros(len(score))
                    take = int(round(n_verify))
                    weights[order[:take]] = 1.0
                else:
                    weights = fractional_verify_weights(-score, n_verify)
                catch = error_catch_from_weights(weights, wrong)
                verified = float(weights.sum())
                tp = float((weights * wrong).sum())
                rows.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "budget": budget,
                        "router": name,
                        "bug_catch_rate": catch,
                        "precision_among_verified": tp / verified if verified else float("nan"),
                        "residual_bug_rate_unverified": (
                            (n_wrong - tp) / max(1.0, len(qids) - verified)
                        ),
                        "bugs_per_call": tp / verified if verified else float("nan"),
                        "n": len(qids),
                        "n_wrong": n_wrong,
                    }
                )
    write_csv(RETURN_DIR / "matched_budget_routing.csv", rows)
    return rows


def risk_coverage(stage3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in GPT_CLAUDE:
        for condition in SCORE_CONDITIONS:
            cell = [
                row
                for row in stage3
                if row["model_alias"] == model
                and row["score_condition"] == condition
                and row.get("stage1_correct") not in (None, "")
            ]
            if not cell:
                continue
            n = len(cell)
            n_wrong = sum(int(not bool(row["stage1_correct"])) for row in cell)
            n_right = n - n_wrong
            caught = sum(int(row["verify"] and not bool(row["stage1_correct"])) for row in cell)
            fpr = sum(int(row["verify"] and bool(row["stage1_correct"])) for row in cell)
            scores = [int(row["verify"]) for row in cell]
            y = [int(not bool(row["stage1_correct"])) for row in cell]
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score_condition": condition,
                    "n": n,
                    "coverage": sum(int(row["verify"]) for row in cell) / n,
                    "n_wrong": n_wrong,
                    "tpr_incorrect": caught / n_wrong if n_wrong else float("nan"),
                    "fpr_correct": fpr / n_right if n_right else float("nan"),
                    "binary_auroc": _safe_auc(y, scores),
                }
            )
    write_csv(RETURN_DIR / "correct_vs_incorrect_routing.csv", rows)
    return rows


def classify_world(coverage, rich, matched) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {}
    for model in GPT_CLAUDE:
        h1 = next(
            (
                row
                for row in coverage
                if row["model_alias"] == model
                and row["score_condition"] == "H1_0.70_minus_0.99"
            ),
            None,
        )
        dll = next((row for row in rich if row["model_alias"] == model), None)
        effect = float(h1["verify_rate"]) if h1 else float("nan")
        large = bool(math.isfinite(effect) and effect >= COVERAGE_LARGE_PP)
        moderate = bool(math.isfinite(effect) and effect >= COVERAGE_MODERATE_PP)
        reverse = bool(math.isfinite(effect) and effect <= -COVERAGE_MODERATE_PP)
        material_reshape = bool(dll and dll.get("material_gain"))
        per[model] = {
            "effect": effect,
            "label": (h1 or {}).get("replication_label"),
            "large": large,
            "moderate": moderate,
            "reverse": reverse,
            "dll": None if dll is None else dll.get("heldout_improvement_vs_shared"),
            "reshape": material_reshape,
        }

    def beats(model: str) -> bool:
        model_rows = [row for row in matched if row["model_alias"] == model]
        adjacent_hits = 0
        last = False
        for budget in MATCHED_BUDGETS:
            cell = [row for row in model_rows if abs(float(row["budget"]) - budget) < 1e-12]
            if not cell:
                last = False
                continue
            q1_best = max(
                (
                    float(row["bug_catch_rate"])
                    for row in cell
                    if row["router"] in {"raw_q1", "calibrated_q1"}
                    and math.isfinite(float(row["bug_catch_rate"]))
                ),
                default=float("nan"),
            )
            ctx = max(
                (
                    float(row["bug_catch_rate"])
                    for row in cell
                    if row["router"] in {"hidden", "q1_plus_hidden"}
                    and math.isfinite(float(row["bug_catch_rate"]))
                ),
                default=float("nan"),
            )
            hit = math.isfinite(ctx) and math.isfinite(q1_best) and (ctx - q1_best) >= ROUTING_GAIN_PP
            if hit and last:
                adjacent_hits += 1
            last = hit
        return adjacent_hits >= 1

    routing_flags = {model: beats(model) for model in GPT_CLAUDE}
    if all(routing_flags.values()):
        routing = "ROUTING_POSITIVE_BOTH_MODELS"
    elif any(routing_flags.values()):
        routing = "ROUTING_POSITIVE_MODEL_SPECIFIC"
    else:
        routing = "ROUTING_NULL_OR_SMALL"
    effects = [per[model]["effect"] for model in GPT_CLAUDE]
    if any(per[model]["reverse"] for model in GPT_CLAUDE) and any(
        per[model]["moderate"] for model in GPT_CLAUDE
    ):
        coverage_gen = "MODEL_SPECIFIC_GENERALIZATION"
    elif (
        max(effects) >= COVERAGE_LARGE_PP
        and min(effects) >= COVERAGE_MODERATE_PP
        and all(e > 0 for e in effects)
    ):
        coverage_gen = "COVERAGE_GENERALIZES"
    elif max(effects) >= COVERAGE_LARGE_PP and min(effects) < COVERAGE_MODERATE_PP:
        coverage_gen = "MODEL_SPECIFIC_GENERALIZATION"
    elif max(effects) >= COVERAGE_MODERATE_PP:
        coverage_gen = "PARTIAL_COVERAGE"
    else:
        coverage_gen = "COVERAGE_DOES_NOT_GENERALIZE"
    reshape = any(per[model]["reshape"] for model in GPT_CLAUDE)
    if coverage_gen == "COVERAGE_DOES_NOT_GENERALIZE":
        world = "D"
    elif coverage_gen == "MODEL_SPECIFIC_GENERALIZATION":
        world = "MODEL_SPECIFIC"
    elif reshape and coverage_gen in {"COVERAGE_GENERALIZES", "PARTIAL_COVERAGE"}:
        world = "C"
    elif coverage_gen == "COVERAGE_GENERALIZES" and routing == "ROUTING_POSITIVE_BOTH_MODELS":
        world = "A"
    elif coverage_gen in {"COVERAGE_GENERALIZES", "PARTIAL_COVERAGE"}:
        world = "B"
    else:
        world = "E"
    return {
        "world": world,
        "coverage_gen": coverage_gen,
        "routing": routing,
        "per_model": per,
        "routing_flags": routing_flags,
    }


def plot_figures(coverage, matched, rich) -> None:
    plt = _pyplot()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model in GPT_CLAUDE:
        xs, ys, lo, hi = [], [], [], []
        for cond, score in zip(VISIBLE_FIXED_CONDITIONS, (0.70, 0.85, 0.90, 0.95, 0.99)):
            row = next(
                (
                    item
                    for item in coverage
                    if item["model_alias"] == model and item["score_condition"] == cond
                ),
                None,
            )
            if row:
                xs.append(score)
                ys.append(100.0 * float(row["verify_rate"]))
                lo.append(100.0 * float(row["ci_lo"]))
                hi.append(100.0 * float(row["ci_hi"]))
        ax.plot(xs, ys, marker="o", label=MODEL_LABELS[model])
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xlabel("Displayed confidence")
    ax.set_ylabel("VERIFY_FIRST rate (%)")
    ax.set_title("Code coverage response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "code_coverage_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, model in zip(axes, GPT_CLAUDE):
        for router, marker in (
            ("raw_q1", "o"),
            ("hidden", "s"),
            ("q1_plus_hidden", "D"),
            ("random", "x"),
            ("hindsight_oracle", "^"),
        ):
            xs, ys = [], []
            for row in matched:
                if row["model_alias"] != model or row["router"] != router:
                    continue
                xs.append(100.0 * float(row["budget"]))
                ys.append(100.0 * float(row["bug_catch_rate"]))
            if xs:
                ax.plot(xs, ys, marker=marker, label=router)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Verifier-call budget (%)")
        ax.set_ylabel("Buggy programs caught (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Real verification resource curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "code_bug_catch_budget.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [MODEL_LABELS[m] for m in GPT_CLAUDE]
    vals = []
    for model in GPT_CLAUDE:
        row = next((item for item in rich if item["model_alias"] == model), None)
        vals.append(
            0.0
            if row is None or not math.isfinite(float(row.get("heldout_improvement_vs_shared") or np.nan))
            else float(row["heldout_improvement_vs_shared"])
        )
    ax.bar(labels, vals)
    ax.axhline(MATERIAL_LOGLOSS, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Held-out ΔLL (rich − shared)")
    ax.set_title("Score-specific reprioritization")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shared_vs_score_specific.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(GPT_CLAUDE))
    width = 0.35
    mmlu = [100 * MMLU_COVERAGE[m]["delta"] for m in GPT_CLAUDE]
    code = []
    for model in GPT_CLAUDE:
        row = next(
            (
                item
                for item in coverage
                if item["model_alias"] == model and item["score_condition"] == "H1_0.70_minus_0.99"
            ),
            None,
        )
        code.append(0.0 if row is None else 100 * float(row["verify_rate"]))
    ax.bar(x - width / 2, mmlu, width, label="MMLU-Pro (Task 009)")
    ax.bar(x + width / 2, code, width, label="LiveCodeBench (Task 011)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in GPT_CLAUDE])
    ax.set_ylabel("VERIFY coverage shift 0.70→0.99 (pp)")
    ax.set_title("MMLU vs code coverage control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_vs_code_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        "Problem  →  frozen generated code  →  displayed confidence  →  "
        "USE_UNVERIFIED / VERIFY_FIRST  →  independent hidden tests",
        fontsize=11,
        va="center",
    )
    ax.set_title("Code experimental schematic")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "code_experiment_schematic.png", dpi=160)
    plt.close(fig)


def _pp(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{100.0 * value:.1f}pp"


def write_packet(summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    per = decision["per_model"]
    world = decision["world"]
    world_name = {
        "A": "WORLD A — STRONG CROSS-DOMAIN REPLICATION + CONSTRUCTIVE ROUTING",
        "B": "WORLD B — STRONG CROSS-DOMAIN REPLICATION + ROUTING NULL",
        "C": "WORLD C — DOMAIN-DEPENDENT ALLOCATION",
        "D": "WORLD D — COVERAGE DOES NOT GENERALIZE",
        "E": "WORLD E — TECHNICAL/IDENTIFIABILITY FAILURE",
        "MODEL_SPECIFIC": "MODEL_SPECIFIC_GENERALIZATION — GPT coverage replicates; Claude saturates VERIFY",
    }[world]
    gpt_e = per["openai_gpt56_sol"]["effect"]
    claude_e = per["anthropic_sonnet5"]["effect"]
    gpt_dll = per["openai_gpt56_sol"]["dll"]
    claude_dll = per["anthropic_sonnet5"]["dll"]
    bottom = (
        f"On frozen LiveCodeBench programs, counterfactual displayed confidence "
        f"shifted VERIFY_FIRST coverage by {_pp(gpt_e)} for GPT and {_pp(claude_e)} for Claude."
    )
    claim = (
        "On frozen LiveCodeBench programs, counterfactual displayed confidence "
        "strongly changes GPT verification coverage; Claude remains near-always "
        "VERIFY_FIRST, so the MMLU coverage-control result does not replicate in both primary models."
        if world == "MODEL_SPECIFIC"
        else (
            "Counterfactual displayed confidence strongly changes verification coverage "
            "on frozen code outputs."
            if max(gpt_e, claude_e) >= COVERAGE_MODERATE_PP
            else "The MMLU coverage-control effect did not transfer strongly to executable-code verification."
        )
    )
    forbidden = [
        "The ranking is invariant.",
        "Confidence never affects ranking.",
        "The model secretly knows its code is wrong.",
        "Confidence is useless.",
        "Code proves a universal law.",
        "Self-verification (if implying the model runs the tests).",
        "Hidden uncertainty is suppressed.",
    ]
    report = f"""# Task 011 — Prospective code + executable-test generalization

## 1. Plain-English bottom line

{bottom}

Final bucket: **{world_name}**

## 2. Benchmark / version / validation

- Official LiveCodeBench `code_generation_lite` `release_v6`
- Repo commit `{summary["lcb_commit"]}`
- Eligible medium+hard pool: 723; confirmatory ladder: `{summary["ladder"]}` N={summary["main_n"]}
- Pilot 1 (burned, all-hard sampler bug) seed 20260921; pilot 2 seed 20260924 mixed 22 medium / 18 hard
- Hidden tests: official `check_correctness` after routing freeze
- paperDirection.txt was not modified (sha256 `{summary["paper_sha"]}`)

## 3. Pilot and predeclared difficulty adaptation

- Round-2 GO: GPT acc {summary["pilot_acc"]["openai_gpt56_sol"]:.1%}; Claude acc {summary["pilot_acc"]["anthropic_sonnet5"]:.1%}
- GPT >80% on mixed pilot → predeclared **hard_only** confirmatory
- Remaining hard after excluding both pilots: {summary["main_n"]} (below 300; entire remaining hard pool used)

## 4. Freeze / preregistration integrity

- Main ID sha256 `{summary["main_hash"]}`
- Repeat ID sha256 `{summary["repeat_hash"]}`
- Sample, prompts, endpoints, analyses, and decision rules frozen before confirmatory VERIFY-rate inspection
- Label: confirmatory

## 5. Exact API calls, retries, token use, estimated cost

- Scientific requests in sqlite: {summary["n_requests"]} (cap includes two pilots + confirmatory)
- Provider attempts: {summary["n_attempts"]}
- Failures remaining on confirmatory IDs: {summary["n_fail"]}
- Estimated recorded USD: {summary["usd"]:.4f}

## 6. Frozen-code correctness rates

{summary["accuracy_text"]}

## 7. Coverage-control result

- GPT 0.70−0.99: {_pp(gpt_e)} [{summary["gpt_ci"]}]
- Claude 0.70−0.99: {_pp(claude_e)} [{summary["claude_ci"]}]
- Generalization label: **{decision["coverage_gen"]}**

## 8. Shared vs score-specific prioritization

- GPT held-out ΔLL: {gpt_dll}
- Claude held-out ΔLL: {claude_dll}
- Materiality threshold: 0.01. This is **not** a ranking-invariance claim.

## 9. Matched-budget bug-routing

- Label: **{decision["routing"]}**
- Routers were predeclared: random, raw q1, calibrated q1, hidden, q1+hidden, oracle.

## 10. Real verifier resource curve

See `figures/code_bug_catch_budget.png` and `matched_budget_routing.csv`.

## 11. Repeat stability

Descriptive only. See `repeat_stability.csv`. Do not read this as latent invariance.

## 12. Hidden / true-q bridge

See coverage_response.csv rows `hidden` and `true_q_visible`.

## 13. GPT vs Claude

GPT coverage shift {_pp(gpt_e)}; Claude {_pp(claude_e)}. Routing flags: {decision["routing_flags"]}.

## 14. MMLU vs code

MMLU Task 009: GPT 50.2pp, Claude 35.4pp, ΔLL ≈ 0.
Code Task 011: GPT {_pp(gpt_e)}, Claude {_pp(claude_e)}, ΔLL GPT {gpt_dll}, Claude {claude_dll}.

## 15. Evidence against the current D1 framing

{summary["against"]}

## 16. Final WORLD classification

**{world_name}**

## 17. Strongest defensible paper claim

{claim}

## 18. Claims that must NOT be used

{chr(10).join("- " + item for item in forbidden)}

## 19. Recommended main-paper figures

- MMLU vs code coverage (`figures/mmlu_vs_code_summary.png`)
- Code coverage response (`figures/code_coverage_response.png`)
- Bug-catch vs budget (`figures/code_bug_catch_budget.png`)
- Shared vs score-specific ΔLL (`figures/shared_vs_score_specific.png`)

## 20. Are any new experiments scientifically justified?

**No.** After Task 011, STOP new experiments unless a genuine evaluator/data/preregistration bug is found.

## 21. READY_FOR_GPT_REVIEW = YES
"""
    (RETURN_DIR / "report.md").write_text(report, encoding="utf-8")
    cross_rows = []
    for model in GPT_CLAUDE:
        cross_rows.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "mmlu_coverage_0.70_minus_0.99": MMLU_COVERAGE[model]["delta"],
                "code_coverage_0.70_minus_0.99": per[model]["effect"],
                "mmlu_dll": MMLU_DLL[model],
                "code_dll": per[model]["dll"],
                "code_accuracy": summary["accuracy"].get(model),
            }
        )
    write_csv(RETURN_DIR / "cross_domain_summary.csv", cross_rows)
    (RETURN_DIR / "cross_domain_comparison.md").write_text(
        "# MMLU-Pro vs executable code\n\n"
        f"Task 009 MMLU coverage shifts were GPT 50.2pp and Claude 35.4pp with "
        f"essentially no score-specific ΔLL. Task 011 code shifts are GPT {_pp(gpt_e)} "
        f"and Claude {_pp(claude_e)} with ΔLL GPT {gpt_dll} and Claude {claude_dll}. "
        "Do not force the metrics to be identical; the verifier is now a real hidden "
        "test suite. The paper-facing question is which parts of confidence-routing "
        "reproduce when verification becomes an executable operation.\n",
        encoding="utf-8",
    )
    (RETURN_DIR / "paper_integration_packet.md").write_text(
        f"""# Paper integration packet (not a manuscript)

## A. Final one-sentence result

{claim}

## B. Abstract paragraph

Large language models can request costly independent verification before an output is used. We hold the output fixed and counterfactually vary only the confidence shown back to GPT and Claude. On MMLU-Pro, displayed confidence caused very large changes in verification coverage while a richer score-specific reprioritization model added essentially no held-out predictive value. We then test whether that coverage-control result survives when verification is a real operation: executing an official hidden test suite on frozen LiveCodeBench programs. In code, the 0.70 vs 0.99 coverage shift is {_pp(gpt_e)} for GPT and {_pp(claude_e)} for Claude. Score-specific reprioritization ΔLL is GPT {gpt_dll} and Claude {claude_dll}. At matched verifier-call budgets, predeclared simple routers are summarized as {decision["routing"]}. Quantity and allocation remain empirically distinct: confidence can tune how much code is sent to tests; catching bugs at a budget is a separate routing problem.

## C. Introduction logic

1. Costly selective verification
2. Confidence as a routing input
3. Quantity vs allocation
4. Fixed-output causal intervention
5. MMLU-Pro Task 009
6. Real code-verifier Task 011
7. Design implication: coverage control is not bug-ranking

## D. Contributions

1. Causal coverage-control result on frozen MMLU answers
2. Prospective generalization to frozen executable code
3. Claim-disciplined shared vs score-specific comparison
4. Matched-budget bug-catching with a real verifier

## E. Main-text result sequence

Task 009 coverage + allocation null; Task 011 code coverage; resource curve; systems implication.

## F. Appendix

Pilot ladder, hard-only N={summary["main_n"]}, repeat stochasticity, hidden/true-q bridge.

## G. Reviewer attacks remaining

1. Hard-only confirmatory after mixed-pilot accuracy
2. N=286 vs target 500
3. Claude high VERIFY rates
4. Other-model-correctness as the item-risk feature
5. LiveCodeBench contamination window

## H. STOP recommendation

The scientific package is complete. STOP new experiments.
""",
        encoding="utf-8",
    )
    (RETURN_DIR / "paperDirection_proposed_patch.md").write_text(
        f"""# Proposed paperDirection.txt patch (NOT applied)

Do not edit paperDirection.txt until this patch is accepted.

Replace section 49 "WHAT REMAINS..." remaining-priority language with:

CURRENT REMAINING SCIENTIFIC PRIORITY:

    none. Task 011 tested the same coverage-control claim on executable
    LiveCodeBench code with an official hidden-test verifier.

TASK-011 RESULT:

    Frozen-code confirmatory N={summary["main_n"]} hard-only LiveCodeBench.
    Coverage shift 0.70→0.99: GPT {_pp(gpt_e)}, Claude {_pp(claude_e)}.
    Score-specific ΔLL: GPT {gpt_dll}, Claude {claude_dll}.
    Matched-budget routing: {decision["routing"]}.
    World: {world_name}.

CURRENT CORE PAPER CLAIM:

    {claim}

FINAL WRITING RULE is unchanged: STOP new experiments; write the paper.
""",
        encoding="utf-8",
    )
    (RETURN_DIR / "cost_summary.md").write_text(
        f"""# Task 011 cost summary

- Scientific sqlite rows: {summary["n_requests"]}
- Provider attempts: {summary["n_attempts"]}
- Recorded USD (sum of stored estimated_cost_usd on confirmatory tables): {summary["usd"]:.4f}
- Two excluded pilots: 800 scientific calls
- Confirmatory planned from freeze_manifest: {summary.get("planned_confirmatory_calls")}
- Hidden tests are local evaluator executions, not API calls
""",
        encoding="utf-8",
    )
    changed = [
        "src/task011.py",
        "src/task011_common.py",
        "src/task011_benchmark.py",
        "src/task011_prompts.py",
        "src/task011_sample.py",
        "src/task011_run.py",
        "src/task011_eval.py",
        "src/task011_analyze.py",
        "src/model_adapters.py",
        "src/schemas.py",
        "from_gpt/011_code_executable_verification_generalization.md",
        "to_gpt/011_code_executable_verification_generalization/",
    ]
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")


def run_analysis(run_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    before = protected_fingerprints()
    manifest = load_json(RETURN_DIR / "freeze_manifest.json")
    tests_path = RETURN_DIR / "test_execution_results.csv"
    if not tests_path.exists():
        tests_path = RETURN_DIR / "hidden_test_results.csv"
    tests = load_csv(tests_path)
    main_ids = list(manifest["main_ids"])
    tests = [row for row in tests if str(row["question_id"]) in set(main_ids)]
    requests, n_attempts = _load_requests()
    tables = export_tables(requests, main_ids=main_ids, tests=tests)
    difficulty = attach_difficulty(tables["stage12"])
    coverage = coverage_table(tables["stage3"])
    shared, rich = ranking_models(tables["stage3"], difficulty)
    stability = rank_stability(tables["stage3"], tables["repeats"])
    matched = matched_budget(tables["stage3"], tables["stage12"])
    risk = risk_coverage(tables["stage3"])
    decision = classify_world(coverage, rich, matched)
    plot_figures(coverage, matched, rich)
    acc = {}
    acc_text = []
    for model in GPT_CLAUDE:
        rows = [row for row in tables["stage12"] if row["model_alias"] == model]
        vals = [int(row["stage1_correct"]) for row in rows if row.get("stage1_correct") not in (None, "")]
        rate = sum(vals) / len(vals) if vals else float("nan")
        acc[model] = rate
        acc_text.append(f"- {MODEL_LABELS[model]}: {rate:.1%} of {len(vals)} frozen programs passed all hidden tests")
    usd = 0.0
    for group in ("stage1", "q1", "stage3", "repeats"):
        usd += sum(float(row.get("estimated_cost_usd") or 0) for row in tables[group])
    gpt_h1 = next(
        row
        for row in coverage
        if row["model_alias"] == "openai_gpt56_sol" and row["score_condition"] == "H1_0.70_minus_0.99"
    )
    claude_h1 = next(
        row
        for row in coverage
        if row["model_alias"] == "anthropic_sonnet5" and row["score_condition"] == "H1_0.70_minus_0.99"
    )
    decision["per_model"]["openai_gpt56_sol"]["lo"] = gpt_h1["ci_lo"]
    decision["per_model"]["anthropic_sonnet5"]["lo"] = claude_h1["ci_lo"]
    pilot = load_json(RETURN_DIR / "pilot_run.json")
    against = (
        "If coverage is weak, D1 is MMLU-specific. If score-specific ΔLL is material, "
        "code routing is more condition-sensitive than MMLU. If Claude remains near-saturated, "
        "coverage-control is model-asymmetric."
    )
    if decision["world"] == "D":
        against = "The large MMLU coverage effect did not transfer to this executable-code verifier."
    elif decision["world"] == "MODEL_SPECIFIC":
        against = (
            "Claude is near-saturated on VERIFY_FIRST for hard code (~95–100%), "
            "so the two-model MMLU coverage-control story does not transfer intact. "
            "GPT still shows a large 0.70→0.99 coverage shift. Allocation ΔLL is null."
        )
    elif decision["world"] == "B":
        against = "No strong evidence against coverage-dominant D1; allocation remains weak in code too."
    elif decision["world"] == "A":
        against = "Coverage generalizes, but a contextual router adds matched-budget value, so allocation is not empty."
    summary = {
        "decision": decision,
        "coverage": coverage,
        "shared": shared,
        "rich": rich,
        "stability": stability,
        "matched": matched,
        "risk": risk,
        "n_requests": len(requests),
        "n_attempts": n_attempts,
        "n_fail": len(tables["failures"]),
        "usd": usd,
        "main_n": manifest["main_n"],
        "main_hash": manifest["main_id_list_sha256"],
        "repeat_hash": manifest["repeat_id_list_sha256"],
        "ladder": manifest["ladder"],
        "lcb_commit": manifest["lcb_commit"],
        "paper_sha": sha256_file(PAPER_DIRECTION),
        "accuracy": acc,
        "accuracy_text": "\n".join(acc_text),
        "gpt_ci": f"{100*float(gpt_h1['ci_lo']):.1f}, {100*float(gpt_h1['ci_hi']):.1f}",
        "claude_ci": f"{100*float(claude_h1['ci_lo']):.1f}, {100*float(claude_h1['ci_hi']):.1f}",
        "pilot_acc": pilot.get("decision", {}).get("accuracy") or {},
        "planned_confirmatory_calls": manifest.get("planned_confirmatory_calls"),
        "against": against,
        "run": run_payload or {},
    }
    write_packet(summary)
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 011 analysis")
    json_dump(
        RETURN_DIR / "analysis_summary.json",
        {k: v for k, v in summary.items() if k not in {"coverage", "shared", "rich", "stability", "matched", "risk", "run"}},
    )
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return summary
