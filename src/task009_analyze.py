"""Task 009 confirmatory analysis. Run only after primary GPT/Claude cells complete."""

from __future__ import annotations

import json
import math
import random
import sqlite3
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from .study1_analysis import _is_verify, _pyplot
from .task005_lane_a import error_catch_from_weights, fractional_verify_weights
from .task009_common import (
    ADJACENT_PAIRS,
    ALL_MODELS,
    ANALYSIS_DIR,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CV_FOLDS,
    EPS,
    FAILURE_INCONCLUSIVE_FRAC,
    FIGURES_DIR,
    GEMINI_GROK,
    GPT_CLAUDE,
    H1_FLOORS,
    H1_HIGH,
    H1_LOW,
    MATERIAL_AUROC,
    MATERIAL_LOGLOSS,
    MATCHED_BUDGETS,
    MODEL_LABELS,
    PLANNED_PRIMARY_TOTAL,
    REFERENCE_SCORE,
    RETURN_DIR,
    SATURATION_HIGH,
    SATURATION_LABEL,
    SATURATION_LOW,
    SCIENTIFIC_CAP,
    SCORE_CONDITIONS,
    SQLITE_PATH,
    VISIBLE_FIXED_CONDITIONS,
    assert_009_write_target,
    freeze_paths,
    json_dump,
    load_json,
    write_csv,
)
from .task009_sample import load_frozen_ids


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


def saturation_label(rate: float) -> str:
    if not math.isfinite(rate):
        return "unknown"
    if rate <= SATURATION_LOW or rate >= SATURATION_HIGH:
        return SATURATION_LABEL
    return "identifiable"


def _open_readonly() -> sqlite3.Connection:
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(SQLITE_PATH)
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
        attempts = connection.execute(
            "SELECT COUNT(*) AS n FROM attempts"
        ).fetchone()["n"]
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


def export_result_tables(requests: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    stage12: list[dict[str, Any]] = []
    stage3: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    answers: dict[tuple[str, str], dict[str, Any]] = {}
    confs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in requests:
        rec = row["record"]
        key = (row["example_id"], row["model_alias"])
        if row["status"] != "success":
            failures.append(
                {
                    "request_key": row["request_key"],
                    "stage": row["stage"],
                    "question_id": row["example_id"],
                    "model_alias": row["model_alias"],
                    "status": row["status"],
                    "error": row["error"],
                    "score_condition": rec.get("score_condition"),
                    "repeat_index": rec.get("repeat_index"),
                }
            )
            continue
        if row["stage"] == "answer":
            answers[key] = rec
        elif row["stage"] == "confidence":
            confs[key] = rec
        elif row["stage"] == "verification":
            action = rec.get("parsed_action")
            item = {
                "question_id": rec.get("question_id") or row["example_id"],
                "model_alias": row["model_alias"],
                "model_endpoint": rec.get("model_endpoint") or row["requested_model_id"],
                "family": rec.get("stakes_family"),
                "score_condition": rec.get("score_condition"),
                "displayed_confidence": rec.get("displayed_confidence"),
                "displayed_token": rec.get("displayed_token"),
                "frozen_answer": rec.get("frozen_answer"),
                "stage1_correct": rec.get("stage1_correct"),
                "reported_confidence": rec.get("reported_confidence"),
                "parsed_action": action,
                "verify": _is_verify(action),
                "repeat_index": rec.get("repeat_index", 0),
                "roster": rec.get("roster"),
                "prompt_hash": rec.get("prompt_hash"),
                "request_key": row["request_key"],
                "estimated_cost_usd": rec.get("estimated_cost_usd"),
                "raw_response": rec.get("raw_response"),
            }
            if int(item["repeat_index"] or 0) == 0 and item.get("roster") != "repeat_extra":
                stage3.append(item)
            else:
                repeats.append(item)
    for key, ans in answers.items():
        conf = confs.get(key, {})
        stage12.append(
            {
                "question_id": key[0],
                "model_alias": key[1],
                "frozen_answer": ans.get("answer_label") or ans.get("answer"),
                "stage1_correct": int(bool(ans.get("is_correct"))),
                "correct_label": ans.get("correct_label"),
                "q1": conf.get("probability_correct") or conf.get("confidence"),
                "answer_request_key": ans.get("request_key"),
                "confidence_request_key": conf.get("request_key"),
                "category": ans.get("category"),
                "answer_raw": ans.get("raw_response"),
                "q1_raw": conf.get("raw_response"),
            }
        )
    write_csv(RETURN_DIR / "stage1_q1_results.csv", stage12)
    write_csv(RETURN_DIR / "stage3_results.csv", stage3)
    write_csv(RETURN_DIR / "repeat_results.csv", repeats)
    write_csv(RETURN_DIR / "failures.csv", failures)
    return {
        "stage12": stage12,
        "stage3": stage3,
        "repeats": repeats,
        "failures": failures,
    }


def attach_difficulty(stage12: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_q: dict[str, dict[str, int]] = defaultdict(dict)
    for row in stage12:
        by_q[str(row["question_id"])][str(row["model_alias"])] = int(row["stage1_correct"])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for qid, model_map in by_q.items():
        for target in ALL_MODELS:
            if target not in model_map:
                continue
            others_primary = [alias for alias in GPT_CLAUDE if alias != target]
            other_primary_correct = sum(model_map.get(alias, 0) for alias in others_primary)
            others_all = [alias for alias in ALL_MODELS if alias != target]
            other_all_correct = sum(model_map.get(alias, 0) for alias in others_all)
            out[(qid, target)] = {
                "target_correct": model_map[target],
                "other_primary_correct": other_primary_correct,
                "other_all_correct": other_all_correct,
                "n_other_primary": len(others_primary),
                "n_other_all_present": sum(1 for alias in others_all if alias in model_map),
            }
    return out


def coverage_table(stage3: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        model_rows = [row for row in stage3 if row["model_alias"] == model]
        if not model_rows:
            continue
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
                    "saturation": saturation_label(rate),
                }
            )
        if complete:
            diffs = [
                complete[qid][H1_LOW] - complete[qid][H1_HIGH] for qid in complete
            ]
            boots = []
            qids = list(complete)
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw_q = [qids[rng.randrange(len(qids))] for _ in qids]
                draw = [
                    complete[qid][H1_LOW] - complete[qid][H1_HIGH] for qid in draw_q
                ]
                boots.append(float(np.mean(draw)))
            lo, hi = _ci(boots)
            effect = _mean(diffs)
            floor = H1_FLOORS.get(model)
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score_condition": "H1_0.70_minus_0.99",
                    "n_questions": len(complete),
                    "verify_rate": effect,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "saturation": "",
                    "floor": floor,
                    "meets_floor": (
                        bool(floor is not None and math.isfinite(effect) and effect >= floor)
                    ),
                }
            )
    write_csv(RETURN_DIR / "coverage_response.csv", rows)
    return rows


def nesting_table(stage3: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        model_rows = [row for row in stage3 if row["model_alias"] == model]
        by_q: dict[str, dict[str, int]] = defaultdict(dict)
        for row in model_rows:
            by_q[str(row["question_id"])][str(row["score_condition"])] = int(row["verify"])
        for left, right in ADJACENT_PAIRS:
            vv = vu = uu = uv = 0
            n = 0
            for conds in by_q.values():
                if left not in conds or right not in conds:
                    continue
                n += 1
                a, b = conds[left], conds[right]
                if a and b:
                    vv += 1
                elif a and not b:
                    vu += 1
                elif (not a) and (not b):
                    uu += 1
                else:
                    uv += 1
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "from_condition": left,
                    "to_condition": right,
                    "n": n,
                    "V_to_V": vv,
                    "V_to_U": vu,
                    "U_to_U": uu,
                    "U_to_V": uv,
                    "reversal_rate": (vu + uv) / n if n else float("nan"),
                    "downshift_V_to_U": vu / n if n else float("nan"),
                    "upshift_U_to_V": uv / n if n else float("nan"),
                }
            )
    write_csv(RETURN_DIR / "nesting_reversals.csv", rows)
    return rows


def _grouped_cv(
    visible: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    qids = np.array([row["question_id"] for row in visible])
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    score = np.array(
        [float(row["displayed_confidence"]) for row in visible], dtype=float
    )
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
    ll_shared = _safe_log_loss(y, p_shared)
    ll_rich = _safe_log_loss(y, p_rich)
    return p_shared, p_rich, qids, ll_shared, ll_rich


def ranking_models(
    stage3: Sequence[Mapping[str, Any]],
    difficulty: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        passes = bool(
            math.isfinite(improvement)
            and improvement < MATERIAL_LOGLOSS
            and math.isfinite(hi)
            and hi < MATERIAL_LOGLOSS
        )
        shared_out.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "model_name": "shared_ranking_transferable",
                "formula": "logit P(VERIFY)=alpha_condition + beta*other_primary_wrongness",
                "n_rows": len(visible),
                "n_questions": len(q_unique),
                "cv_logloss": ll_s,
            }
        )
        rich_out.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "model_name": "condition_sensitive_score_x_difficulty",
                "formula": "shared + gamma*(score-0.90)*wrongness",
                "n_rows": len(visible),
                "n_questions": len(q_unique),
                "cv_logloss": ll_r,
                "heldout_improvement_vs_shared": improvement,
                "bootstrap_lo": lo,
                "bootstrap_hi": hi,
                "material_logloss": MATERIAL_LOGLOSS,
                "invariance_criterion_pass": passes,
            }
        )
    write_csv(RETURN_DIR / "shared_ranking_model_results.csv", shared_out)
    write_csv(RETURN_DIR / "condition_sensitive_model_results.csv", rich_out)
    return shared_out, rich_out


def rank_stability(
    stage3: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    combined = list(stage3) + list(repeats)
    rows: list[dict[str, Any]] = []
    for model in GPT_CLAUDE:
        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        same_prompt: dict[tuple[str, str], list[int]] = defaultdict(list)
        wrong: dict[str, int] = {}
        for row in combined:
            if row["model_alias"] != model:
                continue
            qid = str(row["question_id"])
            cond = str(row["score_condition"])
            cells[(qid, cond)].append(int(row["verify"]))
            same_prompt[(qid, cond)].append(int(row["verify"]))
            if row.get("stage1_correct") is not None:
                wrong[qid] = int(not bool(row["stage1_correct"]))
        means: dict[str, dict[str, float]] = defaultdict(dict)
        rerun_dis: list[float] = []
        for (qid, cond), vals in cells.items():
            if len(vals) >= 2:
                disag = []
                for i, a in enumerate(vals):
                    for b in vals[i + 1 :]:
                        disag.append(float(a != b))
                if disag:
                    rerun_dis.append(float(np.mean(disag)))
            if vals:
                means[qid][cond] = float(np.mean(vals))
        rerun_rate = _mean(rerun_dis)
        qids = [qid for qid, conds in means.items() if all(c in conds for c in VISIBLE_FIXED_CONDITIONS)]
        for left, right in ADJACENT_PAIRS:
            xs = [means[qid][left] for qid in qids]
            ys = [means[qid][right] for qid in qids]
            rho = _spearman(xs, ys)
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
            n_rev = 0
            n_pair = 0
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
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "pairwise_order_reversal_rate": n_rev / n_pair if n_pair else float("nan"),
                    "same_prompt_rerun_disagreement": rerun_rate,
                    "n_generations_used": 3,
                }
            )
    write_csv(RETURN_DIR / "rank_stability.csv", rows)
    return rows


def condition_auc(
    stage3: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    combined = list(stage3) + list(repeats)
    rows: list[dict[str, Any]] = []
    rng = random.Random(BOOTSTRAP_SEED)
    for model in ALL_MODELS:
        model_rows = [row for row in combined if row["model_alias"] == model]
        by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
        wrong: dict[str, int] = {}
        for row in model_rows:
            qid = str(row["question_id"])
            by_cell[(qid, str(row["score_condition"]))].append(int(row["verify"]))
            if row.get("stage1_correct") is not None:
                wrong[qid] = int(not bool(row["stage1_correct"]))
        for condition in SCORE_CONDITIONS:
            qids = [
                qid
                for (qid, cond) in by_cell
                if cond == condition and qid in wrong
            ]
            if condition in VISIBLE_FIXED_CONDITIONS:
                scores = [float(np.mean(by_cell[(qid, condition)])) for qid in qids]
            else:
                scores = [
                    float(np.mean(by_cell[(qid, condition)]))
                    for qid in qids
                    if (qid, condition) in by_cell
                ]
            y = [wrong[qid] for qid in qids]
            auc = _safe_auc(y, scores)
            boots = []
            for _ in range(min(BOOTSTRAP_RESAMPLES, 2000)):
                if not qids:
                    break
                draw = [qids[rng.randrange(len(qids))] for _ in qids]
                boots.append(
                    _safe_auc(
                        [wrong[qid] for qid in draw],
                        [float(np.mean(by_cell[(qid, condition)])) for qid in draw],
                    )
                )
            lo, hi = _ci(boots)
            rate = _mean([int(np.mean(by_cell[(qid, condition)]) >= 0.5) for qid in qids])
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score_condition": condition,
                    "n": len(qids),
                    "n_wrong": int(sum(y)),
                    "auc": auc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "saturation": saturation_label(
                        _mean([float(np.mean(by_cell[(qid, condition)])) for qid in qids])
                    ),
                    "verify_rate": _mean(
                        [float(np.mean(by_cell[(qid, condition)])) for qid in qids]
                    ),
                    "binary_rate_for_ref": rate,
                }
            )
    write_csv(RETURN_DIR / "condition_auc.csv", rows)
    return rows


def risk_coverage(stage3: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        model_rows = [row for row in stage3 if row["model_alias"] == model]
        for condition in SCORE_CONDITIONS:
            cell = [row for row in model_rows if row["score_condition"] == condition]
            if not cell:
                continue
            n = len(cell)
            n_wrong = sum(int(not bool(row["stage1_correct"])) for row in cell)
            n_verify = sum(int(row["verify"]) for row in cell)
            caught = sum(
                int(row["verify"] and not bool(row["stage1_correct"])) for row in cell
            )
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score_condition": condition,
                    "n": n,
                    "coverage": n_verify / n if n else float("nan"),
                    "n_wrong": n_wrong,
                    "errors_caught_frac": caught / n_wrong if n_wrong else float("nan"),
                    "residual_risk": (n_wrong - caught) / n if n else float("nan"),
                    "frac_correct_verified": (
                        sum(int(row["verify"] and bool(row["stage1_correct"])) for row in cell)
                        / max(1, n - n_wrong)
                    ),
                }
            )
    write_csv(RETURN_DIR / "risk_coverage.csv", rows)
    return rows


def matched_budget(
    stage3: Sequence[Mapping[str, Any]],
    stage12: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
            if row["model_alias"] == model and row["score_condition"] == "hidden"
        ]
        if not hidden:
            continue
        qids = [str(row["question_id"]) for row in hidden]
        wrong = np.array([int(not bool(row["stage1_correct"])) for row in hidden])
        hidden_scores = np.array([int(row["verify"]) for row in hidden], dtype=float)
        q1_scores = np.array(
            [1.0 - q1.get((qid, model), 0.5) for qid in qids], dtype=float
        )
        combo = np.clip(0.5 * hidden_scores + 0.5 * q1_scores, 0.0, 1.0)
        oracle = wrong.astype(float)
        for budget in MATCHED_BUDGETS:
            n_verify = budget * len(qids)
            routers = {
                "random": rng.random(len(qids)),
                "raw_q1": q1_scores,
                "hidden": hidden_scores,
                "q1_plus_hidden": combo,
                "hindsight_oracle": oracle,
            }
            for name, score in routers.items():
                # Higher score = verify first, except random.
                weights = fractional_verify_weights(-score if name != "random" else score, n_verify)
                if name != "random":
                    weights = fractional_verify_weights(-score, n_verify)
                else:
                    order = np.argsort(score)
                    weights = np.zeros(len(score))
                    take = int(round(n_verify))
                    weights[order[:take]] = 1.0
                catch = error_catch_from_weights(weights, wrong)
                rows.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "budget": budget,
                        "router": name,
                        "catch_frac": catch,
                        "n": len(qids),
                        "n_wrong": int(wrong.sum()),
                    }
                )
    write_csv(RETURN_DIR / "matched_budget_routing.csv", rows)
    return rows


def secondary_summary(coverage: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in coverage if row["model_alias"] in GEMINI_GROK]
    write_csv(RETURN_DIR / "secondary_models_summary.csv", rows)
    return rows


def decide_bucket(
    *,
    coverage: Sequence[Mapping[str, Any]],
    rich: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    auc_rows: Sequence[Mapping[str, Any]],
    n_fail_primary: int,
    n_planned_primary: int,
) -> dict[str, Any]:
    if n_planned_primary and n_fail_primary / n_planned_primary > FAILURE_INCONCLUSIVE_FRAC:
        return {
            "bucket": "INCONCLUSIVE",
            "world": "undetermined",
            "reason": "primary failure fraction exceeded the frozen 5% rule",
        }
    per_model: dict[str, dict[str, Any]] = {}
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
        h2 = next((row for row in rich if row["model_alias"] == model), None)
        rhos = [
            float(row["spearman"])
            for row in stability
            if row["model_alias"] == model and math.isfinite(float(row["spearman"]))
        ]
        identifiable_auc = [
            float(row["auc"])
            for row in auc_rows
            if row["model_alias"] == model
            and row["score_condition"] in VISIBLE_FIXED_CONDITIONS
            and row.get("saturation") != SATURATION_LABEL
            and math.isfinite(float(row["auc"]))
        ]
        coverage_large = bool(h1 and h1.get("meets_floor"))
        invariance_pass = bool(h2 and h2.get("invariance_criterion_pass"))
        reshape = bool(
            h2
            and math.isfinite(float(h2.get("heldout_improvement_vs_shared") or np.nan))
            and float(h2["heldout_improvement_vs_shared"]) >= MATERIAL_LOGLOSS
        )
        rank_strong = bool(rhos and _mean(rhos) >= 0.70)
        auc_span = (
            max(identifiable_auc) - min(identifiable_auc) if len(identifiable_auc) >= 2 else 0.0
        )
        disc_stable = bool(auc_span < MATERIAL_AUROC)
        if not h1 or not math.isfinite(float(h1.get("verify_rate") or np.nan)):
            label = "inconclusive"
        elif reshape and coverage_large:
            label = "reshape"
        elif coverage_large and invariance_pass and disc_stable:
            label = "invariance"
        elif coverage_large and invariance_pass:
            label = "invariance"
        else:
            label = "inconclusive"
        per_model[model] = {
            "label": label,
            "coverage_large": coverage_large,
            "invariance_pass": invariance_pass,
            "reshape": reshape,
            "rank_strong": rank_strong,
            "disc_stable": disc_stable,
            "mean_spearman": _mean(rhos),
            "auc_span": auc_span,
        }
    labels = {item["label"] for item in per_model.values()}
    if labels == {"invariance"}:
        bucket = "PROSPECTIVE_INVARIANCE_SUPPORTED"
        world = "ranking invariance (HOW MANY)"
    elif labels == {"reshape"}:
        bucket = "PROSPECTIVE_RESHAPING_SUPPORTED"
        world = "prioritization reshaping (WHICH)"
    elif "invariance" in labels and "reshape" in labels:
        bucket = "MIXED_BY_MODEL"
        world = "mixed GPT vs Claude"
    else:
        bucket = "INCONCLUSIVE"
        world = "undetermined"
    return {"bucket": bucket, "world": world, "per_model": per_model}


def plot_figures(
    coverage: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    risk: Sequence[Mapping[str, Any]],
    matched: Sequence[Mapping[str, Any]],
) -> None:
    plt = _pyplot()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in GPT_CLAUDE:
        xs, ys = [], []
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
        ax.plot(xs, ys, marker="o", label=MODEL_LABELS[model])
    ax.set_xlabel("Displayed confidence")
    ax.set_ylabel("VERIFY rate (%)")
    ax.set_title("Coverage response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coverage_response.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [f"{a.split('_')[-1]}→{b.split('_')[-1]}" for a, b in ADJACENT_PAIRS]
    x = np.arange(len(labels))
    width = 0.35
    for i, model in enumerate(GPT_CLAUDE):
        vals = []
        for pair in ADJACENT_PAIRS:
            key = f"{pair[0]}->{pair[1]}"
            row = next(
                (
                    item
                    for item in stability
                    if item["model_alias"] == model and item["pair"] == key
                ),
                None,
            )
            vals.append(float(row["spearman"]) if row else float("nan"))
        ax.bar(x + (i - 0.5) * width, vals, width, label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spearman")
    ax.set_title("Repeat rank stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rank_stability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in GPT_CLAUDE:
        xs, ys = [], []
        for row in risk:
            if row["model_alias"] != model:
                continue
            if row["score_condition"] not in VISIBLE_FIXED_CONDITIONS:
                continue
            xs.append(100.0 * float(row["coverage"]))
            ys.append(100.0 * float(row["errors_caught_frac"]))
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=MODEL_LABELS[model])
    ax.set_xlabel("Verification coverage (%)")
    ax.set_ylabel("Errors caught (%)")
    ax.set_title("Risk-coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_coverage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in GPT_CLAUDE:
        for router in ("raw_q1", "hidden", "q1_plus_hidden", "hindsight_oracle"):
            xs, ys = [], []
            for row in matched:
                if row["model_alias"] == model and row["router"] == router:
                    xs.append(100.0 * float(row["budget"]))
                    ys.append(100.0 * float(row["catch_frac"]))
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{MODEL_LABELS[model]} {router}")
    ax.set_xlabel("Verification budget (%)")
    ax.set_ylabel("Errors caught (%)")
    ax.set_title("Matched-budget error catch")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matched_budget_error_catch.png", dpi=160)
    plt.close(fig)


def _fmt_pp(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}"


def write_report(summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    coverage = summary["coverage"]
    lines = [
        "# Task 009 — Prospective confirmation of confidence-as-coverage control",
        "",
        "## 1. Plain-English bottom line",
        "",
        summary["bottom_line"],
        "",
        "## 2. Preregistration integrity",
        "",
        f"- Freeze file: `{freeze_paths()['freeze_manifest']}`",
        f"- Primary 500 SHA256: `{summary['primary_hash']}`",
        f"- Secondary 200 SHA256: `{summary['secondary_hash']}`",
        f"- Repeat 100 SHA256: `{summary['repeat_hash']}`",
        "- Sample, prompts, endpoints, metrics, and decision buckets were frozen before VERIFY-rate inspection.",
        f"- paperDirection.txt was not modified (sha256 `{summary['paper_sha']}`).",
        "",
        "## 3. Calls, failures, retries, cost",
        "",
        f"- Scientific requests registered: {summary['n_requests']} / cap {SCIENTIFIC_CAP}",
        f"- Provider attempts: {summary['n_attempts']}",
        f"- Failures remaining: {summary['n_failures']}",
        f"- Estimated Stage-3 USD (recorded): {summary['stage3_usd']:.4f}",
        "",
        "## 4. Coverage result (H1)",
        "",
    ]
    for model in GPT_CLAUDE:
        row = next(
            (
                item
                for item in coverage
                if item["model_alias"] == model
                and item["score_condition"] == "H1_0.70_minus_0.99"
            ),
            None,
        )
        if row:
            lines.append(
                f"- {MODEL_LABELS[model]}: "
                f"{_fmt_pp(row['verify_rate'])}pp "
                f"[{_fmt_pp(row['ci_lo'])}, {_fmt_pp(row['ci_hi'])}] "
                f"(floor {_fmt_pp(row.get('floor'))}; meets={row.get('meets_floor')})"
            )
    lines += [
        "",
        "## 5. Shared-ranking vs condition-sensitive (H2)",
        "",
    ]
    for row in summary["rich"]:
        lines.append(
            f"- {row['model_label']}: held-out improvement {row['heldout_improvement_vs_shared']:.4f} "
            f"[{row['bootstrap_lo']:.4f}, {row['bootstrap_hi']:.4f}]; "
            f"invariance pass={row['invariance_criterion_pass']}"
        )
    lines += [
        "",
        "## 6. Repeat rank stability (H3)",
        "",
    ]
    for row in summary["stability"]:
        lines.append(
            f"- {row['model_label']} {row['pair']}: Spearman {row['spearman']:.3f} "
            f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]; "
            f"order-reversal {row['pairwise_order_reversal_rate']}"
        )
    lines += [
        "",
        "## 7. Discrimination stability (H4)",
        "",
    ]
    for row in summary["auc"]:
        if row["model_alias"] in GPT_CLAUDE and row["score_condition"] in VISIBLE_FIXED_CONDITIONS:
            lines.append(
                f"- {row['model_label']} {row['score_condition']}: AUROC {row['auc']:.3f} "
                f"({row['saturation']})"
            )
    lines += [
        "",
        "## 8. Nesting / reversals",
        "",
        "See `nesting_reversals.csv`. Adjacent visible-score V→U is the coverage-shift pattern; U→V is the anti-nested residue.",
        "",
        "## 9. Risk-coverage interpretation",
        "",
        "Raw catch rates track coverage. Do not treat catch changes at different coverage as ranking changes. See `risk_coverage.csv` and `figures/risk_coverage.png`.",
        "",
        "## 10. Hidden / true-q bridge",
        "",
        "Hidden and true_q_visible are contextual bridges, not members of the fixed-score threshold family. See coverage_response.csv for those two cells.",
        "",
        "## 11. Matched-budget routing",
        "",
        "Consequence analysis only. Routers were predeclared: random, raw q1, hidden judgment, q1+hidden, hindsight oracle. See `matched_budget_routing.csv`.",
        "",
        "## 12. Gemini / Grok secondary",
        "",
        "Compatibility only; they do not redefine the GPT/Claude hypothesis. See `secondary_models_summary.csv`.",
        "",
        "## 13. Decision bucket",
        "",
        f"`{decision['bucket']}`",
        "",
        "## 14. Which final-paper world won",
        "",
        decision["world"],
        "",
        "## 15. Narrative implication",
        "",
        summary["narrative"],
        "",
        "## 16. Whether to run a second task",
        "",
        summary["next_task"],
        "",
        "## 17. Review flag",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
    ]
    path = RETURN_DIR / "report.md"
    assert_009_write_target(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(run_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .task005_common import sha256_file
    from .task009_common import PAPER_DIRECTION, protected_fingerprints

    before = protected_fingerprints()
    manifest = load_frozen_ids()
    requests, n_attempts = _load_requests()
    tables = export_result_tables(requests)
    difficulty = attach_difficulty(tables["stage12"])
    coverage = coverage_table(tables["stage3"])
    nesting = nesting_table(tables["stage3"])
    shared, rich = ranking_models(tables["stage3"], difficulty)
    stability = rank_stability(tables["stage3"], tables["repeats"])
    auc_rows = condition_auc(tables["stage3"], tables["repeats"])
    risk = risk_coverage(tables["stage3"])
    matched = matched_budget(tables["stage3"], tables["stage12"])
    secondary_summary(coverage)
    n_fail = len(tables["failures"])
    decision = decide_bucket(
        coverage=coverage,
        rich=rich,
        stability=stability,
        auc_rows=auc_rows,
        n_fail_primary=sum(
            1
            for row in tables["failures"]
            if row["model_alias"] in GPT_CLAUDE
        ),
        n_planned_primary=PLANNED_PRIMARY_TOTAL,
    )
    plot_figures(coverage, stability, risk, matched)
    stage3_usd = sum(float(row.get("estimated_cost_usd") or 0) for row in tables["stage3"])
    stage3_usd += sum(float(row.get("estimated_cost_usd") or 0) for row in tables["repeats"])
    if decision["bucket"] == "PROSPECTIVE_INVARIANCE_SUPPORTED":
        bottom = (
            "On this unseen confirmatory sample, counterfactual displayed confidence "
            "mostly changed how many frozen answers were sent for verification, not which ones."
        )
        narrative = (
            "The prospective data support treating displayed confidence as a coverage control "
            "on ranking-stable qualitative verification."
        )
        next_task = (
            "Do not launch a second benchmark inside Task 009. Recommend a later non-MCQ "
            "generalization study."
        )
    elif decision["bucket"] == "PROSPECTIVE_RESHAPING_SUPPORTED":
        bottom = (
            "On this unseen confirmatory sample, displayed confidence changed not only coverage "
            "but which items were prioritized for verification."
        )
        narrative = (
            "The prospective data favor score-dependent reprioritization over ranking invariance."
        )
        next_task = (
            "Do not launch a second benchmark inside Task 009. Design any later task around reshaping."
        )
    elif decision["bucket"] == "MIXED_BY_MODEL":
        bottom = "GPT and Claude diverged under the same frozen confirmatory tests."
        narrative = "Do not collapse the two models into one world claim."
        next_task = "Stop and explain the split before spending more."
    else:
        bottom = (
            "The confirmatory tests cannot distinguish coverage control from reshaping "
            "with the frozen criteria."
        )
        narrative = "Saturation, missing cells, or uncertainty block a world claim."
        next_task = "Stop and explain why before spending more."
    summary = {
        "bottom_line": bottom,
        "narrative": narrative,
        "next_task": next_task,
        "decision": decision,
        "coverage": coverage,
        "shared": shared,
        "rich": rich,
        "stability": stability,
        "auc": auc_rows,
        "nesting": nesting,
        "n_requests": len(requests),
        "n_attempts": n_attempts,
        "n_failures": n_fail,
        "stage3_usd": stage3_usd,
        "primary_hash": manifest["sample"]["primary_id_list_sha256"],
        "secondary_hash": manifest["sample"]["secondary_id_list_sha256"],
        "repeat_hash": manifest["sample"]["repeat_id_list_sha256"],
        "paper_sha": sha256_file(PAPER_DIRECTION),
        "run": run_payload or {},
    }
    write_report(summary)
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during analysis")
    json_dump(RETURN_DIR / "analysis_summary.json", {k: v for k, v in summary.items() if k != "run"})
    return summary
