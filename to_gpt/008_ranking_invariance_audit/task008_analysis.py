"""Task 008 quantitative lanes. Zero API calls. Question ID is the resampling unit."""

from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from .task008_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CHECKPOINT_A_CELLS,
    DIFFICULTY_FEATURES,
    GEMINI_GROK_RESULTS,
    GPT_CLAUDE,
    MATERIAL_AUROC,
    MATERIAL_LOGLOSS,
    MATERIAL_REVERSAL,
    MERGED100,
    MODEL_LABELS,
    REFERENCE_SCORE,
    SATURATION_HIGH,
    SATURATION_LOW,
    SCORE_PAIRS,
    STAKES_FAMILIES,
    STUDY1_PRIMARY,
    VISIBLE_CONDITIONS,
    VISIBLE_SCORES,
    coerce_qual_row,
    load_csv,
)

EPS = 1e-6


def _ci(vals: Sequence[float]) -> tuple[float, float]:
    clean = sorted(float(v) for v in vals if math.isfinite(v))
    if not clean:
        return float("nan"), float("nan")
    lo = clean[int(0.025 * (len(clean) - 1))]
    hi = clean[int(0.975 * (len(clean) - 1))]
    return lo, hi


def _mean(vals: Sequence[float]) -> float:
    clean = [float(v) for v in vals if math.isfinite(v)]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    if np.all(np.isnan(scores)):
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _safe_log_loss(y: Sequence[int], p: Sequence[float]) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    p_arr = np.clip(np.asarray(list(p), dtype=float), EPS, 1.0 - EPS)
    if len(y_arr) == 0:
        return float("nan")
    return float(log_loss(y_arr, p_arr, labels=[0, 1]))


def _invlogit(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit(
    X: np.ndarray, y: np.ndarray, *, C: float = np.inf
) -> LogisticRegression | None:
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    for cand in (C, 1e6 if not math.isfinite(C) else C, 1.0):
        try:
            clf = LogisticRegression(C=cand, solver="lbfgs", max_iter=4000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X, y)
            return clf
        except Exception:
            continue
    return None


def fit_logit_original_scale(
    X: np.ndarray, y: np.ndarray, *, C: float = 1.0
) -> np.ndarray | None:
    """Ridge on standardized columns; return coefficients on the original X scale."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    sd = np.clip(X.std(axis=0), 1e-6, None)
    Xs = (X - mean) / sd
    clf = fit_logit(Xs, y, C=C)
    if clf is None:
        return None
    return clf.coef_[0] / sd


def predict_p(clf: LogisticRegression | None, X: np.ndarray, fallback: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(X)
    if clf is None:
        return np.full(n, fallback)
    return np.clip(clf.predict_proba(X)[:, 1], EPS, 1.0 - EPS)


def saturation_label(rate: float) -> str:
    if not math.isfinite(rate):
        return "unknown"
    if rate <= SATURATION_LOW:
        return "low_coverage_saturated"
    if rate >= SATURATION_HIGH:
        return "high_coverage_saturated"
    return "identifiable"


def load_qualitative_grid() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = load_csv(MERGED100)
    rows = [coerce_qual_row(row, source=str(row.get("source") or "merged")) for row in raw]
    feat_rows = load_csv(DIFFICULTY_FEATURES)
    feats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in feat_rows:
        key = (row["question_id"], row["target_model"])
        feats[key] = row
    joined = []
    missing = []
    for row in rows:
        feat = feats.get((row["question_id"], row["model_alias"]))
        if feat is None:
            missing.append((row["question_id"], row["model_alias"]))
            continue
        item = dict(row)
        other_n = int(feat["other_n_correct"])
        item.update(
            {
                "other_n_correct": other_n,
                "other_frac_wrong": 1.0 - other_n / 3.0,
                "difficulty_bin": feat["difficulty_bin"],
                "category": feat["category"],
                "n_choices": int(feat["n_choices"]),
                "question_char_len": int(feat["question_char_len"]),
                "target_correct": int(feat[
                    {
                        "openai_gpt56_sol": "gpt_correct",
                        "anthropic_sonnet5": "claude_correct",
                    }[row["model_alias"]]
                ]),
                "is_wrong": 1 - int(row["stage1_correct"]),
            }
        )
        if int(item["target_correct"]) != int(bool(row["stage1_correct"])):
            raise RuntimeError(
                f"correctness mismatch {row['question_id']}/{row['model_alias']}"
            )
        joined.append(item)
    if missing:
        raise RuntimeError(f"difficulty join incomplete: {missing[:8]}")
    qids = sorted({row["question_id"] for row in joined})
    validation = {
        "n_rows": len(joined),
        "n_questions": len(qids),
        "n_models": sorted({row["model_alias"] for row in joined}),
        "n_families": sorted({row["family"] for row in joined}),
        "n_conditions": sorted({row["score_condition"] for row in joined}),
        "expected_rows": 1600,
        "complete": len(joined) == 1600 and len(qids) == 100,
        "leakage_rule": (
            "GPT difficulty uses Claude/Gemini/Grok Stage-1 correctness; "
            "Claude difficulty uses GPT/Gemini/Grok. Target own correctness is never in difficulty."
        ),
    }
    if not validation["complete"]:
        raise RuntimeError(f"qualitative grid incomplete: {validation}")
    return joined, validation


def load_gemini_grok() -> list[dict[str, Any]]:
    raw = load_csv(GEMINI_GROK_RESULTS)
    return [coerce_qual_row(row, source=str(row.get("source") or "007b")) for row in raw]


def _cell(
    rows: Sequence[Mapping[str, Any]], model: str, family: str, condition: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row["model_alias"] == model
        and row["family"] == family
        and row["score_condition"] == condition
    ]


def _by_question(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["question_id"]: dict(row) for row in rows}


def roc_and_coverage_points(
    rows: Sequence[Mapping[str, Any]], *, sample_role: str = "primary_n100"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roc_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    models = sorted({row["model_alias"] for row in rows})
    families = sorted({row["family"] for row in rows})
    conditions = sorted({row["score_condition"] for row in rows})
    rng = random.Random(BOOTSTRAP_SEED)
    for model in models:
        for family in families:
            for condition in conditions:
                cell = _cell(rows, model, family, condition)
                if not cell:
                    continue
                by_q = _by_question(cell)
                qids = sorted(by_q)
                n_wrong = sum(int(row["is_wrong"]) if "is_wrong" in row else 1 - int(row["stage1_correct"]) for row in cell)
                if "is_wrong" not in cell[0]:
                    for row in cell:
                        row["is_wrong"] = 1 - int(row["stage1_correct"])
                    n_wrong = sum(int(row["is_wrong"]) for row in cell)
                n_correct = len(cell) - n_wrong
                verify_rate = _mean(row["verify"] for row in cell)
                tpr = _mean(row["verify"] for row in cell if int(row["is_wrong"]) == 1)
                fpr = _mean(row["verify"] for row in cell if int(row["is_wrong"]) == 0)
                errors_caught = sum(int(row["verify"]) * int(row["is_wrong"]) for row in cell)
                n_verify = sum(int(row["verify"]) for row in cell)

                def stats(sub: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float]:
                    vr = _mean(r["verify"] for r in sub)
                    tp = _mean(r["verify"] for r in sub if int(r["is_wrong"]) == 1)
                    fp = _mean(r["verify"] for r in sub if int(r["is_wrong"]) == 0)
                    caught = _mean(
                        int(r["verify"]) for r in sub if int(r["is_wrong"]) == 1
                    )
                    return vr, tp, fp, caught

                boot_v, boot_tpr, boot_fpr, boot_catch = [], [], [], []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    draw = [by_q[qids[rng.randrange(len(qids))]] for _ in qids]
                    vr, tp, fp, caught = stats(draw)
                    boot_v.append(vr)
                    boot_tpr.append(tp)
                    boot_fpr.append(fp)
                    boot_catch.append(caught)
                v_lo, v_hi = _ci(boot_v)
                tpr_lo, tpr_hi = _ci(boot_tpr)
                fpr_lo, fpr_hi = _ci(boot_fpr)
                c_lo, c_hi = _ci(boot_catch)
                displayed = cell[0].get("displayed_confidence")
                rec = {
                    "model_alias": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "family": family,
                    "score_condition": condition,
                    "displayed_confidence": displayed,
                    "n_questions": len(qids),
                    "n_wrong": n_wrong,
                    "n_correct": n_correct,
                    "n_verify": n_verify,
                    "verify_rate": verify_rate,
                    "verify_rate_ci_lower": v_lo,
                    "verify_rate_ci_upper": v_hi,
                    "tpr": tpr,
                    "tpr_ci_lower": tpr_lo,
                    "tpr_ci_upper": tpr_hi,
                    "fpr": fpr,
                    "fpr_ci_lower": fpr_lo,
                    "fpr_ci_upper": fpr_hi,
                    "errors_caught": errors_caught,
                    "frac_errors_caught": tpr,
                    "frac_errors_caught_ci_lower": c_lo,
                    "frac_errors_caught_ci_upper": c_hi,
                    "saturation": saturation_label(verify_rate),
                    "sample_role": sample_role,
                    "note": "FPR=P(VERIFY|correct); TPR=P(VERIFY|wrong). Question-bootstrap 95% CI.",
                }
                roc_rows.append(rec)
                risk_rows.append(
                    {
                        **rec,
                        "coverage": verify_rate,
                        "risk_caught": tpr,
                    }
                )
    return roc_rows, risk_rows


def nesting_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    rng = random.Random(BOOTSTRAP_SEED)
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            by_q: dict[str, dict[float, Mapping[str, Any]]] = defaultdict(dict)
            for row in rows:
                if row["model_alias"] != model or row["family"] != family:
                    continue
                if row["displayed_confidence"] is None:
                    continue
                by_q[row["question_id"]][float(row["displayed_confidence"])] = row
            qids = sorted(by_q)
            trajectories = {"VVV": 0, "VVU": 0, "VUU": 0, "UUU": 0, "non_nested": 0}
            for qid in qids:
                scores = by_q[qid]
                if not all(s in scores for s in VISIBLE_SCORES):
                    continue
                pattern = "".join("V" if int(scores[s]["verify"]) else "U" for s in VISIBLE_SCORES)
                if pattern in trajectories:
                    trajectories[pattern] += 1
                else:
                    trajectories["non_nested"] += 1
            nested_ok = sum(trajectories[k] for k in ("VVV", "VVU", "VUU", "UUU"))
            for lo, hi in SCORE_PAIRS:
                vv = vu = uu = uv = 0
                for qid in qids:
                    scores = by_q[qid]
                    if lo not in scores or hi not in scores:
                        continue
                    a = int(scores[lo]["verify"])
                    b = int(scores[hi]["verify"])
                    if a == 1 and b == 1:
                        vv += 1
                    elif a == 1 and b == 0:
                        vu += 1
                    elif a == 0 and b == 0:
                        uu += 1
                    else:
                        uv += 1
                n = vv + vu + uu + uv
                reversal = uv / n if n else float("nan")
                boots = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    draw = [qids[rng.randrange(len(qids))] for _ in qids]
                    uv_b = 0
                    n_b = 0
                    for qid in draw:
                        scores = by_q[qid]
                        if lo not in scores or hi not in scores:
                            continue
                        a = int(scores[lo]["verify"])
                        b = int(scores[hi]["verify"])
                        n_b += 1
                        if a == 0 and b == 1:
                            uv_b += 1
                    if n_b:
                        boots.append(uv_b / n_b)
                lo_ci, hi_ci = _ci(boots)
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "family": family,
                        "from_score": lo,
                        "to_score": hi,
                        "n_questions": n,
                        "n_VV": vv,
                        "n_VU": vu,
                        "n_UU": uu,
                        "n_UV": uv,
                        "reversal_rate": reversal,
                        "reversal_ci_lower": lo_ci,
                        "reversal_ci_upper": hi_ci,
                        "material_reversal": bool(
                            math.isfinite(reversal)
                            and reversal >= MATERIAL_REVERSAL
                            and lo_ci > 0
                        ),
                        "n_perfectly_nested_trajectories": nested_ok,
                        "n_non_nested_trajectories": trajectories["non_nested"],
                        "trajectory_VVV": trajectories["VVV"],
                        "trajectory_VVU": trajectories["VVU"],
                        "trajectory_VUU": trajectories["VUU"],
                        "trajectory_UUU": trajectories["UUU"],
                        "note": (
                            "As displayed score rises, nested VERIFY sets imply UV=0. "
                            "Nesting supports but does not prove ranking invariance."
                        ),
                    }
                )
    return output


def difficulty_logit_interactions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    rng = random.Random(BOOTSTRAP_SEED)
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            visible = [
                row
                for row in rows
                if row["model_alias"] == model
                and row["family"] == family
                and row["score_condition"] in VISIBLE_CONDITIONS
            ]
            by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in visible:
                by_q[row["question_id"]].append(row)
            qids = sorted(by_q)
            y = np.array([int(row["verify"]) for row in visible], dtype=int)
            score = np.array([float(row["displayed_confidence"]) for row in visible])
            diff = np.array([float(row["other_n_correct"]) for row in visible])
            score_c = score - REFERENCE_SCORE
            diff_c = diff - float(np.mean(diff))
            X = np.column_stack([score_c, diff_c, score_c * diff_c])
            coefs = fit_logit_original_scale(X, y, C=1.0)
            if coefs is None:
                coefs = np.array([float("nan")] * 3)
            slope_at = {
                s: float(coefs[1] + coefs[2] * (s - REFERENCE_SCORE)) for s in VISIBLE_SCORES
            }
            boots = {key: [] for key in ("gamma", "b070", "b090", "b099", "delta_ll")}
            oof_base, oof_int = _grouped_cv_interaction(visible)
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw_ids = [qids[rng.randrange(len(qids))] for _ in qids]
                sub = [row for qid in draw_ids for row in by_q[qid]]
                yy = np.array([int(row["verify"]) for row in sub], dtype=int)
                if len(np.unique(yy)) < 2:
                    continue
                ss = np.array([float(row["displayed_confidence"]) for row in sub])
                dd = np.array([float(row["other_n_correct"]) for row in sub])
                sc = ss - REFERENCE_SCORE
                dc = dd - float(np.mean(dd))
                XX = np.column_stack([sc, dc, sc * dc])
                m_coefs = fit_logit_original_scale(XX, yy, C=1.0)
                if m_coefs is None:
                    continue
                g = float(m_coefs[2])
                b = float(m_coefs[1])
                boots["gamma"].append(g)
                boots["b070"].append(b + g * (0.70 - REFERENCE_SCORE))
                boots["b090"].append(b)
                boots["b099"].append(b + g * (0.99 - REFERENCE_SCORE))
            g_lo, g_hi = _ci(boots["gamma"])
            rates = {
                s: _mean(
                    r["verify"]
                    for r in visible
                    if abs(float(r["displayed_confidence"]) - s) < 1e-9
                )
                for s in VISIBLE_SCORES
            }
            saturated_any = any(
                saturation_label(rates[s]) != "identifiable" for s in VISIBLE_SCORES
            )
            gamma = float(coefs[2])
            cv_helps = bool(
                math.isfinite(oof_int)
                and math.isfinite(oof_base)
                and (oof_int - oof_base) <= -MATERIAL_LOGLOSS
            )
            gamma_excludes0 = bool(
                math.isfinite(g_lo)
                and math.isfinite(g_hi)
                and ((g_lo > 0 and g_hi > 0) or (g_lo < 0 and g_hi < 0))
            )
            material = bool(cv_helps and gamma_excludes0)
            output.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "family": family,
                    "n_rows": len(visible),
                    "n_questions": len(qids),
                    "reference_score": REFERENCE_SCORE,
                    "difficulty": "other_models_correct_0to3",
                    "coef_score_at_ref": float(coefs[0]),
                    "beta_at_ref": float(coefs[1]),
                    "gamma_score_x_difficulty": gamma,
                    "gamma_ci_lower": g_lo,
                    "gamma_ci_upper": g_hi,
                    "slope_at_0.70": slope_at[0.70],
                    "slope_at_0.70_ci_lower": _ci(boots["b070"])[0],
                    "slope_at_0.70_ci_upper": _ci(boots["b070"])[1],
                    "slope_at_0.90": slope_at[0.90],
                    "slope_at_0.90_ci_lower": _ci(boots["b090"])[0],
                    "slope_at_0.90_ci_upper": _ci(boots["b090"])[1],
                    "slope_at_0.99": slope_at[0.99],
                    "slope_at_0.99_ci_lower": _ci(boots["b099"])[0],
                    "slope_at_0.99_ci_upper": _ci(boots["b099"])[1],
                    "cv_logloss_no_interaction": oof_base,
                    "cv_logloss_interaction": oof_int,
                    "cv_delta_logloss": (
                        oof_int - oof_base
                        if math.isfinite(oof_int) and math.isfinite(oof_base)
                        else float("nan")
                    ),
                    "material_logit_interaction": material,
                    "saturated_operating_point_present": saturated_any,
                    "verify_rate_0.70": rates[0.70],
                    "verify_rate_0.90": rates[0.90],
                    "verify_rate_0.99": rates[0.99],
                    "interpretation": (
                        "Negative beta means easier items (more other-correct) are less often verified. "
                        "Gamma is the change in that logit slope as displayed score rises. "
                        "Coefficients: ridge C=1 on standardized covariates, mapped back to centered score/difficulty. "
                        "Materiality requires grouped-CV log-loss improvement of at least 0.01 AND a gamma CI excluding 0. "
                        "Unpenalized MLE is unstable under saturation; do not interpret exploded unpenalized gammas."
                    ),
                }
            )
            # secondary covariates robustness on full visible set, in-sample only
            cats = np.array([row["category"] for row in visible])
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_x = enc.fit_transform(cats.reshape(-1, 1))
            extra = np.column_stack(
                [
                    np.array([float(row["question_char_len"]) for row in visible]),
                    np.array([float(row["n_choices"]) for row in visible]),
                ]
            )
            extra = (extra - extra.mean(axis=0)) / np.clip(extra.std(axis=0), 1e-6, None)
            X2 = np.column_stack([X, extra, cat_x[:, 1:]])
            coefs2 = fit_logit_original_scale(X2, y, C=1.0)
            output[-1]["gamma_with_category_length_choices"] = (
                float(coefs2[2]) if coefs2 is not None else float("nan")
            )
    return output


def _grouped_cv_interaction(visible: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    qids = np.array([row["question_id"] for row in visible])
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    score = np.array([float(row["displayed_confidence"]) for row in visible])
    diff = np.array([float(row["other_n_correct"]) for row in visible], dtype=float)
    score_c = score - REFERENCE_SCORE
    unique = np.unique(qids)
    if len(unique) < 5 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    gkf = GroupKFold(n_splits=5)
    p0 = np.zeros(len(y))
    p1 = np.zeros(len(y))
    for train, test in gkf.split(y, y, groups=qids):
        dmean = float(np.mean(diff[train]))
        dc_tr = diff[train] - dmean
        dc_te = diff[test] - dmean
        X0_tr = np.column_stack([score_c[train], dc_tr])
        X0_te = np.column_stack([score_c[test], dc_te])
        X1_tr = np.column_stack([X0_tr, score_c[train] * dc_tr])
        X1_te = np.column_stack([X0_te, score_c[test] * dc_te])
        fallback = float(np.mean(y[train]))
        m0 = fit_logit(X0_tr, y[train])
        m1 = fit_logit(X1_tr, y[train])
        p0[test] = predict_p(m0, X0_te, fallback)
        p1[test] = predict_p(m1, X1_te, fallback)
    return _safe_log_loss(y, p0), _safe_log_loss(y, p1)


def _grouped_cv_models(visible: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    qids = np.array([row["question_id"] for row in visible])
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    score = np.array([float(row["displayed_confidence"]) for row in visible])
    diff = np.array([float(row["other_n_correct"]) for row in visible], dtype=float)
    cond = np.array([row["score_condition"] for row in visible])
    if len(np.unique(qids)) < 5 or len(np.unique(y)) < 2:
        return {k: float("nan") for k in ("intercept", "difficulty", "shared_proxy", "interaction")}
    enc = OneHotEncoder(sparse_output=False, drop="first")
    gkf = GroupKFold(n_splits=5)
    store = {k: np.zeros(len(y)) for k in ("intercept", "difficulty", "interaction")}
    for train, test in gkf.split(y, y, groups=qids):
        fallback = float(np.mean(y[train]))
        enc.fit(cond[train].reshape(-1, 1))
        C_tr = enc.transform(cond[train].reshape(-1, 1))
        C_te = enc.transform(cond[test].reshape(-1, 1))
        dmean = float(np.mean(diff[train]))
        d_tr = (diff[train] - dmean).reshape(-1, 1)
        d_te = (diff[test] - dmean).reshape(-1, 1)
        s_tr = (score[train] - REFERENCE_SCORE).reshape(-1, 1)
        s_te = (score[test] - REFERENCE_SCORE).reshape(-1, 1)
        store["intercept"][test] = predict_p(fit_logit(C_tr, y[train]), C_te, fallback)
        store["difficulty"][test] = predict_p(
            fit_logit(np.hstack([C_tr, d_tr]), y[train]), np.hstack([C_te, d_te]), fallback
        )
        store["interaction"][test] = predict_p(
            fit_logit(np.hstack([C_tr, d_tr, s_tr * d_tr]), y[train]),
            np.hstack([C_te, d_te, s_te * d_te]),
            fallback,
        )
    return {k: _safe_log_loss(y, p) for k, p in store.items()}


def shared_vs_condition_sensitive(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shared_out = []
    rich_out = []
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            visible = [
                row
                for row in rows
                if row["model_alias"] == model
                and row["family"] == family
                and row["score_condition"] in VISIBLE_CONDITIONS
            ]
            cv = _grouped_cv_models(visible)
            y = np.array([int(row["verify"]) for row in visible], dtype=int)
            score = np.array([float(row["displayed_confidence"]) for row in visible])
            diff = np.array([float(row["other_n_correct"]) for row in visible], dtype=float)
            cond = np.array([row["score_condition"] for row in visible])
            qids = np.array([row["question_id"] for row in visible])
            enc_c = OneHotEncoder(sparse_output=False, drop="first")
            enc_i = OneHotEncoder(sparse_output=False, drop="first")
            C = enc_c.fit_transform(cond.reshape(-1, 1))
            I = enc_i.fit_transform(qids.reshape(-1, 1))
            dmean = float(np.mean(diff))
            d = (diff - dmean).reshape(-1, 1)
            s = (score - REFERENCE_SCORE).reshape(-1, 1)
            fallback = float(np.mean(y))
            # Rasch-style: condition intercepts + L2 item intercepts
            rasch = LogisticRegression(C=10.0, solver="lbfgs", max_iter=4000)
            richer = LogisticRegression(C=10.0, solver="lbfgs", max_iter=4000)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rasch.fit(np.hstack([C, I]), y)
                p_s = np.clip(rasch.predict_proba(np.hstack([C, I]))[:, 1], EPS, 1 - EPS)
                ll_s = _safe_log_loss(y, p_s)
            except Exception:
                p_s = np.full(len(y), fallback)
                ll_s = _safe_log_loss(y, p_s)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    richer.fit(np.hstack([C, I, d, s * d]), y)
                p_r = np.clip(richer.predict_proba(np.hstack([C, I, d, s * d]))[:, 1], EPS, 1 - EPS)
                ll_r = _safe_log_loss(y, p_r)
            except Exception:
                p_r = p_s
                ll_r = ll_s
            n = len(y)
            extra_params = 2
            bic_s = n * ll_s * math.log(2)  # placeholder scale; report AIC-style
            aic_s = 2 * (C.shape[1] + I.shape[1] + 1) + 2 * n * ll_s
            aic_r = 2 * (C.shape[1] + I.shape[1] + 1 + extra_params) + 2 * n * ll_r
            delta_ll = ll_r - ll_s
            material = bool(math.isfinite(delta_ll) and delta_ll <= -MATERIAL_LOGLOSS)
            shared_out.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "family": family,
                    "model_name": "shared_rasch_approx",
                    "formula": "logit P(VERIFY)=alpha_condition + u_item (L2 C=10 item intercepts)",
                    "n_rows": n,
                    "n_questions": len(set(qids)),
                    "in_sample_logloss": ll_s,
                    "cv_logloss_condition_intercept": cv["intercept"],
                    "cv_logloss_condition_plus_difficulty": cv["difficulty"],
                    "aic_like": aic_s,
                    "note": (
                        "Item intercepts do not transfer to new questions. "
                        "Grouped-CV rows are the transferable comparison."
                    ),
                }
            )
            rich_out.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "family": family,
                    "model_name": "condition_sensitive_difficulty_slope",
                    "formula": "logit P(VERIFY)=alpha_condition + u_item + beta*diff + gamma*score*diff",
                    "n_rows": n,
                    "n_questions": len(set(qids)),
                    "in_sample_logloss": ll_r,
                    "in_sample_delta_vs_shared": delta_ll,
                    "cv_logloss_interaction": cv["interaction"],
                    "cv_delta_vs_difficulty_only": cv["interaction"] - cv["difficulty"],
                    "material_cv_improvement": bool(
                        math.isfinite(cv["interaction"] - cv["difficulty"])
                        and (cv["interaction"] - cv["difficulty"]) <= -MATERIAL_LOGLOSS
                    ),
                    "material_in_sample_improvement": material,
                    "aic_like": aic_r,
                    "note": (
                        "In-sample Rasch+interaction can overfit. "
                        "Materiality uses grouped-CV log-loss on new questions, threshold 0.01."
                    ),
                }
            )
    return shared_out, rich_out


def condition_auc_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    rng = random.Random(BOOTSTRAP_SEED)
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            aucs = {}
            for condition in VISIBLE_CONDITIONS:
                cell = _cell(rows, model, family, condition)
                by_q = _by_question(cell)
                qids = sorted(by_q)
                y = np.array([int(row["is_wrong"]) for row in cell])
                s = np.array([int(row["verify"]) for row in cell], dtype=float)
                auc = _safe_auc(y, s)
                boots = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    draw = [by_q[qids[rng.randrange(len(qids))]] for _ in qids]
                    yy = np.array([int(r["is_wrong"]) for r in draw])
                    ss = np.array([int(r["verify"]) for r in draw], dtype=float)
                    boots.append(_safe_auc(yy, ss))
                lo, hi = _ci(boots)
                aucs[condition] = {
                    "auc": auc,
                    "lo": lo,
                    "hi": hi,
                    "qids": qids,
                    "by_q": by_q,
                    "saturation": saturation_label(_mean(r["verify"] for r in cell)),
                }
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "family": family,
                        "score_condition": condition,
                        "displayed_confidence": float(condition.split("_")[-1]),
                        "n_questions": len(qids),
                        "verify_rate": _mean(r["verify"] for r in cell),
                        "auroc_verify_vs_wrongness": auc,
                        "auroc_ci_lower": lo,
                        "auroc_ci_upper": hi,
                        "saturation": saturation_label(_mean(r["verify"] for r in cell)),
                        "note": "Binary-action AUROC is coarse; equal to a two-point ROC.",
                    }
                )
            for lo_s, hi_s in SCORE_PAIRS:
                c_lo = f"displayed_{lo_s:.2f}"
                c_hi = f"displayed_{hi_s:.2f}"
                qids = aucs[c_lo]["qids"]
                by_lo = aucs[c_lo]["by_q"]
                by_hi = aucs[c_hi]["by_q"]
                delta = aucs[c_hi]["auc"] - aucs[c_lo]["auc"]
                boots = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    draw = [qids[rng.randrange(len(qids))] for _ in qids]
                    yy = np.array([int(by_lo[q]["is_wrong"]) for q in draw])
                    s1 = np.array([int(by_lo[q]["verify"]) for q in draw], dtype=float)
                    s2 = np.array([int(by_hi[q]["verify"]) for q in draw], dtype=float)
                    boots.append(_safe_auc(yy, s2) - _safe_auc(yy, s1))
                dlo, dhi = _ci(boots)
                both_identifiable = (
                    aucs[c_lo]["saturation"] == "identifiable"
                    and aucs[c_hi]["saturation"] == "identifiable"
                )
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "family": family,
                        "score_condition": f"delta_{lo_s:.2f}_to_{hi_s:.2f}",
                        "displayed_confidence": "",
                        "n_questions": len(qids),
                        "verify_rate": "",
                        "auroc_verify_vs_wrongness": delta,
                        "auroc_ci_lower": dlo,
                        "auroc_ci_upper": dhi,
                        "saturation": (
                            "both_identifiable" if both_identifiable else "includes_saturated_cell"
                        ),
                        "note": "ΔAUROC (higher score minus lower). Ignore pairs that include a saturated cell.",
                    }
                )
    return output


def common_curve_analysis(
    roc_rows: Sequence[Mapping[str, Any]], nest_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    by = defaultdict(list)
    for row in roc_rows:
        if row["score_condition"] not in VISIBLE_CONDITIONS:
            continue
        if row.get("sample_role") != "primary_n100":
            continue
        by[(row["model_alias"], row["family"])].append(row)
    nest_by = {
        (row["model_alias"], row["family"], float(row["from_score"]), float(row["to_score"])): row
        for row in nest_rows
    }
    for (model, family), pts in sorted(by.items()):
        pts = sorted(pts, key=lambda r: float(r["displayed_confidence"]))
        fprs = [float(p["fpr"]) for p in pts]
        tprs = [float(p["tpr"]) for p in pts]
        coverages = [float(p["verify_rate"]) for p in pts]
        # As score rises, coverage should fall; FPR/TPR should be nonincreasing under nested thresholds.
        mono_fpr = all(fprs[i] + 1e-12 >= fprs[i + 1] for i in range(len(fprs) - 1))
        mono_tpr = all(tprs[i] + 1e-12 >= tprs[i + 1] for i in range(len(tprs) - 1))
        slopes = []
        for i in range(len(pts) - 1):
            dx = fprs[i] - fprs[i + 1]
            dy = tprs[i] - tprs[i + 1]
            slopes.append(dy / dx if abs(dx) > 1e-12 else float("nan"))
        concave = True
        finite_slopes = [s for s in slopes if math.isfinite(s)]
        if len(finite_slopes) >= 2:
            concave = finite_slopes[0] + 1e-8 >= finite_slopes[1]
        total_uv = sum(
            int(nest_by[(model, family, lo, hi)]["n_UV"])
            for lo, hi in ((0.70, 0.90), (0.90, 0.99))
            if (model, family, lo, hi) in nest_by
        )
        identifiable = [
            p for p in pts if saturation_label(float(p["verify_rate"])) == "identifiable"
        ]
        output.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "family": family,
                "n_visible_points": len(pts),
                "n_identifiable_points": len(identifiable),
                "fpr_sequence": ",".join(f"{x:.3f}" for x in fprs),
                "tpr_sequence": ",".join(f"{x:.3f}" for x in tprs),
                "coverage_sequence": ",".join(f"{x:.3f}" for x in coverages),
                "monotone_fpr_as_score_rises": mono_fpr,
                "monotone_tpr_as_score_rises": mono_tpr,
                "piecewise_roc_concave": concave,
                "adjacent_uv_count": total_uv,
                "formal_test": "WEAK_THREE_POINT_GEOMETRY",
                "compatible_with_shared_ranking": bool(mono_fpr and mono_tpr and total_uv <= 2),
                "note": (
                    "Three binary operating points cannot strongly test a common ROC. "
                    "Monotone nested (FPR,TPR) is compatible with a threshold shift; "
                    "it does not prove a unique latent ranking."
                ),
            }
        )
    return output


def latent_item_prediction(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    rng = random.Random(BOOTSTRAP_SEED)
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            by_q: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
            for row in rows:
                if row["model_alias"] != model or row["family"] != family:
                    continue
                if row["score_condition"] not in VISIBLE_CONDITIONS:
                    continue
                by_q[row["question_id"]][row["score_condition"]] = row
            qids = sorted(qid for qid, d in by_q.items() if len(d) == 3)
            if len(qids) != 100:
                output.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "status": "INSUFFICIENT_FOR_LATENT_RANK_TEST",
                        "reason": f"expected 100 complete items, found {len(qids)}",
                    }
                )
                continue
            for held in VISIBLE_CONDITIONS:
                train_conds = [c for c in VISIBLE_CONDITIONS if c != held]
                y = []
                p_int = []
                p_diff = []
                p_u = []
                p_u_diff = []
                p_rich = []
                u_scores = []
                y_hold = []
                for qid in qids:
                    recs = by_q[qid]
                    hold = recs[held]
                    train = [recs[c] for c in train_conds]
                    y_h = int(hold["verify"])
                    y_hold.append(y_h)
                    train_rate = _mean(r["verify"] for r in train)
                    u = train_rate - _mean(
                        _mean(by_q[q][c]["verify"] for c in train_conds) for q in qids
                    )
                    u_scores.append(u)
                    y.append(y_h)
                y_arr = np.array(y_hold, dtype=int)
                fallback = float(np.mean(y_arr))
                # intercept only
                p_int = np.full(len(qids), fallback)
                # difficulty only
                d = np.array([float(by_q[q][held]["other_n_correct"]) for q in qids])
                m_d = fit_logit(d.reshape(-1, 1), y_arr)
                p_diff = predict_p(m_d, d.reshape(-1, 1), fallback)
                # shared propensity
                u = np.array(u_scores).reshape(-1, 1)
                m_u = fit_logit(u, y_arr)
                p_u = predict_p(m_u, u, fallback)
                # shared + difficulty
                Xud = np.column_stack([u.ravel(), d])
                m_ud = fit_logit(Xud, y_arr)
                p_ud = predict_p(m_ud, Xud, fallback)
                # richer: u + difficulty + difficulty*(held score)
                held_score = float(held.split("_")[-1])
                Xr = np.column_stack([u.ravel(), d, d * (held_score - REFERENCE_SCORE)])
                m_r = fit_logit(Xr, y_arr)
                p_r = predict_p(m_r, Xr, fallback)
                # These in-sample fits of the held-out mapping still use held-out y for the
                # intercept/slope from u_i -> y_held. For a stricter test, use a question
                # GroupKFold on the mapping.
                def oof_from_X(X: np.ndarray) -> np.ndarray:
                    pred = np.zeros(len(y_arr))
                    gkf = GroupKFold(n_splits=5)
                    groups = np.array(qids)
                    for tr, te in gkf.split(y_arr, y_arr, groups=groups):
                        fb = float(np.mean(y_arr[tr]))
                        if X.shape[1] == 0:
                            pred[te] = fb
                            continue
                        pred[te] = predict_p(fit_logit(X[tr], y_arr[tr]), X[te], fb)
                    return pred

                p_int_oof = np.full(len(y_arr), fallback)
                # intercept OOF is just train mean in each fold
                pred_int = np.zeros(len(y_arr))
                gkf = GroupKFold(n_splits=5)
                groups = np.array(qids)
                for tr, te in gkf.split(y_arr, y_arr, groups=groups):
                    pred_int[te] = float(np.mean(y_arr[tr]))
                p_int_oof = pred_int
                p_diff_oof = oof_from_X(d.reshape(-1, 1))
                p_u_oof = oof_from_X(u)
                p_ud_oof = oof_from_X(Xud)
                p_r_oof = oof_from_X(Xr)
                auc_u = _safe_auc(y_arr, u.ravel())
                ll = {
                    "intercept": _safe_log_loss(y_arr, p_int_oof),
                    "difficulty": _safe_log_loss(y_arr, p_diff_oof),
                    "shared_propensity": _safe_log_loss(y_arr, p_u_oof),
                    "shared_plus_difficulty": _safe_log_loss(y_arr, p_ud_oof),
                    "condition_sensitive": _safe_log_loss(y_arr, p_r_oof),
                }
                # bootstrap AUC of u
                auc_boots = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    idx = [rng.randrange(len(qids)) for _ in qids]
                    auc_boots.append(_safe_auc(y_arr[idx], u.ravel()[idx]))
                alo, ahi = _ci(auc_boots)
                rich_delta = ll["condition_sensitive"] - ll["shared_plus_difficulty"]
                shared_delta = ll["shared_propensity"] - ll["intercept"]
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "family": family,
                        "held_out_condition": held,
                        "held_out_score": held_score,
                        "n_questions": len(qids),
                        "status": "OK",
                        "u_definition": "mean VERIFY on the other two visible scores, de-meaned",
                        "auroc_u_for_heldout_verify": auc_u,
                        "auroc_u_ci_lower": alo,
                        "auroc_u_ci_upper": ahi,
                        "oof_logloss_intercept": ll["intercept"],
                        "oof_logloss_difficulty": ll["difficulty"],
                        "oof_logloss_shared_propensity": ll["shared_propensity"],
                        "oof_logloss_shared_plus_difficulty": ll["shared_plus_difficulty"],
                        "oof_logloss_condition_sensitive": ll["condition_sensitive"],
                        "delta_rich_minus_shared_plus_difficulty": rich_delta,
                        "delta_shared_minus_intercept": shared_delta,
                        "shared_beats_intercept": bool(
                            ll["shared_propensity"] < ll["intercept"] - 0.002
                        ),
                        "rich_materially_beats_shared": bool(
                            math.isfinite(rich_delta) and rich_delta <= -MATERIAL_LOGLOSS
                        ),
                        "note": (
                            "One generation per cell: u_i is a noisy two-observation propensity. "
                            "OOF is 5-fold by question on the held-out mapping. "
                            "rich vs shared+difficulty is the condition-sensitive test; "
                            "do not credit difficulty itself as reshaping."
                        ),
                    }
                )
    return output


def workshop_points() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    study1 = load_csv(STUDY1_PRIMARY)
    ckpt = load_csv(CHECKPOINT_A_CELLS)
    s1_rows = []
    by: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in study1:
        if str(row.get("ok")).lower() not in {"1", "true"}:
            continue
        model = row["model_alias"]
        L = str(row["L"])
        cond = row["display_condition"]
        qid = row["question_id"]
        verify = int(str(row["parsed_action"]) == "VERIFY_FIRST")
        wrong = 0 if str(row["stage1_correct"]).lower() in {"1", "true"} else 1
        by[(model, L, cond, row.get("displayed_confidence") or "")].append(
            {
                "question_id": qid,
                "verify": verify,
                "is_wrong": wrong,
            }
        )
    for (model, L, cond, displayed), cell in sorted(by.items()):
        n_wrong = sum(r["is_wrong"] for r in cell)
        s1_rows.append(
            {
                "source": "study1_numeric",
                "model_alias": model,
                "model_label": MODEL_LABELS.get(model, model),
                "L": L,
                "display_condition": cond,
                "displayed_confidence": displayed,
                "n": len(cell),
                "n_wrong": n_wrong,
                "verify_rate": _mean(r["verify"] for r in cell),
                "tpr": _mean(r["verify"] for r in cell if r["is_wrong"] == 1),
                "fpr": _mean(r["verify"] for r in cell if r["is_wrong"] == 0),
            }
        )
    ckpt_rows = []
    for row in ckpt:
        ckpt_rows.append(
            {
                "source": "checkpoint_A_v2b",
                "model_alias": row["model_id"],
                "model_label": row["model_label"],
                "decision_owner": row["decision_owner"],
                "L": row["error_cost"],
                "hidden_coverage": float(row["hidden_verify_budget"]),
                "hidden_catch": float(row["hidden_error_catch_rate"]),
                "visible_coverage": float(row["visible_verify_rate"]),
                "visible_catch": float(row["visible_error_catch_rate"]),
                "matched_budget_conf_catch": float(row["conf_error_catch_rate"]),
                "hidden_minus_conf_catch": float(row["catch_diff_hidden_minus_conf"]),
                "hidden_minus_conf_ci_lower": float(row["catch_diff_ci_lower"]),
                "hidden_minus_conf_ci_upper": float(row["catch_diff_ci_upper"]),
                "visible_vs_raw_threshold_agreement": float(
                    row["visible_vs_raw_threshold_agreement"]
                ),
                "coverage_shift_visible_minus_hidden": float(
                    row["verify_rate_diff_visible_minus_hidden"]
                ),
            }
        )
    return s1_rows, ckpt_rows


def prospective_matrix() -> list[dict[str, Any]]:
    rows = []
    worlds = ("I_invariance", "R_reshaping")
    grids = {
        "P1": ("hidden", "true_q_visible", "0.70", "0.85", "0.95", "0.99"),
        "P2": ("hidden", "true_q_visible", "0.70", "0.85", "0.90", "0.95", "0.99"),
    }
    ns = (300, 500, 700)
    for world in worlds:
        for grid_name, conds in grids.items():
            n_cond = len(conds)
            for n in ns:
                stage1_primary = n * 2
                q1_primary = n * 2
                stage3_primary = n * 2 * n_cond
                # secondary Gemini/Grok on min(n, 200) for invariance; min(n, 300) for reshaping? task says smaller subset
                n_sec = 200 if n >= 200 else n
                stage1_sec = n_sec * 2
                q1_sec = n_sec * 2
                stage3_sec = n_sec * 2 * n_cond
                for repeat_n, gens in ((100, 3), (150, 3)):
                    extra_gen = gens - 1
                    repeat_cells = min(repeat_n, n) * 2 * n_cond * extra_gen
                    total = (
                        stage1_primary
                        + q1_primary
                        + stage3_primary
                        + stage1_sec
                        + q1_sec
                        + stage3_sec
                        + repeat_cells
                    )
                    rows.append(
                        {
                            "world": world,
                            "grid": grid_name,
                            "n_conditions": n_cond,
                            "conditions": " ".join(conds),
                            "stakes_families": 1,
                            "n_primary_questions": n,
                            "primary_models": "GPT,Claude",
                            "n_secondary_questions": n_sec,
                            "secondary_models": "Gemini,Grok",
                            "stage1_primary": stage1_primary,
                            "q1_primary": q1_primary,
                            "q2_primary": 0,
                            "stage3_primary": stage3_primary,
                            "stage1_secondary": stage1_sec,
                            "q1_secondary": q1_sec,
                            "stage3_secondary": stage3_sec,
                            "repeat_items": min(repeat_n, n),
                            "generations_per_cell": gens,
                            "repeat_extra_stage3": repeat_cells,
                            "total_scientific_calls": total,
                            "q2_included": False,
                            "fresh_stage1_and_q1": True,
                            "sample_frozen": False,
                            "note": (
                                "Counts only. Not launched. One stakes family. "
                                "q2 omitted unless later retained. Repeats are extra generations on Stage-3 cells."
                            ),
                        }
                    )
    return rows


def decide_bucket(
    nest: Sequence[Mapping[str, Any]],
    interact: Sequence[Mapping[str, Any]],
    shared: Sequence[Mapping[str, Any]],
    rich: Sequence[Mapping[str, Any]],
    latent: Sequence[Mapping[str, Any]],
    auc: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    for model in GPT_CLAUDE:
        nest_m = [r for r in nest if r["model_alias"] == model]
        int_m = [r for r in interact if r["model_alias"] == model]
        rich_m = [r for r in rich if r["model_alias"] == model]
        lat_m = [r for r in latent if r["model_alias"] == model and r.get("status") == "OK"]
        auc_delta = [
            r
            for r in auc
            if r["model_alias"] == model
            and str(r["score_condition"]).startswith("delta_")
            and r.get("saturation") == "both_identifiable"
        ]
        max_rev = max(float(r["reversal_rate"]) for r in nest_m)
        any_material_rev = any(bool(r["material_reversal"]) for r in nest_m)
        any_material_int = any(bool(r["material_logit_interaction"]) for r in int_m)
        any_cv_rich = any(bool(r["material_cv_improvement"]) for r in rich_m)
        any_loco_rich = any(bool(r.get("rich_materially_beats_shared")) for r in lat_m)
        shared_predicts = all(bool(r.get("shared_beats_intercept")) for r in lat_m) if lat_m else False
        mean_u_auc = _mean(float(r["auroc_u_for_heldout_verify"]) for r in lat_m)
        ident_int = [
            r for r in int_m if not bool(r["saturated_operating_point_present"])
        ]
        # Claude/GPT both have some saturated cells, so use interaction CIs
        gamma_excludes0 = any(
            (float(r["gamma_ci_lower"]) > 0 and float(r["gamma_ci_upper"]) > 0)
            or (float(r["gamma_ci_lower"]) < 0 and float(r["gamma_ci_upper"]) < 0)
            for r in int_m
        )
        cv_int_helps = any(
            math.isfinite(float(r["cv_delta_logloss"]))
            and float(r["cv_delta_logloss"]) <= -MATERIAL_LOGLOSS
            for r in int_m
        )
        large_auc_shift = any(
            math.isfinite(float(r["auroc_verify_vs_wrongness"]))
            and abs(float(r["auroc_verify_vs_wrongness"])) >= MATERIAL_AUROC
            and (
                (float(r["auroc_ci_lower"]) > 0 and float(r["auroc_ci_upper"]) > 0)
                or (float(r["auroc_ci_lower"]) < 0 and float(r["auroc_ci_upper"]) < 0)
            )
            for r in auc_delta
        )
        reshape_votes = sum(
            [
                any_material_rev,
                any_material_int and cv_int_helps,
                any_cv_rich,
                any_loco_rich,
                large_auc_shift,
            ]
        )
        invariance_votes = sum(
            [
                max_rev < MATERIAL_REVERSAL,
                shared_predicts,
                not any_loco_rich,
                not (any_material_int and cv_int_helps),
                not large_auc_shift,
            ]
        )
        if reshape_votes >= 3 and invariance_votes <= 2:
            label = "reshaping"
        elif invariance_votes >= 4 and reshape_votes <= 1:
            label = "invariance"
        elif ident_int == [] and gamma_excludes0 and not cv_int_helps:
            label = "ambiguous_saturation"
        else:
            label = "mixed_or_weak"
        per_model[model] = {
            "label": label,
            "max_reversal": max_rev,
            "material_reversal": any_material_rev,
            "material_logit_interaction": any_material_int,
            "cv_interaction_helps": cv_int_helps,
            "cv_rich_helps": any_cv_rich,
            "loco_rich_helps": any_loco_rich,
            "shared_predicts_heldout": shared_predicts,
            "mean_u_auc": mean_u_auc,
            "large_auc_shift": large_auc_shift,
            "reshape_votes": reshape_votes,
            "invariance_votes": invariance_votes,
        }
    gpt = per_model["openai_gpt56_sol"]["label"]
    claude = per_model["anthropic_sonnet5"]["label"]
    if gpt == "invariance" and claude == "invariance":
        bucket = "INVARIANCE_SUPPORTED_EXPLORATORY"
        favor = "WORLD 1"
        strength = "MODERATELY_FAVORS"
    elif gpt == "reshaping" and claude == "reshaping":
        bucket = "RESHAPING_SUPPORTED_EXPLORATORY"
        favor = "WORLD 2"
        strength = "MODERATELY_FAVORS"
    elif gpt != claude and {"invariance", "reshaping"} <= {gpt, claude}:
        bucket = "MIXED_OR_MODEL_SPECIFIC"
        favor = "WORLD 3"
        strength = "MODERATELY_FAVORS"
    else:
        # both mixed/weak/ambiguous, or one invariance and one mixed
        if gpt == "invariance" and claude in {"mixed_or_weak", "ambiguous_saturation"}:
            bucket = "INVARIANCE_SUPPORTED_EXPLORATORY"
            favor = "WORLD 1"
            strength = "WEAKLY_FAVORS"
        elif claude == "invariance" and gpt in {"mixed_or_weak", "ambiguous_saturation"}:
            bucket = "INVARIANCE_SUPPORTED_EXPLORATORY"
            favor = "WORLD 1"
            strength = "WEAKLY_FAVORS"
        elif "reshaping" in {gpt, claude}:
            bucket = "MIXED_OR_MODEL_SPECIFIC"
            favor = "WORLD 3"
            strength = "WEAKLY_FAVORS"
        else:
            bucket = "AMBIGUOUS_NEEDS_REPEATS"
            favor = "WORLD 1"
            strength = "AMBIGUOUS"
    return {
        "decision_bucket": bucket,
        "favored_world": favor,
        "favor_strength": strength,
        "per_model": per_model,
        "rules": {
            "material_reversal": MATERIAL_REVERSAL,
            "material_logloss": MATERIAL_LOGLOSS,
            "material_auroc": MATERIAL_AUROC,
            "unit": "question_id",
        },
    }
