"""Task 010 zero-call analyses. Do not write paperDirection or 007–009 data."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from .study1_analysis import _pyplot
from .task005_lane_a import (
    crossfit_isotonic,
    error_catch_from_weights,
    fractional_verify_weights,
)
from .task009_analyze import (
    _ci,
    _mean,
    _safe_auc,
    _safe_log_loss,
    _spearman,
    fit_logit,
    predict_p,
)
from .task009_common import (
    ADJACENT_PAIRS,
    BOOTSTRAP_SEED,
    CV_FOLDS,
    EPS,
    GPT_CLAUDE,
    MATERIAL_LOGLOSS,
    REFERENCE_SCORE,
)
from .task010_common import (
    BETWEEN_MODEL_GAMMA,
    CLAUDE_008_GAMMA,
    DEVELOPMENT_ANCHOR_SOURCE,
    FAMILY2_TARGET_RHO,
    FIGURES_DIR,
    GPT_008_GAMMA,
    INJECTION_MULTIPLIERS,
    LARGE_RANK_CORR,
    LARGE_REVERSAL,
    LATENT_BOOT,
    MATCHED_BUDGETS,
    MODEL_LABELS,
    MODERATE_RANK_CORR,
    MODERATE_REVERSAL,
    N_BOOT_POWER,
    N_SIM,
    PRIMARY_ANCHOR_GAMMA,
    ROUTING_BOOT,
    ROUTING_BUDGETS,
    ROUTING_MATERIAL_PP,
    ROUTING_MODEST_PP,
    SIM_SEED,
    SMALL_RANK_CORR,
    SMALL_REVERSAL,
    TASK009_DIR,
    TIGHT_MIN_POWER,
    VISIBLE_FIXED,
    VISIBLE_FIXED_CONDITIONS,
    load_csv,
)

GPT, CLAUDE = GPT_CLAUDE


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _as_int(value: Any) -> int:
    if value in ("", None):
        return 0
    return int(float(value))


def load_009() -> dict[str, Any]:
    stage12 = load_csv(TASK009_DIR / "stage1_q1_results.csv")
    stage3 = load_csv(TASK009_DIR / "stage3_results.csv")
    repeats = load_csv(TASK009_DIR / "repeat_results.csv")
    for row in stage12:
        row["stage1_correct"] = int(_as_bool(row["stage1_correct"]) or str(row["stage1_correct"]) == "1")
        row["q1"] = _as_float(row["q1"])
        row["question_id"] = str(row["question_id"])
        row["model_alias"] = str(row["model_alias"])
    for row in list(stage3) + list(repeats):
        row["question_id"] = str(row["question_id"])
        row["model_alias"] = str(row["model_alias"])
        row["score_condition"] = str(row["score_condition"])
        row["verify"] = _as_int(row["verify"])
        row["stage1_correct"] = _as_bool(row["stage1_correct"])
        row["displayed_confidence"] = _as_float(row.get("displayed_confidence"))
        row["reported_confidence"] = _as_float(row.get("reported_confidence"))
        row["repeat_index"] = _as_int(row.get("repeat_index"))
        row["roster"] = str(row.get("roster") or "")
    difficulty: dict[tuple[str, str], dict[str, Any]] = {}
    by_q: dict[str, dict[str, int]] = defaultdict(dict)
    q1 = {}
    for row in stage12:
        by_q[row["question_id"]][row["model_alias"]] = int(row["stage1_correct"])
        q1[(row["question_id"], row["model_alias"])] = float(row["q1"])
    for qid, models in by_q.items():
        for target in GPT_CLAUDE:
            if target not in models:
                continue
            other = CLAUDE if target == GPT else GPT
            other_correct = int(models.get(other, 0))
            difficulty[(qid, target)] = {
                "other_primary_correct": other_correct,
                "difficulty": 1.0 - other_correct,
                "target_correct": int(models[target]),
                "wrong": int(not models[target]),
            }
    return {
        "stage12": stage12,
        "stage3": stage3,
        "repeats": repeats,
        "difficulty": difficulty,
        "q1": q1,
        "correct": {qid: {m: v for m, v in models.items()} for qid, models in by_q.items()},
    }


def visible_design(data: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    difficulty = data["difficulty"]
    rows = []
    for row in data["stage3"]:
        if row["model_alias"] != model:
            continue
        if row["score_condition"] not in VISIBLE_FIXED_CONDITIONS:
            continue
        if row["roster"] not in {"primary", ""}:
            continue
        feat = difficulty.get((row["question_id"], model))
        if feat is None:
            continue
        item = dict(row)
        item["difficulty"] = float(feat["difficulty"])
        item["wrong"] = int(feat["wrong"])
        item["q1"] = float(data["q1"].get((row["question_id"], model), np.nan))
        rows.append(item)
    return rows


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def bootstrap_improvement(
    y: np.ndarray,
    p_s: np.ndarray,
    p_r: np.ndarray,
    qids: np.ndarray,
    rng: random.Random,
    n_boot: int,
) -> tuple[float, float, float]:
    by_q: dict[str, list[int]] = defaultdict(list)
    for idx, qid in enumerate(qids):
        by_q[str(qid)].append(idx)
    unique = list(dict.fromkeys(str(q) for q in qids))
    boots = []
    for _ in range(n_boot):
        draw = [unique[rng.randrange(len(unique))] for _ in unique]
        idx = np.array([i for qid in draw for i in by_q[qid]], dtype=int)
        boots.append(_nll(y[idx], p_s[idx]) - _nll(y[idx], p_r[idx]))
    lo, hi = _ci(boots)
    return float(np.mean(boots)), lo, hi


def _prepare_h2_arrays(visible: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qids = np.array([row["question_id"] for row in visible])
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    score = np.array([float(row["displayed_confidence"]) for row in visible], dtype=float)
    diff = np.array([float(row["difficulty"]) for row in visible], dtype=float)
    cond = np.array([row["score_condition"] for row in visible])
    enc = OneHotEncoder(sparse_output=False, drop="first")
    gkf = GroupKFold(n_splits=CV_FOLDS)
    splits = list(gkf.split(y, y, groups=qids))
    folds = []
    dmean_global = float(np.mean(diff))
    for train, test in splits:
        enc.fit(cond[train].reshape(-1, 1))
        c_tr = enc.transform(cond[train].reshape(-1, 1))
        c_te = enc.transform(cond[test].reshape(-1, 1))
        dmean = float(np.mean(diff[train]))
        d_tr = (diff[train] - dmean).reshape(-1, 1)
        d_te = (diff[test] - dmean).reshape(-1, 1)
        s_tr = (score[train] - REFERENCE_SCORE).reshape(-1, 1)
        s_te = (score[test] - REFERENCE_SCORE).reshape(-1, 1)
        folds.append(
            {
                "train": train,
                "test": test,
                "X_shared_tr": np.hstack([c_tr, d_tr]),
                "X_shared_te": np.hstack([c_te, d_te]),
                "X_rich_tr": np.hstack([c_tr, d_tr, s_tr * d_tr]),
                "X_rich_te": np.hstack([c_te, d_te, s_te * d_te]),
            }
        )
    unique = list(dict.fromkeys(str(q) for q in qids))
    by_q_lists: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(qids):
        by_q_lists[str(qid)].append(i)
    q_idx = np.array([by_q_lists[qid] for qid in unique], dtype=int)
    return {
        "qids": qids,
        "unique": unique,
        "q_idx": q_idx,
        "folds": folds,
        "y": y,
        "diff": diff,
        "score": score,
        "cond": cond,
        "dmean": dmean_global,
    }


def _bootstrap_improvements(
    y: np.ndarray,
    p_s: np.ndarray,
    p_r: np.ndarray,
    q_idx: np.ndarray,
    n_boot: int,
    rng: random.Random,
) -> np.ndarray:
    n_q = int(q_idx.shape[0])
    rng_np = np.random.default_rng(rng.randrange(2**31))
    draws = rng_np.integers(0, n_q, size=(n_boot, n_q))
    idx = q_idx[draws].reshape(n_boot, -1)
    y_b = y[idx]
    ps_b = np.clip(p_s[idx], EPS, 1.0 - EPS)
    pr_b = np.clip(p_r[idx], EPS, 1.0 - EPS)
    nll_s = -np.mean(y_b * np.log(ps_b) + (1.0 - y_b) * np.log(1.0 - ps_b), axis=1)
    nll_r = -np.mean(y_b * np.log(pr_b) + (1.0 - y_b) * np.log(1.0 - pr_b), axis=1)
    return nll_s - nll_r


def h2_from_y(
    y: np.ndarray, design: Mapping[str, Any], rng: random.Random, n_boot: int
) -> dict[str, Any]:
    p_s = np.zeros(len(y))
    p_r = np.zeros(len(y))
    for fold in design["folds"]:
        train, test = fold["train"], fold["test"]
        fallback = float(np.mean(y[train]))
        p_s[test] = predict_p(fit_logit(fold["X_shared_tr"], y[train]), fold["X_shared_te"], fallback)
        p_r[test] = predict_p(fit_logit(fold["X_rich_tr"], y[train]), fold["X_rich_te"], fallback)
    improvement = _nll(y, p_s) - _nll(y, p_r)
    boots = _bootstrap_improvements(y, p_s, p_r, design["q_idx"], n_boot, rng)
    lo, hi = _ci(boots)
    return {
        "improvement": improvement,
        "bootstrap_hi": hi,
        "bootstrap_lo": lo,
        "invariance_pass": bool(
            math.isfinite(improvement)
            and improvement < MATERIAL_LOGLOSS
            and math.isfinite(hi)
            and hi < MATERIAL_LOGLOSS
        ),
    }


def fit_shared_dgp(visible: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    cond = np.array([row["score_condition"] for row in visible])
    diff = np.array([float(row["difficulty"]) for row in visible])
    score = np.array([float(row["displayed_confidence"]) for row in visible])
    enc = OneHotEncoder(sparse_output=False, drop="first")
    c = enc.fit_transform(cond.reshape(-1, 1))
    d = (diff - float(np.mean(diff))).reshape(-1, 1)
    clf = fit_logit(np.hstack([c, d]), y)
    fallback = float(np.mean(y))
    p = predict_p(clf, np.hstack([c, d]), fallback)
    return {
        "enc": enc,
        "clf": clf,
        "fallback": fallback,
        "dmean": float(np.mean(diff)),
        "p": p,
        "y": y,
        "cond": cond,
        "diff": diff,
        "score": score,
        "qids": np.array([row["question_id"] for row in visible]),
        "wrong": np.array([int(row["wrong"]) for row in visible]),
    }


def _pairwise_reversal(xs: np.ndarray, ys: np.ndarray) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    s1 = np.sign(xs[:, None] - xs[None, :])
    s2 = np.sign(ys[:, None] - ys[None, :])
    iu = np.triu_indices(n, k=1)
    valid = (s1[iu] != 0) & (s2[iu] != 0)
    tot = int(valid.sum())
    if tot == 0:
        return float("nan")
    return float(np.mean(s1[iu][valid] != s2[iu][valid]))


def _prepare_latent_index(qids: np.ndarray, cond: np.ndarray) -> np.ndarray:
    by: dict[str, dict[str, int]] = defaultdict(dict)
    for i, qid in enumerate(qids):
        by[str(qid)][str(cond[i])] = i
    complete = [
        qid
        for qid, item in by.items()
        if all(c in item for c in VISIBLE_FIXED_CONDITIONS)
    ]
    return np.array(
        [[by[qid][c] for c in VISIBLE_FIXED_CONDITIONS] for qid in complete],
        dtype=int,
    )


def _latent_from_idx(eta: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    if len(idx) == 0:
        return {
            "mean_adjacent_spearman": float("nan"),
            "mean_pairwise_reversal": float("nan"),
            "spearman_70_85": float("nan"),
            "spearman_95_99": float("nan"),
        }
    mat = eta[idx]
    rhos = []
    revs = []
    for left, right in ADJACENT_PAIRS:
        jl = VISIBLE_FIXED_CONDITIONS.index(left)
        jr = VISIBLE_FIXED_CONDITIONS.index(right)
        rhos.append(_spearman(mat[:, jl], mat[:, jr]))
        revs.append(_pairwise_reversal(mat[:, jl], mat[:, jr]))
    return {
        "mean_adjacent_spearman": _mean(rhos),
        "mean_pairwise_reversal": _mean(revs),
        "spearman_70_85": rhos[0] if rhos else float("nan"),
        "spearman_95_99": rhos[-1] if rhos else float("nan"),
    }


def _latent_metrics(eta: np.ndarray, qids: np.ndarray, cond: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return _latent_from_idx(eta, _prepare_latent_index(qids, cond))


def _auc_from_eta(eta: np.ndarray, wrong: np.ndarray, cond: np.ndarray) -> dict[str, float]:
    out = {}
    for condition in VISIBLE_FIXED_CONDITIONS:
        mask = cond == condition
        out[condition] = _safe_auc(wrong[mask], eta[mask])
    return out


def run_positive_control(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng_master = np.random.default_rng(SIM_SEED)
    py_rng = random.Random(SIM_SEED)
    results: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    design_notes: dict[str, Any] = {
        "n_sim": N_SIM,
        "n_boot_power": N_BOOT_POWER,
        "sim_seed": SIM_SEED,
        "primary_anchor_gamma": PRIMARY_ANCHOR_GAMMA,
        "gpt_008_gamma": GPT_008_GAMMA,
        "claude_008_gamma": CLAUDE_008_GAMMA,
        "between_model_gamma": BETWEEN_MODEL_GAMMA,
        "anchor_source": DEVELOPMENT_ANCHOR_SOURCE,
        "equivalence_boundary": MATERIAL_LOGLOSS,
        "detection_rule": (
            "Fail invariance if held-out improvement >= 0.01 or question-bootstrap "
            "95% upper bound >= 0.01, matching Task 009."
        ),
    }
    for model in GPT_CLAUDE:
        visible = visible_design(data, model)
        design = _prepare_h2_arrays(visible)
        dgp = fit_shared_dgp(visible)
        lat_idx = _prepare_latent_index(dgp["qids"], dgp["cond"])
        base_eta = np.log(np.clip(dgp["p"], EPS, 1 - EPS) / np.clip(1 - dgp["p"], EPS, 1 - EPS))
        d_c = dgp["diff"] - dgp["dmean"]
        s_c = dgp["score"] - REFERENCE_SCORE
        var_shared_item = (
            float(np.var(dgp["clf"].coef_[0][-1] * d_c.ravel()))
            if dgp["clf"] is not None
            else float(np.var(d_c))
        )
        for family, grid in (
            [
                ("score_x_difficulty", [(m, m * PRIMARY_ANCHOR_GAMMA) for m in INJECTION_MULTIPLIERS]),
                (
                    "generic_item_noise",
                    [
                        (
                            rho,
                            math.sqrt(
                                max(var_shared_item, 1e-6)
                                * (1.0 - min(max(rho, 0.05), 0.999))
                                / min(max(rho, 0.05), 0.999)
                            ),
                        )
                        for rho in FAMILY2_TARGET_RHO
                    ],
                ),
            ]
        ):
            for mag_label, mag in grid:
                detect = 0
                detect_point = 0
                imps = []
                his = []
                spearmans = []
                reversals = []
                auc_shift = []
                print(
                    f"  {MODEL_LABELS[model]} {family} mag={mag_label} n_sim={N_SIM}",
                    flush=True,
                )
                t_cell = time.time()
                for sim in range(N_SIM):
                    if family == "score_x_difficulty":
                        extra = mag * s_c * d_c.ravel()
                        sigma_used = 0.0
                    else:
                        extra = mag * rng_master.normal(size=len(base_eta))
                        sigma_used = float(mag)
                    eta = base_eta + extra
                    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
                    y = (rng_master.random(len(p)) < p).astype(int)
                    fit = h2_from_y(y, design, py_rng, N_BOOT_POWER)
                    lat = _latent_from_idx(eta, lat_idx)
                    aucs = _auc_from_eta(eta, dgp["wrong"], dgp["cond"])
                    auc_shift.append(max(aucs.values()) - min(aucs.values()) if aucs else float("nan"))
                    imps.append(fit["improvement"])
                    his.append(fit["bootstrap_hi"])
                    spearmans.append(lat["mean_adjacent_spearman"])
                    reversals.append(lat["mean_pairwise_reversal"])
                    detected = int(
                        (fit["improvement"] >= MATERIAL_LOGLOSS)
                        or (
                            math.isfinite(fit["bootstrap_hi"])
                            and fit["bootstrap_hi"] >= MATERIAL_LOGLOSS
                        )
                    )
                    detect += detected
                    if fit["improvement"] >= MATERIAL_LOGLOSS:
                        detect_point += 1
                    results.append(
                        {
                            "model_alias": model,
                            "model_label": MODEL_LABELS[model],
                            "family": family,
                            "magnitude_label": mag_label,
                            "magnitude": mag if family == "score_x_difficulty" else sigma_used,
                            "sim": sim,
                            "heldout_improvement": fit["improvement"],
                            "bootstrap_hi": fit["bootstrap_hi"],
                            "detected": detected,
                            "latent_adjacent_spearman": lat["mean_adjacent_spearman"],
                            "latent_pairwise_reversal": lat["mean_pairwise_reversal"],
                        }
                    )
                power = detect / N_SIM
                print(
                    f"    power={power:.3f} mean_imp={_mean(imps):.4f} "
                    f"rho={_mean(spearmans):.3f} [{time.time()-t_cell:.1f}s]",
                    flush=True,
                )
                power_rows.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "family": family,
                        "magnitude_label": mag_label,
                        "magnitude": mag if family == "score_x_difficulty" else float(mag),
                        "target_latent_rho": mag_label if family == "generic_item_noise" else float("nan"),
                        "n_sim": N_SIM,
                        "power_full_009_rule": power,
                        "power_point_estimate_only": detect_point / N_SIM,
                        "mean_improvement": _mean(imps),
                        "p05_improvement": float(np.nanpercentile(imps, 5)),
                        "p95_improvement": float(np.nanpercentile(imps, 95)),
                        "mean_bootstrap_hi": _mean(his),
                        "mean_latent_adjacent_spearman": _mean(spearmans),
                        "mean_pairwise_reversal": _mean(reversals),
                        "mean_auc_span": _mean(auc_shift),
                        "false_positive_if_zero": power
                        if (family == "score_x_difficulty" and mag == 0)
                        or (family == "generic_item_noise" and mag_label == 0.99)
                        else "",
                    }
                )
    return results, power_rows, design_notes


def detection_floor(power_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    floors: dict[str, Any] = {}
    for model in GPT_CLAUDE:
        floors[model] = {}
        for family in ("score_x_difficulty", "generic_item_noise"):
            subset = [
                row
                for row in power_rows
                if row["model_alias"] == model and row["family"] == family
            ]
            subset = sorted(subset, key=lambda r: abs(float(r["magnitude"])))
            chosen = None
            for row in subset:
                if float(row["power_full_009_rule"]) >= TIGHT_MIN_POWER:
                    chosen = row
                    break
            floors[model][family] = chosen
    return floors


def choose_template(floors: Mapping[str, Any], power_rows: Sequence[Mapping[str, Any]]) -> str:
    """Pick TIGHT / MODERATE / WEAK from predeclared floors. Conservative across models/families."""

    def classify(row: Mapping[str, Any] | None) -> str:
        if row is None:
            return "WEAK"
        rho = float(row.get("mean_latent_adjacent_spearman") or np.nan)
        rev = float(row.get("mean_pairwise_reversal") or np.nan)
        # Small disturbance detected with 80% power → tight
        if math.isfinite(rho) and rho >= SMALL_RANK_CORR and (not math.isfinite(rev) or rev <= SMALL_REVERSAL + 1e-9):
            return "TIGHT"
        if math.isfinite(rho) and rho >= MODERATE_RANK_CORR and (not math.isfinite(rev) or rev <= MODERATE_REVERSAL + 1e-9):
            return "MODERATE"
        return "WEAK"

    labels = []
    for model in GPT_CLAUDE:
        for family in ("score_x_difficulty", "generic_item_noise"):
            labels.append(classify(floors.get(model, {}).get(family)))
    if labels.count("WEAK") >= 1 and "TIGHT" not in labels and "MODERATE" not in labels:
        return "WEAK"
    # Conservative: weakest label that appears
    if "WEAK" in labels:
        # If some families are tight/moderate but another is weak, use WEAK only if
        # 80% power is never reached except at large disruption.
        large_ok = []
        for model in GPT_CLAUDE:
            for family in ("score_x_difficulty", "generic_item_noise"):
                subset = [
                    row
                    for row in power_rows
                    if row["model_alias"] == model and row["family"] == family
                ]
                reached = False
                for row in subset:
                    rho = float(row.get("mean_latent_adjacent_spearman") or np.nan)
                    if float(row["power_full_009_rule"]) >= TIGHT_MIN_POWER:
                        reached = True
                        if math.isfinite(rho) and rho > LARGE_RANK_CORR:
                            # detected before becoming large
                            large_ok.append("better_than_weak")
                        else:
                            large_ok.append("weak")
                if not reached:
                    large_ok.append("weak")
        if "weak" in large_ok and "better_than_weak" not in large_ok:
            return "WEAK"
        if "weak" in large_ok:
            return "MODERATE"
    if "MODERATE" in labels and "TIGHT" in labels:
        return "MODERATE"
    if all(x == "TIGHT" for x in labels):
        return "TIGHT"
    if "TIGHT" in labels and "MODERATE" not in labels and "WEAK" not in labels:
        return "TIGHT"
    return "MODERATE"


def observed_vs_predicted_auc(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SIM_SEED + 1)
    py = random.Random(SIM_SEED + 1)
    rows = []
    for model in GPT_CLAUDE:
        visible = visible_design(data, model)
        dgp = fit_shared_dgp(visible)
        q_unique = list(dict.fromkeys(str(q) for q in dgp["qids"]))
        by_q: dict[str, list[int]] = defaultdict(list)
        for i, qid in enumerate(dgp["qids"]):
            by_q[str(qid)].append(i)
        for condition, score in zip(VISIBLE_FIXED_CONDITIONS, VISIBLE_FIXED):
            mask = dgp["cond"] == condition
            y_obs = dgp["y"][mask]
            wrong = dgp["wrong"][mask]
            p = dgp["p"][mask]
            coverage = float(np.mean(y_obs))
            obs_auc = _safe_auc(wrong, y_obs)
            # Monte Carlo expected binary AUROC under shared p
            sims = []
            for _ in range(400):
                y_s = (rng.random(len(p)) < p).astype(int)
                sims.append(_safe_auc(wrong, y_s))
            pred = _mean(sims)
            lo, hi = _ci(sims)
            # question bootstrap residual of observed
            boots_obs = []
            boots_pred = []
            qids_c = [str(dgp["qids"][i]) for i in np.where(mask)[0]]
            y_list = y_obs
            w_list = wrong
            p_list = p
            n = len(qids_c)
            for _ in range(min(1000, ROUTING_BOOT)):
                draw = rng.integers(0, n, size=n)
                boots_obs.append(_safe_auc(w_list[draw], y_list[draw]))
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "score": score,
                    "score_condition": condition,
                    "observed_coverage": coverage,
                    "observed_binary_auroc": obs_auc,
                    "predicted_shared_ranking_auroc": pred,
                    "predicted_lo": lo,
                    "predicted_hi": hi,
                    "residual_obs_minus_pred": obs_auc - pred,
                    "n": int(mask.sum()),
                    "n_wrong": int(wrong.sum()),
                }
            )
    return rows


def latent_rank_stability(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rng = np.random.default_rng(SIM_SEED + 2)
    out_corr: list[dict[str, Any]] = []
    out_pp: list[dict[str, Any]] = []
    labels = []
    for model in GPT_CLAUDE:
        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in list(data["stage3"]) + list(data["repeats"]):
            if row["model_alias"] != model:
                continue
            if row["score_condition"] not in VISIBLE_FIXED_CONDITIONS:
                continue
            cells[(row["question_id"], row["score_condition"])].append(int(row["verify"]))
        qids = sorted(
            {
                qid
                for (qid, cond), vals in cells.items()
                if cond == "displayed_0.70" and len(vals) >= 3
            }
        )
        qids = [
            qid
            for qid in qids
            if all(len(cells.get((qid, c), [])) >= 3 for c in VISIBLE_FIXED_CONDITIONS)
        ]
        # k[i,c] successes in 3 trials
        k = np.zeros((len(qids), 5), dtype=float)
        n = np.full_like(k, 3.0)
        for i, qid in enumerate(qids):
            for j, cond in enumerate(VISIBLE_FIXED_CONDITIONS):
                vals = cells[(qid, cond)][:3]
                k[i, j] = float(sum(vals))
        # MAP hierarchical: logit p = a_c + u_i + e_{i,c}
        a = np.log((k.mean(axis=0) + 0.5) / (3 - k.mean(axis=0) + 0.5))
        u = np.zeros(len(qids))
        e = np.zeros_like(k)
        tau2 = 1.0
        sig2 = 1.0
        for _ in range(80):
            eta = a.reshape(1, -1) + u.reshape(-1, 1) + e
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
            # gradient step on u, e, a
            grad_eta = k - n * p
            a += 0.15 * grad_eta.mean(axis=0)
            u += 0.15 * (grad_eta.sum(axis=1) - u / max(tau2, 1e-6))
            e += 0.15 * (grad_eta - e / max(sig2, 1e-6))
            u -= u.mean()
            e -= e.mean(axis=0, keepdims=True)
            tau2 = float(np.mean(u**2) + 1e-4)
            sig2 = float(np.mean(e**2) + 1e-4)
        eta = a.reshape(1, -1) + u.reshape(-1, 1) + e
        p_lat = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
        raw = k / 3.0
        for j, (left, right) in enumerate(ADJACENT_PAIRS):
            jl = VISIBLE_FIXED_CONDITIONS.index(left)
            jr = VISIBLE_FIXED_CONDITIONS.index(right)
            raw_rho = _spearman(raw[:, jl], raw[:, jr])
            lat_rho = _spearman(p_lat[:, jl], p_lat[:, jr])
            lat_pearson = float(np.corrcoef(eta[:, jl], eta[:, jr])[0, 1])
            # posterior predictive raw spearman
            pp = []
            for _ in range(LATENT_BOOT):
                y1 = rng.binomial(3, np.clip(p_lat[:, jl], 1e-6, 1 - 1e-6)) / 3.0
                y2 = rng.binomial(3, np.clip(p_lat[:, jr], 1e-6, 1 - 1e-6)) / 3.0
                pp.append(_spearman(y1, y2))
            lo, hi = _ci(pp)
            out_corr.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "pair": f"{left}->{right}",
                    "n_questions": len(qids),
                    "raw_spearman": raw_rho,
                    "latent_spearman": lat_rho,
                    "latent_logit_pearson": lat_pearson,
                    "pp_raw_spearman_mean": _mean(pp),
                    "pp_raw_spearman_lo": lo,
                    "pp_raw_spearman_hi": hi,
                    "tau2": tau2,
                    "sig2": sig2,
                }
            )
            out_pp.append(
                {
                    "model_alias": model,
                    "pair": f"{left}->{right}",
                    "observed_raw_spearman": raw_rho,
                    "expected_raw_spearman_if_latent": _mean(pp),
                    "attenuation": raw_rho - _mean(pp) if math.isfinite(raw_rho) else float("nan"),
                }
            )
        mean_lat = _mean([row["latent_spearman"] for row in out_corr if row["model_alias"] == model])
        mean_raw = _mean([row["raw_spearman"] for row in out_corr if row["model_alias"] == model])
        if not qids:
            labels.append("INSUFFICIENT_REPEAT_INFORMATION")
        elif mean_lat >= 0.85:
            labels.append("LATENT_STABILITY_STRONG")
        elif mean_lat >= 0.70:
            labels.append("LATENT_STABILITY_MODERATE")
        else:
            labels.append("LATENT_STABILITY_WEAK_OR_MIXED")
        _ = mean_raw
    if "INSUFFICIENT_REPEAT_INFORMATION" in labels:
        claim = "INSUFFICIENT_REPEAT_INFORMATION"
    elif all(x == "LATENT_STABILITY_STRONG" for x in labels):
        claim = "LATENT_STABILITY_STRONG"
    elif "LATENT_STABILITY_WEAK_OR_MIXED" in labels:
        claim = "LATENT_STABILITY_WEAK_OR_MIXED"
    else:
        claim = "LATENT_STABILITY_MODERATE"
    return out_corr, out_pp, claim


def _cv_with_extra(
    visible: Sequence[Mapping[str, Any]], extra: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    qids = np.array([row["question_id"] for row in visible])
    y = np.array([int(row["verify"]) for row in visible], dtype=int)
    score = np.array([float(row["displayed_confidence"]) for row in visible])
    diff = np.array([float(row["difficulty"]) for row in visible])
    cond = np.array([row["score_condition"] for row in visible])
    enc = OneHotEncoder(sparse_output=False, drop="first")
    gkf = GroupKFold(n_splits=CV_FOLDS)
    p_shared = np.zeros(len(y))
    p_rich = np.zeros(len(y))
    xextra = extra.reshape(-1, 1)
    for train, test in gkf.split(y, y, groups=qids):
        fallback = float(np.mean(y[train]))
        enc.fit(cond[train].reshape(-1, 1))
        c_tr = enc.transform(cond[train].reshape(-1, 1))
        c_te = enc.transform(cond[test].reshape(-1, 1))
        dmean = float(np.mean(diff[train]))
        d_tr = (diff[train] - dmean).reshape(-1, 1)
        d_te = (diff[test] - dmean).reshape(-1, 1)
        p_shared[test] = predict_p(
            fit_logit(np.hstack([c_tr, d_tr]), y[train]),
            np.hstack([c_te, d_te]),
            fallback,
        )
        p_rich[test] = predict_p(
            fit_logit(np.hstack([c_tr, d_tr, xextra[train]]), y[train]),
            np.hstack([c_te, d_te, xextra[test]]),
            fallback,
        )
    return (
        _safe_log_loss(y, p_shared),
        _safe_log_loss(y, p_rich),
        p_shared,
        p_rich,
        qids,
    )


def mismatch_robustness(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(SIM_SEED + 3)
    rows = []
    for model in GPT_CLAUDE:
        visible = visible_design(data, model)
        q1 = np.clip(np.array([float(row["q1"]) for row in visible]), EPS, 1 - EPS)
        disp = np.clip(np.array([float(row["displayed_confidence"]) for row in visible]), EPS, 1 - EPS)
        diff = np.array([float(row["difficulty"]) for row in visible])
        extras = {
            "signed_mismatch": disp - q1,
            "abs_mismatch": np.abs(disp - q1),
            "logit_mismatch": np.abs(np.log(disp / (1 - disp)) - np.log(q1 / (1 - q1))),
            "score_x_difficulty": (disp - REFERENCE_SCORE) * (diff - np.mean(diff)),
        }
        y = np.array([int(row["verify"]) for row in visible], dtype=int)
        for name, extra in extras.items():
            ll_s, ll_r, p_s, p_r, qids = _cv_with_extra(visible, extra)
            improvement = ll_s - ll_r
            _, lo, hi = bootstrap_improvement(y, p_s, p_r, qids, rng, N_BOOT_POWER)
            # coefficient from in-sample fit for direction
            enc = OneHotEncoder(sparse_output=False, drop="first")
            cond = np.array([row["score_condition"] for row in visible])
            c = enc.fit_transform(cond.reshape(-1, 1))
            d = (diff - np.mean(diff)).reshape(-1, 1)
            clf = fit_logit(np.hstack([c, d, extra.reshape(-1, 1)]), y)
            coef = float(clf.coef_[0][-1]) if clf is not None else float("nan")
            rows.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "predictor": name,
                    "status": "POST_HOC_ROBUSTNESS — NOT PREREGISTERED TASK-009 CONFIRMATION",
                    "heldout_improvement": improvement,
                    "bootstrap_lo": lo,
                    "bootstrap_hi": hi,
                    "material_0_01": bool(improvement >= MATERIAL_LOGLOSS and hi >= MATERIAL_LOGLOSS),
                    "coef_direction": coef,
                    "n_rows": len(visible),
                }
            )
    return rows


def matched_budget_routing(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    rng = np.random.default_rng(SIM_SEED + 4)
    rows: list[dict[str, Any]] = []
    for model in GPT_CLAUDE:
        hidden = [
            row
            for row in data["stage3"]
            if row["model_alias"] == model and row["score_condition"] == "hidden"
        ]
        qids = [row["question_id"] for row in hidden]
        wrong = np.array([int(not row["stage1_correct"]) for row in hidden], dtype=int)
        hidden_s = np.array([int(row["verify"]) for row in hidden], dtype=float)
        q1 = np.array([float(data["q1"][(qid, model)]) for qid in qids])
        correct = 1 - wrong
        cal = crossfit_isotonic(q1, correct, np.array(qids), seed=BOOTSTRAP_SEED)
        cal = np.where(np.isfinite(cal), cal, q1)
        other = CLAUDE if model == GPT else GPT
        other_wrong = np.array(
            [1 - int(data["correct"][qid].get(other, 0)) for qid in qids], dtype=float
        )
        combo = np.clip(0.5 * hidden_s + 0.5 * (1.0 - q1), 0, 1)
        routers = {
            "random": None,
            "raw_q1": 1.0 - q1,
            "calibrated_q1": 1.0 - cal,
            "hidden": hidden_s,
            "q1_plus_hidden": combo,
            "hindsight_oracle": wrong.astype(float),
            "loo_other_primary_wrong": other_wrong,
        }
        n = len(qids)
        for budget in MATCHED_BUDGETS:
            n_verify = budget * n
            for name, score in routers.items():
                def catch_from(score_vec: np.ndarray | None, idx: np.ndarray | None = None) -> float:
                    w = wrong if idx is None else wrong[idx]
                    if name == "random":
                        order = np.arange(len(w))
                        rng2 = np.random.default_rng(int(budget * 1000) + (0 if idx is None else int(idx[0])))
                        order = rng2.permutation(len(w))
                        weights = np.zeros(len(w))
                        take = int(round(n_verify if idx is None else budget * len(w)))
                        weights[order[:take]] = 1.0
                        return error_catch_from_weights(weights, w)
                    s = score if idx is None else score[idx]
                    weights = fractional_verify_weights(-s, n_verify if idx is None else budget * len(w))
                    return error_catch_from_weights(weights, w)

                point = catch_from(score)
                boots = []
                for _ in range(ROUTING_BOOT):
                    draw = rng.integers(0, n, size=n)
                    if name == "random":
                        order = rng.permutation(n)
                        weights = np.zeros(n)
                        take = int(round(n_verify))
                        weights[order[:take]] = 1.0
                        boots.append(error_catch_from_weights(weights, wrong[draw]))
                    else:
                        weights = fractional_verify_weights(-score[draw], n_verify)
                        boots.append(error_catch_from_weights(weights, wrong[draw]))
                lo, hi = _ci(boots)
                s_use = np.zeros(n) if score is None else score
                weights = (
                    fractional_verify_weights(-s_use, n_verify)
                    if name != "random"
                    else None
                )
                if name == "random":
                    precision = float("nan")
                    residual = float("nan")
                else:
                    verified = weights
                    n_ver = float(verified.sum())
                    tp = float((verified * wrong).sum())
                    precision = tp / n_ver if n_ver else float("nan")
                    residual = float(((1.0 - verified) * wrong).sum()) / n
                rows.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "budget": budget,
                        "router": name,
                        "catch_frac": point,
                        "catch_lo": lo,
                        "catch_hi": hi,
                        "precision_among_verified": precision,
                        "residual_error_rate_unverified": residual,
                        "n": n,
                        "n_wrong": int(wrong.sum()),
                        "evaluation_only": name == "loo_other_primary_wrong",
                    }
                )
    # classify using frozen scale
    realistic = {"raw_q1", "calibrated_q1", "hidden", "q1_plus_hidden"}
    gains = []
    for model in GPT_CLAUDE:
        for budget in ROUTING_BUDGETS:
            subset = [
                row
                for row in rows
                if row["model_alias"] == model and row["budget"] == budget
            ]
            base = max(
                (row["catch_frac"] for row in subset if row["router"] in realistic),
                default=float("nan"),
            )
            hidden = next(row["catch_frac"] for row in subset if row["router"] == "hidden")
            q1h = next(row["catch_frac"] for row in subset if row["router"] == "q1_plus_hidden")
            raw = next(row["catch_frac"] for row in subset if row["router"] == "raw_q1")
            cal = next(row["catch_frac"] for row in subset if row["router"] == "calibrated_q1")
            strongest_real = max(hidden, q1h, raw, cal)
            # gain of best among hidden/q1+hidden over best q1-only? Task: gain over strongest realistic baseline.
            # "material routing gain" means some router beats the others. Compare q1+hidden vs best of q1/cal/hidden?
            # Use best of {hidden, q1+hidden} minus best of {raw_q1, calibrated_q1} as the incremental judgment gain,
            # AND best realistic minus random as overall.
            random_c = next(row["catch_frac"] for row in subset if row["router"] == "random")
            gain_vs_random = strongest_real - random_c
            q1_best = max(raw, cal)
            judgment_gain = strongest_real - q1_best
            gains.append((model, budget, judgment_gain, gain_vs_random, strongest_real, q1_best))
    # material: >=10pp over strongest realistic baseline replicated both models at 20-40%
    # interpret as: the *winning* realistic router vs the *next* class. The spec says
    # "error-catch gain over the strongest realistic baseline" which is tautological if the winner IS the baseline.
    # Read as: matched-budget gain of hidden/q1+hidden relative to q1, which is the practical question.
    both_material = True
    both_modest = True
    for model in GPT_CLAUDE:
        model_gains = [g[2] for g in gains if g[0] == model]
        if not model_gains or max(model_gains) < ROUTING_MATERIAL_PP:
            both_material = False
        if not model_gains or max(model_gains) < ROUTING_MODEST_PP:
            both_modest = False
    one_material = False
    for model in GPT_CLAUDE:
        model_gains = [g[2] for g in gains if g[0] == model]
        if model_gains and max(model_gains) >= ROUTING_MATERIAL_PP:
            one_material = True
    if both_material:
        label = "MATERIAL_ROUTING_GAIN"
    elif one_material or (not both_modest):
        # modest if 3-10pp or inconsistent
        label = "MODEST_ROUTING_GAIN"
    else:
        maxg = max((g[2] for g in gains), default=0.0)
        if maxg >= ROUTING_MODEST_PP:
            label = "MODEST_ROUTING_GAIN"
        else:
            label = "NO_MATERIAL_ROUTING_GAIN"
    return rows, label


def plot_all(
    power_rows: Sequence[Mapping[str, Any]],
    auc_rows: Sequence[Mapping[str, Any]],
    latent_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
) -> None:
    plt = _pyplot()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, family, title in zip(
        axes,
        ("score_x_difficulty", "generic_item_noise"),
        ("Family 1: score × difficulty", "Family 2: generic item noise"),
    ):
        for model in GPT_CLAUDE:
            subset = [r for r in power_rows if r["model_alias"] == model and r["family"] == family]
            if family == "score_x_difficulty":
                xs = [float(r["magnitude"]) for r in subset]
            else:
                xs = [float(r["target_latent_rho"]) for r in subset]
            ys = [100.0 * float(r["power_full_009_rule"]) for r in subset]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=MODEL_LABELS[model])
        ax.axhline(80, color="0.5", ls="--", lw=1)
        ax.set_title(title)
        ax.set_ylabel("Power to fail 009 invariance (%)")
        ax.legend()
        if family == "score_x_difficulty":
            ax.set_xlabel("Injected γ (009 coding)")
        else:
            ax.set_xlabel("Target latent rank correlation")
            ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "positive_control_power.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for model in GPT_CLAUDE:
        subset = [r for r in auc_rows if r["model_alias"] == model]
        xs = [float(r["score"]) for r in subset]
        ax.plot(xs, [float(r["observed_binary_auroc"]) for r in subset], marker="o", label=f"{MODEL_LABELS[model]} observed")
        ax.plot(
            xs,
            [float(r["predicted_shared_ranking_auroc"]) for r in subset],
            marker="s",
            ls="--",
            label=f"{MODEL_LABELS[model]} shared-model predicted",
        )
    ax.set_xlabel("Displayed confidence")
    ax.set_ylabel("Binary-action AUROC")
    ax.set_title("Observed vs shared-ranking predicted AUROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "observed_vs_shared_predicted_auc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labels = [f"{a.split('_')[-1]}→{b.split('_')[-1]}" for a, b in ADJACENT_PAIRS]
    x = np.arange(len(labels))
    width = 0.18
    for i, model in enumerate(GPT_CLAUDE):
        raw = []
        lat = []
        for pair in ADJACENT_PAIRS:
            key = f"{pair[0]}->{pair[1]}"
            row = next(r for r in latent_rows if r["model_alias"] == model and r["pair"] == key)
            raw.append(float(row["raw_spearman"]))
            lat.append(float(row["latent_spearman"]))
        ax.bar(x + (i * 2 - 1.5) * width, raw, width, label=f"{MODEL_LABELS[model]} raw")
        ax.bar(x + (i * 2 - 0.5) * width, lat, width, label=f"{MODEL_LABELS[model]} latent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rank correlation")
    ax.set_title("Raw 3-draw Spearman vs hierarchical latent")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latent_rank_stability.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    show = ["random", "raw_q1", "calibrated_q1", "hidden", "q1_plus_hidden", "hindsight_oracle"]
    for ax, model in zip(axes, GPT_CLAUDE):
        for name in show:
            subset = [r for r in routing_rows if r["model_alias"] == model and r["router"] == name]
            xs = [float(r["budget"]) for r in subset]
            ys = [100.0 * float(r["catch_frac"]) for r in subset]
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Verification budget")
        ax.set_ylabel("% of errors caught")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matched_budget_routing.png", dpi=160)
    plt.close(fig)
