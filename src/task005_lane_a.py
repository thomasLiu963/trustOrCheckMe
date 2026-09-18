"""Lane A: zero-cost Study-1 / V2 diagnostics. Read-only on historical files."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .bootstrap import paired_bootstrap
from .calibration import fit_isotonic
from .config import PROJECT_ROOT
from .study1_analysis import (
    EXTREME,
    MODEL_LABELS,
    NEAR_THRESHOLD,
    VERIFY,
    _is_verify,
    _pyplot,
    _write_csv,
    mean_or_nan,
    paired_contrast,
    rate_with_ci,
)
from .study1_sample import hash_id_list, load_frozen_ids
from .study1_schemas import (
    STUDY1_GRID_L10,
    STUDY1_GRID_L20,
    STUDY1_MODEL_ALIASES,
    load_study1_config,
)
from .task005_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LANE_A_DIR,
    PRIMARY_FAMILY,
    SATURATION_LABEL,
    STUDY1_SQLITE,
    V2_SQLITE,
    file_fingerprint,
    load_json_records,
    sha256_file,
)

D1_MODELS = STUDY1_MODEL_ALIASES
V2_MODELS_ROUTING = STUDY1_MODEL_ALIASES  # D1 models only; Gemini/Grok not added
OWNERS = ("ai_system", "human")
ERROR_COSTS = (2.0, 5.0, 10.0, 20.0)
TASK001_ID_HASH = "badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pp(value: float) -> float:
    return 100.0 * value


def load_study1_primary_rows() -> list[dict[str, Any]]:
    records = load_json_records(STUDY1_SQLITE, stage="verification")
    rows: list[dict[str, Any]] = []
    for record in records:
        stake = record.get("_stake") or {}
        repeat_index = int(record.get("repeat_index", stake.get("repeat_index", 0)))
        phase = record.get("phase")
        if phase == "repeats" or repeat_index != 0:
            continue
        action = record.get("parsed_action")
        if isinstance(action, dict):
            action = action.get("value") or action.get("action")
        rows.append(
            {
                "question_id": record.get("question_id") or record.get("example_id"),
                "model_alias": record["model_alias"],
                "model_endpoint": record.get("model_endpoint")
                or record.get("returned_model_id"),
                "L": float(record.get("L", stake.get("L"))),
                "C": float(record.get("C", stake.get("C", 1.0))),
                "display_condition": record.get("display_condition")
                or stake.get("display_condition"),
                "reported_confidence": record.get("reported_confidence"),
                "displayed_confidence": record.get("displayed_confidence")
                if record.get("displayed_confidence") is not None
                else stake.get("displayed_confidence"),
                "frozen_answer": record.get("frozen_answer"),
                "stage1_correct": bool(record.get("stage1_correct")),
                "parsed_action": str(action),
                "repeat_index": 0,
                "phase": "primary",
                "request_key": record.get("request_key"),
            }
        )
    return rows


def load_study1_repeat_rows() -> list[dict[str, Any]]:
    records = load_json_records(STUDY1_SQLITE, stage="verification")
    config = load_study1_config()
    _selected, repeats, _sh, _rh = load_frozen_ids(config)
    repeat_set = set(repeats)
    rows: list[dict[str, Any]] = []
    for record in records:
        stake = record.get("_stake") or {}
        qid = record.get("question_id") or record.get("example_id")
        if qid not in repeat_set:
            continue
        action = record.get("parsed_action")
        if isinstance(action, dict):
            action = action.get("value") or action.get("action")
        rows.append(
            {
                "question_id": qid,
                "model_alias": record["model_alias"],
                "L": float(record.get("L", stake.get("L"))),
                "display_condition": record.get("display_condition")
                or stake.get("display_condition"),
                "displayed_confidence": record.get("displayed_confidence")
                if record.get("displayed_confidence") is not None
                else stake.get("displayed_confidence"),
                "stage1_correct": bool(record.get("stage1_correct")),
                "parsed_action": str(action),
                "repeat_index": int(
                    record.get("repeat_index", stake.get("repeat_index", 0))
                ),
            }
        )
    return rows


def load_v2_answers() -> dict[tuple[str, str], dict[str, Any]]:
    records = load_json_records(V2_SQLITE, stage="answer")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["example_id"]), str(record["model_id"]))
        out[key] = record
    return out


def load_v2_confidence() -> dict[tuple[str, str], dict[str, Any]]:
    records = load_json_records(V2_SQLITE, stage="confidence")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["example_id"]), str(record.get("model_id") or record.get("model_alias")))
        out[key] = record
    return out


def load_v2_primary_verification() -> list[dict[str, Any]]:
    answers = load_v2_answers()
    records = load_json_records(V2_SQLITE, stage="verification")
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("prompt_family") != PRIMARY_FAMILY:
            continue
        if record.get("prompt_version") not in {PRIMARY_FAMILY, None}:
            # historical verification prompt_version is the family name
            if record.get("prompt_version") != PRIMARY_FAMILY:
                continue
        model = str(record.get("model_id") or record.get("model_alias"))
        qid = str(record["example_id"])
        answer = answers.get((qid, model))
        if answer is None:
            continue
        rows.append(
            {
                "question_id": qid,
                "model_alias": model,
                "decision_owner": record.get("decision_owner"),
                "confidence_visibility": record.get("confidence_visibility"),
                "L": float(record["error_cost"]),
                "C": float(record.get("verification_cost", 1.0)),
                "parsed_action": record.get("action"),
                "probability_correct": float(record["probability_correct"]),
                "frozen_answer": record.get("frozen_answer_label"),
                "stage1_correct": bool(answer["is_correct"]),
            }
        )
    return rows


def subset_study1(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    L: float | None = None,
    condition: str | None = None,
    displayed: float | None = None,
    manipulated_only: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if model is not None and row["model_alias"] != model:
            continue
        if L is not None and float(row["L"]) != float(L):
            continue
        if condition is not None and row["display_condition"] != condition:
            continue
        if manipulated_only and not str(row["display_condition"]).startswith("manipulated"):
            continue
        if displayed is not None:
            value = row["displayed_confidence"]
            if value is None or abs(float(value) - float(displayed)) > 1e-12:
                continue
        selected.append(dict(row))
    return selected


def validate_lane_a(
    primary: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
    v2_before: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    config = load_study1_config()
    selected, repeat_ids, selected_hash, repeat_hash = load_frozen_ids(config)
    qids = sorted({row["question_id"] for row in primary})
    if len(qids) != 100:
        issues.append(f"primary unique questions {len(qids)} != 100")
    if hash_id_list(selected) != TASK001_ID_HASH:
        issues.append("frozen Study-1 ID hash mismatch")
    if selected_hash != TASK001_ID_HASH:
        issues.append("load_frozen_ids hash mismatch")
    if set(qids) != set(selected):
        issues.append("primary question IDs do not match frozen Study-1 IDs")
    by_model = defaultdict(int)
    by_L = defaultdict(int)
    by_cond = defaultdict(int)
    by_cell = defaultdict(int)
    for row in primary:
        by_model[row["model_alias"]] += 1
        by_L[float(row["L"])] += 1
        by_cond[row["display_condition"]] += 1
        by_cell[
            (row["question_id"], row["model_alias"], float(row["L"]), row["display_condition"])
        ] += 1
        if row["parsed_action"] not in {VERIFY, "USE_UNVERIFIED"}:
            issues.append(f"unexpected action {row['parsed_action']}")
        if row["stage1_correct"] is None:
            issues.append("missing stage1_correct")
        if row["frozen_answer"] in {None, ""}:
            issues.append("missing frozen_answer")
    if dict(by_model) != {"openai_gpt56_sol": 1400, "anthropic_sonnet5": 1400}:
        issues.append(f"primary model counts {dict(by_model)}")
    if dict(by_L) != {10.0: 1400, 20.0: 1400}:
        issues.append(f"primary L counts {dict(by_L)}")
    for condition, count in by_cond.items():
        if count != 400:
            issues.append(f"{condition} count {count} != 400")
    if any(count != 1 for count in by_cell.values()):
        issues.append("duplicate primary cells")
    grids = {
        10.0: list(STUDY1_GRID_L10),
        20.0: list(STUDY1_GRID_L20),
    }
    for model in D1_MODELS:
        for L, expected in grids.items():
            observed = sorted(
                {
                    float(row["displayed_confidence"])
                    for row in primary
                    if row["model_alias"] == model
                    and float(row["L"]) == L
                    and str(row["display_condition"]).startswith("manipulated")
                }
            )
            if observed != expected:
                issues.append(f"displayed grid mismatch {model} L={L}: {observed}")
    # repeats: 20 questions, 3 observations per cell
    repeat_q = {row["question_id"] for row in repeats}
    if repeat_q != set(repeat_ids):
        issues.append("repeat question IDs mismatch")
    cells = defaultdict(set)
    for row in repeats:
        cells[
            (row["question_id"], row["model_alias"], float(row["L"]), row["display_condition"])
        ].add(int(row["repeat_index"]))
    if len(cells) != 560:
        issues.append(f"repeat cells {len(cells)} != 560")
    if any(indexes != {0, 1, 2} for indexes in cells.values()):
        issues.append("repeat cells missing a generation")
    v2_after = file_fingerprint(V2_SQLITE)
    if v2_after["sha256"] != v2_before["sha256"]:
        issues.append("historical V2 sqlite hash changed during Lane A")
    study1_after = file_fingerprint(STUDY1_SQLITE)
    ok = not issues
    return {
        "ok": ok,
        "issues": issues,
        "n_primary": len(primary),
        "n_primary_questions": len(qids),
        "by_model": dict(by_model),
        "by_L": {str(k): v for k, v in by_L.items()},
        "by_condition": dict(by_cond),
        "n_repeat_questions": len(repeat_q),
        "n_repeat_cells_with_3_obs": sum(
            1 for indexes in cells.values() if indexes == {0, 1, 2}
        ),
        "frozen_id_hash": selected_hash,
        "repeat_id_hash": repeat_hash,
        "historical_v2": dict(v2_before),
        "historical_v2_after": dict(v2_after),
        "study1_sqlite": study1_after,
        "paperDirection_sha256": sha256_file(PROJECT_ROOT / "paperDirection.txt"),
        "stop": not ok,
    }


def score_dominance_rows(primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in D1_MODELS:
        for L in (10.0, 20.0):
            grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
            for displayed in grid:
                subset = subset_study1(
                    primary, model=model, L=L, displayed=displayed, manipulated_only=True
                )
                stats = rate_with_ci(subset)
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "contrast_type": "rate_by_displayed_score",
                        "displayed_confidence": displayed,
                        "n_questions": stats["n_questions"],
                        "verification_rate": stats["rate"],
                        "rate_low": "",
                        "rate_high": "",
                        "difference": "",
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                        "verify_to_use": "",
                        "use_to_verify": "",
                        "no_change": "",
                    }
                )
            near = NEAR_THRESHOLD[L]
            extreme = EXTREME[L]
            for kind, low, high in (
                ("near_threshold", near[0], near[1]),
                ("extreme", extreme[0], extreme[1]),
            ):
                contrast = paired_contrast(primary, model=model, L=L, low=low, high=high)
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "contrast_type": kind,
                        "displayed_confidence": f"{low}->{high}",
                        "n_questions": contrast["n_questions"],
                        "verification_rate": "",
                        "rate_low": contrast["rate_low"],
                        "rate_high": contrast["rate_high"],
                        "difference": contrast["difference"],
                        "ci_lower": contrast["ci_lower"],
                        "ci_upper": contrast["ci_upper"],
                        "verify_to_use": contrast["verify_to_use"],
                        "use_to_verify": contrast["use_to_verify"],
                        "no_change": contrast["no_change"],
                    }
                )
            hidden = rate_with_ci(subset_study1(primary, model=model, L=L, condition="hidden"))
            visible = rate_with_ci(
                subset_study1(primary, model=model, L=L, condition="true_confidence_visible")
            )
            for label, stats in (("hidden", hidden), ("true_confidence_visible", visible)):
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "contrast_type": "reference_rate",
                        "displayed_confidence": label,
                        "n_questions": stats["n_questions"],
                        "verification_rate": stats["rate"],
                        "rate_low": "",
                        "rate_high": "",
                        "difference": "",
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                        "verify_to_use": "",
                        "use_to_verify": "",
                        "no_change": "",
                    }
                )
    return output


def _or_with_correction(a: float, b: float, c: float, d: float) -> dict[str, Any]:
    sparse = min(a, b, c, d) == 0
    aa, bb, cc, dd = (x + 0.5 for x in (a, b, c, d))
    odds = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    log_or = math.log(odds)
    return {
        "odds_ratio_haldane_anscombe": odds,
        "or_ci_lower": math.exp(log_or - 1.96 * se),
        "or_ci_upper": math.exp(log_or + 1.96 * se),
        "sparse_cells": sparse,
    }


def bootstrap_rate_difference(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    by_q = {
        row["question_id"]: (_is_verify(row["parsed_action"]), bool(row["stage1_correct"]))
        for row in rows
    }
    qids = sorted(by_q)

    def stat(ids: Sequence[str]) -> float:
        wrong_v = [by_q[qid][0] for qid in ids if not by_q[qid][1]]
        corr_v = [by_q[qid][0] for qid in ids if by_q[qid][1]]
        p_wrong = mean_or_nan(wrong_v) if wrong_v else float("nan")
        p_corr = mean_or_nan(corr_v) if corr_v else float("nan")
        if math.isnan(p_wrong) or math.isnan(p_corr):
            return float("nan")
        return p_wrong - p_corr

    estimate = stat(qids)
    rng = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [qids[rng.randrange(len(qids))] for _ in qids]
        value = stat(draw)
        if math.isfinite(value):
            samples.append(value)
    samples.sort()
    if samples:
        lower = samples[int(0.025 * (len(samples) - 1))]
        upper = samples[int(0.975 * (len(samples) - 1))]
    else:
        lower = upper = float("nan")
    return {
        "delta": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_valid_resamples": len(samples),
    }


def fixed_score_tables(primary: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contingency: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for model in D1_MODELS:
        for L in (10.0, 20.0):
            grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
            for displayed in grid:
                subset = subset_study1(
                    primary, model=model, L=L, displayed=displayed, manipulated_only=True
                )
                wrong_v = sum(
                    1
                    for row in subset
                    if not row["stage1_correct"] and _is_verify(row["parsed_action"])
                )
                wrong_u = sum(
                    1
                    for row in subset
                    if not row["stage1_correct"] and not _is_verify(row["parsed_action"])
                )
                corr_v = sum(
                    1
                    for row in subset
                    if row["stage1_correct"] and _is_verify(row["parsed_action"])
                )
                corr_u = sum(
                    1
                    for row in subset
                    if row["stage1_correct"] and not _is_verify(row["parsed_action"])
                )
                n_wrong = wrong_v + wrong_u
                n_correct = corr_v + corr_u
                n_verify = wrong_v + corr_v
                n_use = wrong_u + corr_u
                saturated = n_verify == 0 or n_use == 0 or n_wrong == 0 or n_correct == 0
                p_v_wrong = wrong_v / n_wrong if n_wrong else float("nan")
                p_v_corr = corr_v / n_correct if n_correct else float("nan")
                boot = bootstrap_rate_difference(subset)
                or_stats = _or_with_correction(wrong_v, wrong_u, corr_v, corr_u)
                row = {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "L": L,
                    "displayed_confidence": displayed,
                    "n_questions": len(subset),
                    "wrong_verify": wrong_v,
                    "wrong_use": wrong_u,
                    "correct_verify": corr_v,
                    "correct_use": corr_u,
                    "n_wrong": n_wrong,
                    "n_correct": n_correct,
                    "p_verify_given_wrong": p_v_wrong,
                    "p_verify_given_correct": p_v_corr,
                    "delta_p_verify_wrong_minus_correct": boot["delta"],
                    "delta_ci_lower": boot["ci_lower"],
                    "delta_ci_upper": boot["ci_upper"],
                    "identifiable": (not saturated),
                    "saturation_label": SATURATION_LABEL if saturated else "",
                    **or_stats,
                }
                contingency.append(row)
                sensitivity.append(
                    {
                        **row,
                        "useful_item_sensitivity_pilot": (
                            "NOT_IDENTIFIABLE_ACTION_SATURATED"
                            if saturated
                            else (
                                "PILOT_POSITIVE"
                                if math.isfinite(boot["delta"]) and boot["delta"] > 0
                                else "PILOT_NONPOSITIVE"
                            )
                        ),
                    }
                )
    return contingency, sensitivity


def cohens_kappa(a_v_b_v: int, a_v_b_u: int, a_u_b_v: int, a_u_b_u: int) -> float:
    n = a_v_b_v + a_v_b_u + a_u_b_v + a_u_b_u
    if n == 0:
        return float("nan")
    po = (a_v_b_v + a_u_b_u) / n
    p_a_v = (a_v_b_v + a_v_b_u) / n
    p_b_v = (a_v_b_v + a_u_b_v) / n
    pe = p_a_v * p_b_v + (1 - p_a_v) * (1 - p_b_v)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def hidden_continuity_rows(primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in D1_MODELS:
        for L in (10.0, 20.0):
            hidden = {
                row["question_id"]: _is_verify(row["parsed_action"])
                for row in subset_study1(primary, model=model, L=L, condition="hidden")
            }
            grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
            for displayed in grid:
                fixed = subset_study1(
                    primary, model=model, L=L, displayed=displayed, manipulated_only=True
                )
                hv_fv = hv_fu = hu_fv = hu_fu = 0
                agree_wrong = []
                agree_corr = []
                pairs = []
                for row in fixed:
                    qid = row["question_id"]
                    if qid not in hidden:
                        continue
                    h = hidden[qid]
                    f = _is_verify(row["parsed_action"])
                    pairs.append((qid, h, f))
                    if h and f:
                        hv_fv += 1
                    elif h and not f:
                        hv_fu += 1
                    elif (not h) and f:
                        hu_fv += 1
                    else:
                        hu_fu += 1
                    agree = int(h == f)
                    if row["stage1_correct"]:
                        agree_corr.append(agree)
                    else:
                        agree_wrong.append(agree)
                n = hv_fv + hv_fu + hu_fv + hu_fu
                agreement = (hv_fv + hu_fu) / n if n else float("nan")
                hidden_both = min(hv_fv + hv_fu, hu_fv + hu_fu)
                fixed_both = min(hv_fv + hu_fv, hv_fu + hu_fu)
                kappa_ok = hidden_both >= 5 and fixed_both >= 5
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "displayed_confidence": displayed,
                        "hidden_verify__fixed_verify": hv_fv,
                        "hidden_verify__fixed_use": hv_fu,
                        "hidden_use__fixed_verify": hu_fv,
                        "hidden_use__fixed_use": hu_fu,
                        "n": n,
                        "raw_agreement": agreement,
                        "agreement_wrong": mean_or_nan(agree_wrong),
                        "agreement_correct": mean_or_nan(agree_corr),
                        "cohens_kappa": cohens_kappa(hv_fv, hv_fu, hu_fv, hu_fu)
                        if kappa_ok
                        else "",
                        "kappa_reported": kappa_ok,
                    }
                )
    return output


def repeat_noise_rows(
    repeats: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in repeats:
        key = (
            row["question_id"],
            row["model_alias"],
            float(row["L"]),
            row["display_condition"],
            row.get("displayed_confidence"),
        )
        grouped[key][int(row["repeat_index"])] = row
    output: list[dict[str, Any]] = []
    # per-cell agreement
    for key, gens in grouped.items():
        if set(gens) != {0, 1, 2}:
            continue
        actions = [_is_verify(gens[i]["parsed_action"]) for i in (0, 1, 2)]
        n_verify = sum(actions)
        agree3 = int(len(set(actions)) == 1)
        split21 = int(sorted(actions).count(1) in {1, 2} and not agree3)
        pairs = [(0, 1), (0, 2), (1, 2)]
        pairwise = sum(actions[i] != actions[j] for i, j in pairs) / 3.0
        qid, model, L, cond, displayed = key
        correct = bool(gens[0]["stage1_correct"])
        output.append(
            {
                "section": "per_cell",
                "question_id": qid,
                "model_alias": model,
                "L": L,
                "display_condition": cond,
                "displayed_confidence": displayed,
                "stage1_correct": correct,
                "all_three_agree": agree3,
                "split_2_1": int(len(set(actions)) == 2),
                "pairwise_disagreement": pairwise,
                "verify_propensity": n_verify / 3.0,
                "primary_verify": actions[0],
            }
        )
    # directional persistence of wrong-vs-correct gap on 20-q subset
    repeat_q = {row["question_id"] for row in repeats}
    primary_sub = [row for row in primary if row["question_id"] in repeat_q]
    for model in D1_MODELS:
        for L in (10.0, 20.0):
            grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
            for displayed in grid:
                prim_cell = subset_study1(
                    primary_sub, model=model, L=L, displayed=displayed, manipulated_only=True
                )
                prim_delta = bootstrap_rate_difference(prim_cell)["delta"]
                # 3-gen mean: one row per question with fractional verify
                by_q: dict[str, list[int]] = defaultdict(list)
                correct_by = {}
                for row in repeats:
                    if row["model_alias"] != model or float(row["L"]) != L:
                        continue
                    if row.get("displayed_confidence") is None:
                        continue
                    if abs(float(row["displayed_confidence"]) - float(displayed)) > 1e-12:
                        continue
                    if not str(row["display_condition"]).startswith("manipulated"):
                        continue
                    by_q[row["question_id"]].append(_is_verify(row["parsed_action"]))
                    correct_by[row["question_id"]] = bool(row["stage1_correct"])
                fake_rows = []
                for qid, acts in by_q.items():
                    if len(acts) != 3:
                        continue
                    prop = sum(acts) / 3.0
                    fake_rows.append(
                        {
                            "question_id": qid,
                            "parsed_action": VERIFY if prop >= 0.5 else "USE_UNVERIFIED",
                            "verify_propensity": prop,
                            "stage1_correct": correct_by[qid],
                        }
                    )
                wrong = [row["verify_propensity"] for row in fake_rows if not row["stage1_correct"]]
                corr = [row["verify_propensity"] for row in fake_rows if row["stage1_correct"]]
                gen3_delta = mean_or_nan(wrong) - mean_or_nan(corr)
                output.append(
                    {
                        "section": "directional_persistence",
                        "model_alias": model,
                        "L": L,
                        "displayed_confidence": displayed,
                        "n_repeat_questions": len(fake_rows),
                        "primary_delta_p_v_wrong_minus_correct": prim_delta,
                        "gen3_mean_delta": gen3_delta,
                        "direction_same": (
                            math.isfinite(prim_delta)
                            and math.isfinite(gen3_delta)
                            and ((prim_delta > 0 and gen3_delta > 0) or (prim_delta < 0 and gen3_delta < 0) or (prim_delta == 0 and gen3_delta == 0))
                        ),
                    }
                )
    return output


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        return float(log_loss(y, clipped, labels=[0, 1]))
    except ValueError:
        return float("nan")


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def grouped_cv_incremental(primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Correctness ~ displayed score vs displayed score + action. Question-grouped CV."""
    output: list[dict[str, Any]] = []
    rng_seed = BOOTSTRAP_SEED
    for model in D1_MODELS:
        for L in (10.0, 20.0):
            rows = [
                row
                for row in primary
                if row["model_alias"] == model
                and float(row["L"]) == L
                and str(row["display_condition"]).startswith("manipulated")
                and row["displayed_confidence"] is not None
            ]
            if not rows:
                continue
            qids = np.array([row["question_id"] for row in rows])
            y = np.array([int(row["stage1_correct"]) for row in rows], dtype=int)
            score = np.array([float(row["displayed_confidence"]) for row in rows])
            action = np.array([_is_verify(row["parsed_action"]) for row in rows], dtype=float)
            unique_q = np.unique(qids)
            n_splits = min(5, len(unique_q))
            gkf = GroupKFold(n_splits=n_splits)
            oof_base = np.full(len(rows), np.nan)
            oof_aug = np.full(len(rows), np.nan)
            unstable = False
            for train_idx, test_idx in gkf.split(score, y, groups=qids):
                y_train = y[train_idx]
                if len(np.unique(y_train)) < 2:
                    unstable = True
                    continue
                Xb = score[train_idx].reshape(-1, 1)
                Xa = np.column_stack([score[train_idx], action[train_idx]])
                try:
                    m0 = LogisticRegression(max_iter=1000, solver="lbfgs")
                    m0.fit(Xb, y_train)
                    oof_base[test_idx] = m0.predict_proba(score[test_idx].reshape(-1, 1))[:, 1]
                    m1 = LogisticRegression(max_iter=1000, solver="lbfgs")
                    m1.fit(Xa, y_train)
                    oof_aug[test_idx] = m1.predict_proba(
                        np.column_stack([score[test_idx], action[test_idx]])
                    )[:, 1]
                except ValueError:
                    unstable = True
            mask = np.isfinite(oof_base) & np.isfinite(oof_aug)
            if mask.sum() < 20:
                output.append(
                    {
                        "model_alias": model,
                        "L": L,
                        "n_rows": len(rows),
                        "n_questions": len(unique_q),
                        "unstable": True,
                        "note": "grouped CV failed due to separation/sparsity",
                    }
                )
                continue
            y_m = y[mask]
            base_ll = _safe_log_loss(y_m, oof_base[mask])
            aug_ll = _safe_log_loss(y_m, oof_aug[mask])
            base_auc = _safe_auroc(y_m, oof_base[mask])
            aug_auc = _safe_auroc(y_m, oof_aug[mask])
            # question bootstrap of OOF metrics: resample questions, take their rows
            by_q_idx: dict[str, list[int]] = defaultdict(list)
            for i, qid in enumerate(qids):
                if mask[i]:
                    by_q_idx[str(qid)].append(i)
            q_list = sorted(by_q_idx)
            rng = random.Random(rng_seed)
            d_ll = []
            d_auc = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw = [q_list[rng.randrange(len(q_list))] for _ in q_list]
                idx = [j for qid in draw for j in by_q_idx[qid]]
                yy = y[idx]
                if len(np.unique(yy)) < 2:
                    continue
                d_ll.append(
                    _safe_log_loss(yy, oof_aug[idx]) - _safe_log_loss(yy, oof_base[idx])
                )
                d_auc.append(_safe_auroc(yy, oof_aug[idx]) - _safe_auroc(yy, oof_base[idx]))
            d_ll = [x for x in d_ll if math.isfinite(x)]
            d_auc = [x for x in d_auc if math.isfinite(x)]
            d_ll.sort()
            d_auc.sort()

            def ci(vals: list[float]) -> tuple[float, float]:
                if not vals:
                    return float("nan"), float("nan")
                return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]

            ll_lo, ll_hi = ci(d_ll)
            auc_lo, auc_hi = ci(d_auc)
            output.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "L": L,
                    "n_rows": int(mask.sum()),
                    "n_questions": len(unique_q),
                    "n_folds": n_splits,
                    "baseline_log_loss": base_ll,
                    "augmented_log_loss": aug_ll,
                    "delta_log_loss_action_minus_score": aug_ll - base_ll,
                    "delta_log_loss_ci_lower": ll_lo,
                    "delta_log_loss_ci_upper": ll_hi,
                    "baseline_auroc": base_auc,
                    "augmented_auroc": aug_auc,
                    "delta_auroc_action_minus_score": aug_auc - base_auc,
                    "delta_auroc_ci_lower": auc_lo,
                    "delta_auroc_ci_upper": auc_hi,
                    "unstable": unstable,
                    "note": "negative delta_log_loss means action improved prediction",
                }
            )
    return output


def fractional_verify_weights(q: np.ndarray, n_verify: float) -> np.ndarray:
    """Checkpoint A tie handling: lowest-q first, fractional inclusion at the cutoff."""
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


def error_catch_from_weights(weights: np.ndarray, wrong: np.ndarray) -> float:
    n_wrong = float(wrong.sum())
    if n_wrong == 0:
        return float("nan")
    return float((weights * wrong).sum()) / n_wrong


def crossfit_isotonic(
    q: np.ndarray, correct: np.ndarray, qids: np.ndarray, seed: int = 20260904
) -> np.ndarray:
    unique = np.unique(qids)
    n_splits = min(5, len(unique))
    gkf = GroupKFold(n_splits=n_splits)
    calibrated = np.full(len(q), np.nan)
    dummy = np.zeros(len(q))
    for train_idx, test_idx in gkf.split(dummy, correct, groups=qids):
        calibrator = fit_isotonic(q[train_idx].tolist(), correct[train_idx].tolist())
        calibrated[test_idx] = np.array(calibrator.predict(q[test_idx].tolist()))
    return calibrated


def routing_baseline_rows(v2_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in V2_MODELS_ROUTING:
        for owner in OWNERS:
            for L in ERROR_COSTS:
                hidden = [
                    row
                    for row in v2_rows
                    if row["model_alias"] == model
                    and row["decision_owner"] == owner
                    and float(row["L"]) == L
                    and row["confidence_visibility"] == "hidden"
                ]
                if len(hidden) != 500:
                    # still report whatever we have
                    pass
                if not hidden:
                    continue
                qids = np.array([row["question_id"] for row in hidden])
                q = np.array([float(row["probability_correct"]) for row in hidden])
                wrong = np.array([not row["stage1_correct"] for row in hidden])
                correct = np.array([row["stage1_correct"] for row in hidden], dtype=int)
                hid = np.array([_is_verify(row["parsed_action"]) for row in hidden])
                n_verify = float(hid.sum())
                catch_hidden = error_catch_from_weights(hid.astype(float), wrong)
                catch_raw = error_catch_from_weights(
                    fractional_verify_weights(q, n_verify), wrong
                )
                q_cal = crossfit_isotonic(q, correct, qids)
                catch_cal = error_catch_from_weights(
                    fractional_verify_weights(q_cal, n_verify), wrong
                )
                n_wrong = int(wrong.sum())
                output.append(
                    {
                        "section": "matched_budget_cell",
                        "model_alias": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "decision_owner": owner,
                        "L": L,
                        "n_questions": len(hidden),
                        "n_wrong": n_wrong,
                        "hidden_verify_budget": n_verify,
                        "hidden_verify_rate": n_verify / len(hidden),
                        "hidden_error_catch": catch_hidden,
                        "raw_confidence_error_catch_same_budget": catch_raw,
                        "calibrated_confidence_error_catch_same_budget": catch_cal,
                        "hidden_minus_raw_pp": _pp(catch_hidden - catch_raw)
                        if math.isfinite(catch_hidden) and math.isfinite(catch_raw)
                        else float("nan"),
                        "hidden_minus_calibrated_pp": _pp(catch_hidden - catch_cal)
                        if math.isfinite(catch_hidden) and math.isfinite(catch_cal)
                        else float("nan"),
                    }
                )
                for budget_frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                    n_bud = budget_frac * len(hidden)
                    output.append(
                        {
                            "section": "raw_q_catch_curve",
                            "model_alias": model,
                            "model_label": MODEL_LABELS.get(model, model),
                            "decision_owner": owner,
                            "L": L,
                            "n_questions": len(hidden),
                            "n_wrong": n_wrong,
                            "budget_fraction": budget_frac,
                            "raw_confidence_error_catch": error_catch_from_weights(
                                fractional_verify_weights(q, n_bud), wrong
                            ),
                            "calibrated_error_catch": error_catch_from_weights(
                                fractional_verify_weights(q_cal, n_bud), wrong
                            ),
                            "hidden_verify_rate": n_verify / len(hidden),
                            "hidden_error_catch": catch_hidden,
                        }
                    )
    return output


def power_precision_rows(primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Question-level precision/power for a future fixed-score or catch-rate study."""
    output: list[dict[str, Any]] = []
    error_rates = {}
    for model in D1_MODELS:
        rows = [row for row in primary if row["model_alias"] == model]
        # correctness is question-level; take one row per question
        by_q = {row["question_id"]: bool(row["stage1_correct"]) for row in rows}
        p_err = 1.0 - mean_or_nan(int(v) for v in by_q.values())
        error_rates[model] = p_err
    # Observed identifiable Claude-like deltas from non-saturated cells later filled
    scenarios = [
        ("tiny_5pp", 0.05),
        ("small_10pp", 0.10),
        ("medium_15pp", 0.15),
        ("large_20pp", 0.20),
        ("very_large_30pp", 0.30),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ns = (100, 200, 400, 600, 800, 1000)
    n_sims = 2000
    for model, p_err in error_rates.items():
        for n in ns:
            expected_wrong = n * p_err
            expected_correct = n * (1 - p_err)
            for name, delta in scenarios:
                # two-proportion difference CI width using expected counts
                # assume P(V|correct)=0.5, P(V|wrong)=0.5+delta clipped
                p0 = 0.50
                p1 = min(0.99, max(0.01, p0 + delta))
                se = math.sqrt(
                    p1 * (1 - p1) / max(expected_wrong, 1e-9)
                    + p0 * (1 - p0) / max(expected_correct, 1e-9)
                )
                ci_width = 3.92 * se
                # Monte Carlo power: two-sample test of proportions, question as unit
                rejects = 0
                for _ in range(n_sims):
                    n_w = rng.binomial(n, p_err)
                    n_c = n - n_w
                    if n_w < 2 or n_c < 2:
                        continue
                    v_w = rng.binomial(n_w, p1)
                    v_c = rng.binomial(n_c, p0)
                    phat1 = v_w / n_w
                    phat0 = v_c / n_c
                    p_pool = (v_w + v_c) / n
                    se_null = math.sqrt(p_pool * (1 - p_pool) * (1 / n_w + 1 / n_c))
                    if se_null <= 0:
                        continue
                    z = (phat1 - phat0) / se_null
                    if abs(z) >= 1.96:
                        rejects += 1
                output.append(
                    {
                        "model_alias": model,
                        "empirical_error_rate": p_err,
                        "N_questions": n,
                        "expected_n_wrong": expected_wrong,
                        "expected_n_correct": expected_correct,
                        "scenario": name,
                        "assumed_delta_p_verify_wrong_minus_correct": delta,
                        "approx_95ci_width": ci_width,
                        "sim_power_two_proportion_alpha05": rejects / n_sims,
                        "n_sims": n_sims,
                        "note": "Pilot precision check; not a confirmatory sample-size freeze.",
                    }
                )
    return output


def _plot_item_sensitivity(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    axes = axes.ravel()
    i = 0
    for model in D1_MODELS:
        for L in (10.0, 20.0):
            ax = axes[i]
            sub = [
                row
                for row in rows
                if row["model_alias"] == model and float(row["L"]) == L
            ]
            xs = [float(row["displayed_confidence"]) for row in sub]
            y1 = [row["p_verify_given_wrong"] for row in sub]
            y0 = [row["p_verify_given_correct"] for row in sub]
            ax.plot(xs, y1, marker="o", color="#D55E00", label="P(VERIFY | wrong)")
            ax.plot(xs, y0, marker="s", color="#0072B2", label="P(VERIFY | correct)")
            for x, row in zip(xs, sub):
                if row["saturation_label"]:
                    ax.axvline(x, color="#888888", alpha=0.25, linewidth=6)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{MODEL_LABELS[model]} L={int(L)}")
            ax.set_xlabel("Displayed score")
            ax.set_ylabel("Verification rate")
            if i == 0:
                ax.legend(frameon=False)
            i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_routing(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
    for ax, model in zip(axes, D1_MODELS):
        sub = [
            row
            for row in rows
            if row["model_alias"] == model
            and row.get("section") == "matched_budget_cell"
        ]
        # average over owner/L for a simple view, plus per-cell points
        budgets = [row["hidden_verify_rate"] for row in sub]
        hid = [row["hidden_error_catch"] for row in sub]
        raw = [row["raw_confidence_error_catch_same_budget"] for row in sub]
        cal = [row["calibrated_confidence_error_catch_same_budget"] for row in sub]
        ax.scatter(budgets, hid, label="Hidden action", color="#D55E00")
        ax.scatter(budgets, raw, label="Raw q, same budget", color="#0072B2", marker="s")
        ax.scatter(
            budgets, cal, label="Calibrated q, same budget", color="#009E73", marker="^"
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Verification budget")
        ax.set_ylabel("Error-catch rate")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_power(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for model, color in zip(D1_MODELS, ("#0072B2", "#D55E00")):
        for scenario, style in (("small_10pp", "-"), ("large_20pp", "--")):
            sub = [
                row
                for row in rows
                if row["model_alias"] == model and row["scenario"] == scenario
            ]
            xs = [row["N_questions"] for row in sub]
            ys = [row["sim_power_two_proportion_alpha05"] for row in sub]
            ax.plot(
                xs,
                ys,
                linestyle=style,
                color=color,
                marker="o",
                label=f"{MODEL_LABELS[model]} {scenario}",
            )
    ax.axhline(0.8, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("N questions")
    ax.set_ylabel("Simulated power")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_validation_md(payload: Mapping[str, Any], path: Path) -> None:
    issues = payload.get("issues") or []
    lines = [
        "# Lane A data validation",
        "",
        f"- Primary cells: **{payload['n_primary']}**",
        f"- Unique questions: **{payload['n_primary_questions']}** (target 100)",
        f"- Per-model: `{payload['by_model']}`",
        f"- Per-L: `{payload['by_L']}`",
        f"- Per-condition: `{payload['by_condition']}`",
        f"- Repeat questions: **{payload['n_repeat_questions']}** (target 20)",
        f"- Repeat cells with three observations: **{payload['n_repeat_cells_with_3_obs']}** / 560",
        f"- Frozen ID hash: `{payload['frozen_id_hash']}`",
        f"- Historical V2 sha256: `{payload['historical_v2']['sha256']}`",
        f"- Historical V2 unchanged during Lane A: **{payload['historical_v2']['sha256'] == payload['historical_v2_after']['sha256']}**",
        f"- Study 1 sqlite sha256: `{payload['study1_sqlite']['sha256']}`",
        f"- paperDirection.txt sha256: `{payload['paperDirection_sha256']}`",
        "",
    ]
    if issues:
        lines.append("## Issues")
        lines.extend(f"- {item}" for item in issues)
        lines.append("")
        lines.append("Lane A STOPPED because joins or counts are unreliable.")
    else:
        lines.append("No material validation issues. Correctness and question-ID joins are reliable.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_notes(
    *,
    contingency: Sequence[Mapping[str, Any]],
    cv_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    gpt_sat = [
        row
        for row in contingency
        if row["model_alias"] == "openai_gpt56_sol" and row["saturation_label"]
    ]
    claude_ident = [
        row
        for row in contingency
        if row["model_alias"] == "anthropic_sonnet5" and row["identifiable"]
    ]
    lines = [
        "# Lane A analysis notes",
        "",
        "Pilot signal check on the frozen 100-question Study-1 sample. Not a publication-grade evidence-retention estimate. Not confirmatory.",
        "",
        "## Units",
        "",
        "Question identity is the statistical unit whenever correctness is involved. Repeated L/condition cells of the same question are not treated as independent questions. The 20-question repeat subset keeps three generations as repeated observations of those questions.",
        "",
        "## Saturation",
        "",
        f"GPT manipulated cells labeled `{SATURATION_LABEL}`: **{len(gpt_sat)}** / 10.",
        "",
        "A constant binary action at a fixed displayed score is **not** evidence that internal evidence was destroyed or suppressed. It means item discrimination is not identifiable from that binary action in that cell.",
        "",
        "## Claude identifiable cells",
        "",
        f"{len(claude_ident)} of 10 Claude manipulated cells have both actions and both correctness classes represented enough to form a 2×2 table without a zero action margin.",
        "",
        "## Incremental prediction",
        "",
        "Grouped 5-fold CV by question ID. Baseline predicts Stage-1 correctness from the experimentally assigned displayed score. Augmented adds the VERIFY/USE action. Displayed score is assigned independently of correctness in manipulated cells, so the baseline is a near-null. If action carries useful item-specific error information, log-loss should fall and AUROC should rise. Complete separation is reported as unstable rather than forced.",
        "",
        "## Routing baseline",
        "",
        "Historical V2-B, 500 questions, GPT and Claude only. Hidden verification vs raw-q ranking vs cross-fitted isotonic-q ranking at the hidden policy's verification count. Ties use Checkpoint A fractional cutoff inclusion. Repeated L/owner cells are reported separately and are not treated as independent questions.",
        "",
        "## Power simulation",
        "",
        "Uses empirical Study-1 error rates and labeled hypothetical deltas. It does not declare a required N because an earlier critique proposed 600–1000. The table is the N → precision/power relationship.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_lane_a() -> dict[str, Any]:
    LANE_A_DIR.mkdir(parents=True, exist_ok=True)
    (LANE_A_DIR / "figures").mkdir(parents=True, exist_ok=True)
    v2_before = file_fingerprint(V2_SQLITE)
    study1_before = file_fingerprint(STUDY1_SQLITE)
    paper_before = sha256_file(PROJECT_ROOT / "paperDirection.txt")
    primary = load_study1_primary_rows()
    repeats = load_study1_repeat_rows()
    validation = validate_lane_a(primary, repeats, v2_before)
    write_validation_md(validation, LANE_A_DIR / "validation.md")
    if not validation["ok"]:
        return {"ok": False, "validation": validation, "api_calls": 0}
    dominance = score_dominance_rows(primary)
    contingency, sensitivity = fixed_score_tables(primary)
    continuity = hidden_continuity_rows(primary)
    noise = repeat_noise_rows(repeats, primary)
    cv_rows = grouped_cv_incremental(primary)
    v2_rows = load_v2_primary_verification()
    routing = routing_baseline_rows(v2_rows)
    power = power_precision_rows(primary)
    _write_csv(LANE_A_DIR / "score_dominance.csv", dominance)
    _write_csv(LANE_A_DIR / "fixed_score_correctness_contingency.csv", contingency)
    _write_csv(LANE_A_DIR / "useful_item_sensitivity.csv", sensitivity)
    _write_csv(LANE_A_DIR / "hidden_action_continuity.csv", continuity)
    _write_csv(LANE_A_DIR / "repeat_noise_decomposition.csv", noise)
    _write_csv(LANE_A_DIR / "grouped_cv_incremental_error_prediction.csv", cv_rows)
    matched = [row for row in routing if row.get("section") == "matched_budget_cell"]
    curves = [row for row in routing if row.get("section") == "raw_q_catch_curve"]
    _write_csv(LANE_A_DIR / "existing_routing_baseline.csv", matched)
    _write_csv(LANE_A_DIR / "existing_routing_catch_curves.csv", curves)
    _write_csv(LANE_A_DIR / "power_precision_simulation.csv", power)
    write_analysis_notes(
        contingency=contingency, cv_rows=cv_rows, path=LANE_A_DIR / "analysis_notes.md"
    )
    _plot_item_sensitivity(contingency, LANE_A_DIR / "figures" / "fixed_score_item_sensitivity.png")
    _plot_routing(routing, LANE_A_DIR / "figures" / "matched_budget_routing.png")
    _plot_power(power, LANE_A_DIR / "figures" / "power_precision.png")
    # reuse Study 1 dominance curves
    from .study1_analysis import manipulated_curve, _plot_response_curve, reference_rates

    for model in D1_MODELS:
        for L in (10.0, 20.0):
            points = manipulated_curve(primary, model, L)
            refs = reference_rates(primary, model, L)
            _plot_response_curve(
                points,
                refs,
                LANE_A_DIR / "figures" / f"score_dominance_{model}_L{int(L)}.png",
                f"{MODEL_LABELS[model]} L={int(L)} displayed-score verification",
            )
    v2_after = file_fingerprint(V2_SQLITE)
    study1_after = file_fingerprint(STUDY1_SQLITE)
    paper_after = sha256_file(PROJECT_ROOT / "paperDirection.txt")
    assert v2_before["sha256"] == v2_after["sha256"]
    assert study1_before["sha256"] == study1_after["sha256"]
    assert paper_before == paper_after
    summary = {
        "ok": True,
        "api_calls": 0,
        "validation": validation,
        "n_gpt_saturated_manipulated_cells": sum(
            1
            for row in contingency
            if row["model_alias"] == "openai_gpt56_sol" and row["saturation_label"]
        ),
        "n_claude_identifiable_manipulated_cells": sum(
            1
            for row in contingency
            if row["model_alias"] == "anthropic_sonnet5" and row["identifiable"]
        ),
        "cv": cv_rows,
        "routing_n": len(matched),
    }
    (LANE_A_DIR / "lane_a_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run_lane_a()
    print(json.dumps({"ok": summary["ok"], "api_calls": 0}, indent=2))
    if not summary["ok"]:
        raise SystemExit("Lane A stopped on validation failure")


if __name__ == "__main__":
    main()
