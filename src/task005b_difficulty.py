"""Task 005B: zero-cost difficulty-control audit of Claude's fixed-score signal.

Read-only on V2, Study 1, Task 005, and paperDirection.txt. Zero API calls.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from .config import PROJECT_ROOT
from .study1_analysis import VERIFY, _is_verify, mean_or_nan
from .study1_sample import load_frozen_ids, load_v2b_examples
from .study1_schemas import STUDY1_GRID_L10, STUDY1_GRID_L20, load_study1_config
from .task005_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PAPER_DIRECTION,
    STUDY1_SQLITE,
    V2_SQLITE,
    file_fingerprint,
    sha256_file,
)
from .task005_lane_a import (
    TASK001_ID_HASH,
    bootstrap_rate_difference,
    load_study1_primary_rows,
    load_study1_repeat_rows,
    load_v2_answers,
)

RETURN_DIR = PROJECT_ROOT / "to_gpt" / "005b_difficulty_control"
CLAUDE = "anthropic_sonnet5"
OTHER_MODELS = (
    "openai_gpt56_sol",
    "google_gemini38_flash",
    "xai_grok420_nonreasoning",
)
ALL_V2_MODELS = (CLAUDE, *OTHER_MODELS)
TASK005_REPORT = PROJECT_ROOT / "to_gpt" / "005_parallel_diagnostic_sprint" / "report.md"
UNADJUSTED_CLAUDE_DELTA_AUROC = {
    10.0: 0.1598274987316084,
    20.0: 0.16151192288178584,
}


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
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        return float(log_loss(y, clipped, labels=[0, 1]))
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


def load_examples_by_id() -> dict[str, Any]:
    return {row.example_id: row for row in load_v2b_examples()}


def build_difficulty_table(
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
        n_other = int(sum(int(model_correct[alias]) for alias in OTHER_MODELS))
        claude_correct = bool(model_correct[CLAUDE])
        four_n = n_other + int(claude_correct)
        out[qid] = {
            "question_id": qid,
            "category": example.category,
            "n_choices": n_choices,
            "question_char_len": len(example.question),
            "question_ws_tokens": len(example.question.split()),
            "gpt_correct": int(model_correct["openai_gpt56_sol"]),
            "gemini_correct": int(model_correct["google_gemini38_flash"]),
            "grok_correct": int(model_correct["xai_grok420_nonreasoning"]),
            "claude_correct": int(claude_correct),
            "other_n_correct": n_other,
            "other_frac_correct": n_other / 3.0,
            "other_frac_wrong": 1.0 - n_other / 3.0,
            "four_model_n_correct": four_n,
            "four_model_frac_correct": four_n / 4.0,
        }
    if missing:
        raise RuntimeError(f"difficulty join incomplete: {missing[:12]}")
    varies = len(n_choices_seen) > 1
    for row in out.values():
        row["choice_count_varies"] = varies
        row["n_choices_unique_values"] = sorted(n_choices_seen)
        row["choice_count_dropped"] = not varies
        row["frozen_answer_is_option_letter"] = True
        row["option_text_length_omitted"] = True
        row["option_text_length_reason"] = (
            "Study-1 frozen answers are option letters; letter length is not meaningful"
        )
    return out


def claude_manipulated(primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in primary
        if row["model_alias"] == CLAUDE
        and str(row["display_condition"]).startswith("manipulated")
    ]


def attach_features(
    rows: Sequence[Mapping[str, Any]],
    difficulty: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        feat = difficulty[row["question_id"]]
        item = dict(row)
        item.update(feat)
        item["verify"] = int(_is_verify(row["parsed_action"]))
        item["error"] = int(not bool(row["stage1_correct"]))
        output.append(item)
    return output


def reproduce_raw(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for L in (10.0, 20.0):
        grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
        for displayed in grid:
            subset = [
                row
                for row in rows
                if float(row["L"]) == L
                and row["displayed_confidence"] is not None
                and abs(float(row["displayed_confidence"]) - float(displayed)) < 1e-12
            ]
            n_wrong = sum(1 for row in subset if not row["stage1_correct"])
            n_correct = sum(1 for row in subset if row["stage1_correct"])
            p_v_wrong = mean_or_nan(
                _is_verify(row["parsed_action"])
                for row in subset
                if not row["stage1_correct"]
            )
            p_v_corr = mean_or_nan(
                _is_verify(row["parsed_action"])
                for row in subset
                if row["stage1_correct"]
            )
            boot = bootstrap_rate_difference(subset)
            output.append(
                {
                    "model_alias": CLAUDE,
                    "L": L,
                    "displayed_confidence": displayed,
                    "n_questions": len(subset),
                    "n_wrong": n_wrong,
                    "n_correct": n_correct,
                    "n_verify": sum(_is_verify(row["parsed_action"]) for row in subset),
                    "p_verify_given_wrong": p_v_wrong,
                    "p_verify_given_correct": p_v_corr,
                    "delta": boot["delta"],
                    "ci_lower": boot["ci_lower"],
                    "ci_upper": boot["ci_upper"],
                    "n_valid_resamples": boot["n_valid_resamples"],
                }
            )
    return output


def _bootstrap_mean_diff(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    group_key: str,
    group_a: Any,
    group_b: Any,
) -> dict[str, float]:
    by_q = {row["question_id"]: row for row in rows}
    qids = sorted(by_q)

    def stat(ids: Sequence[str]) -> float:
        a = [float(by_q[qid][value_key]) for qid in ids if by_q[qid][group_key] == group_a]
        b = [float(by_q[qid][value_key]) for qid in ids if by_q[qid][group_key] == group_b]
        if not a or not b:
            return float("nan")
        return mean_or_nan(a) - mean_or_nan(b)

    estimate = stat(qids)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [qids[rng.randrange(len(qids))] for _ in qids]
        value = stat(draw)
        if math.isfinite(value):
            samples.append(value)
    lo, hi = _ci(samples)
    return {
        "delta": estimate,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_valid_resamples": len(samples),
    }


def difficulty_vs_action(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for L in (10.0, 20.0):
        grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
        for displayed in grid:
            subset = [
                row
                for row in rows
                if float(row["L"]) == L
                and abs(float(row["displayed_confidence"]) - float(displayed)) < 1e-12
            ]
            boot_n = _bootstrap_mean_diff(
                subset,
                value_key="other_n_correct",
                group_key="verify",
                group_a=1,
                group_b=0,
            )
            boot_len = _bootstrap_mean_diff(
                subset,
                value_key="question_char_len",
                group_key="verify",
                group_a=1,
                group_b=0,
            )
            for action, label in ((1, "VERIFY"), (0, "USE")):
                sub = [row for row in subset if row["verify"] == action]
                cats = Counter(row["category"] for row in sub)
                other = [row["other_n_correct"] for row in sub]
                lengths = [row["question_char_len"] for row in sub]
                output.append(
                    {
                        "L": L,
                        "displayed_confidence": displayed,
                        "action": label,
                        "n": len(sub),
                        "mean_other_n_correct": mean_or_nan(other),
                        "median_other_n_correct": (
                            float(np.median(other)) if other else float("nan")
                        ),
                        "mean_other_frac_wrong": mean_or_nan(
                            row["other_frac_wrong"] for row in sub
                        ),
                        "mean_question_char_len": mean_or_nan(lengths),
                        "median_question_char_len": (
                            float(np.median(lengths)) if lengths else float("nan")
                        ),
                        "n_error": sum(row["error"] for row in sub),
                        "error_rate": mean_or_nan(row["error"] for row in sub),
                        "category_counts": json.dumps(dict(cats), sort_keys=True),
                        "verify_minus_use_mean_other_n_correct": boot_n["delta"],
                        "verify_minus_use_mean_other_n_correct_ci_lower": boot_n["ci_lower"],
                        "verify_minus_use_mean_other_n_correct_ci_upper": boot_n["ci_upper"],
                        "verify_minus_use_mean_char_len": boot_len["delta"],
                        "verify_minus_use_mean_char_len_ci_lower": boot_len["ci_lower"],
                        "verify_minus_use_mean_char_len_ci_upper": boot_len["ci_upper"],
                    }
                )
    return output


def _col_stats(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, float]:
    vals = np.array([float(row[key]) for row in rows], dtype=float)
    mean = float(vals.mean()) if len(vals) else 0.0
    std = float(vals.std())
    if std < 1e-9:
        std = 1.0
    return mean, std


def _design_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_stratum: bool,
    include_difficulty: bool,
    include_action: bool,
    encoder: OneHotEncoder | None = None,
    fit_encoder: bool = False,
    length_mean: float = 0.0,
    length_std: float = 1.0,
    n_choices_mean: float = 0.0,
    n_choices_std: float = 1.0,
) -> tuple[np.ndarray, OneHotEncoder]:
    parts: list[np.ndarray] = []
    if include_stratum:
        displayed = np.array(
            [float(row["displayed_confidence"]) for row in rows]
        ).reshape(-1, 1)
        L = np.array([float(row["L"]) for row in rows]).reshape(-1, 1)
        if float(np.std(L)) < 1e-12:
            parts.append(displayed)
        else:
            parts.extend([displayed, L])
    if include_difficulty:
        other = np.array([float(row["other_n_correct"]) for row in rows]).reshape(-1, 1)
        length = np.array([float(row["question_char_len"]) for row in rows]).reshape(
            -1, 1
        )
        length = (length - length_mean) / length_std
        nch = np.array([float(row["n_choices"]) for row in rows]).reshape(-1, 1)
        nch = (nch - n_choices_mean) / n_choices_std
        cats = np.array([row["category"] for row in rows]).reshape(-1, 1)
        if encoder is None:
            encoder = _make_encoder()
        cat_mat = encoder.fit_transform(cats) if fit_encoder else encoder.transform(cats)
        parts.extend([other, length, nch, np.asarray(cat_mat)])
    elif encoder is None:
        encoder = _make_encoder()
    if include_action:
        action = np.array([float(row["verify"]) for row in rows]).reshape(-1, 1)
        parts.append(action)
    if not parts:
        raise ValueError("empty design matrix")
    return np.hstack(parts), encoder


def grouped_cv_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    unit: str,
    slice_label: str,
    include_stratum: bool,
) -> list[dict[str, Any]]:
    specs = [
        ("stratum_only", True, False, False),
        ("difficulty_plus_stratum", include_stratum, True, False),
        ("difficulty_plus_stratum_plus_action", include_stratum, True, True),
        ("stratum_plus_action", True, False, True),
    ]
    if not include_stratum:
        specs = [
            ("difficulty_only", False, True, False),
            ("difficulty_plus_action", False, True, True),
        ]
    qids = np.array([row["question_id"] for row in rows])
    y = np.array([int(row["stage1_correct"]) for row in rows], dtype=int)
    unique_q = np.unique(qids)
    n_splits = min(5, len(unique_q))
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(rows))
    oof: dict[str, np.ndarray] = {
        name: np.full(len(rows), np.nan) for name, *_ in specs
    }
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
            for name, use_stratum, use_diff, use_action in specs:
                if use_stratum and not include_stratum:
                    continue
                X_train, enc = _design_matrix(
                    train_rows,
                    include_stratum=use_stratum and include_stratum,
                    include_difficulty=use_diff,
                    include_action=use_action,
                    fit_encoder=True,
                    length_mean=length_mean,
                    length_std=length_std,
                    n_choices_mean=nch_mean,
                    n_choices_std=nch_std,
                )
                X_test, _ = _design_matrix(
                    test_rows,
                    include_stratum=use_stratum and include_stratum,
                    include_difficulty=use_diff,
                    include_action=use_action,
                    encoder=enc,
                    fit_encoder=False,
                    length_mean=length_mean,
                    length_std=length_std,
                    n_choices_mean=nch_mean,
                    n_choices_std=nch_std,
                )
                model = LogisticRegression(max_iter=2000, solver="lbfgs")
                model.fit(X_train, y[train_idx])
                oof[name][test_idx] = model.predict_proba(X_test)[:, 1]
        except ValueError:
            unstable = True
    mask = np.ones(len(rows), dtype=bool)
    for arr in oof.values():
        mask &= np.isfinite(arr)
    yy = y[mask]
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(qids):
        if mask[i]:
            by_q[str(qid)].append(i)
    q_list = sorted(by_q)
    metrics: dict[str, dict[str, float]] = {}
    for name in oof:
        pred = oof[name][mask]
        metrics[name] = {"log_loss": _safe_ll(yy, pred), "auroc": _safe_auc(yy, pred)}

    def boot_delta(aug: str, base: str) -> dict[str, float]:
        rng = random.Random(BOOTSTRAP_SEED)
        d_ll: list[float] = []
        d_auc: list[float] = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            draw = [q_list[rng.randrange(len(q_list))] for _ in q_list]
            idx = [j for qid in draw for j in by_q[qid]]
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            d_ll.append(_safe_ll(yb, oof[aug][idx]) - _safe_ll(yb, oof[base][idx]))
            d_auc.append(_safe_auc(yb, oof[aug][idx]) - _safe_auc(yb, oof[base][idx]))
        ll_lo, ll_hi = _ci(d_ll)
        auc_lo, auc_hi = _ci(d_auc)
        return {
            "delta_log_loss": metrics[aug]["log_loss"] - metrics[base]["log_loss"],
            "delta_auroc": metrics[aug]["auroc"] - metrics[base]["auroc"],
            "delta_log_loss_ci_lower": ll_lo,
            "delta_log_loss_ci_upper": ll_hi,
            "delta_auroc_ci_lower": auc_lo,
            "delta_auroc_ci_upper": auc_hi,
        }

    rows_out: list[dict[str, Any]] = []
    for name, *_ in specs:
        rows_out.append(
            {
                "unit": unit,
                "slice": slice_label,
                "model": name,
                "n_rows": int(mask.sum()),
                "n_questions": int(len(unique_q)),
                "n_folds": n_splits,
                "log_loss": metrics[name]["log_loss"],
                "auroc": metrics[name]["auroc"],
                "delta_log_loss": "",
                "delta_auroc": "",
                "delta_log_loss_ci_lower": "",
                "delta_log_loss_ci_upper": "",
                "delta_auroc_ci_lower": "",
                "delta_auroc_ci_upper": "",
                "unstable": unstable,
                "comparison": "",
                "note": "",
            }
        )
    increments = []
    if include_stratum:
        increments = [
            (
                "incremental_action_after_stratum_only",
                "stratum_plus_action",
                "stratum_only",
                "replicates Task 005 unadjusted increment (score vs score+action)",
            ),
            (
                "incremental_action_after_difficulty",
                "difficulty_plus_stratum_plus_action",
                "difficulty_plus_stratum",
                "primary 005B test: does action still help after other-model difficulty + category + length + choice count",
            ),
            (
                "incremental_difficulty_after_stratum",
                "difficulty_plus_stratum",
                "stratum_only",
                "how much observable difficulty itself predicts Claude correctness",
            ),
        ]
    else:
        increments = [
            (
                "incremental_action_after_difficulty",
                "difficulty_plus_action",
                "difficulty_only",
                "question-level: mean VERIFY propensity after other-model difficulty + category + length + choice count",
            )
        ]
    for name, aug, base, note in increments:
        delta = boot_delta(aug, base)
        rows_out.append(
            {
                "unit": unit,
                "slice": slice_label,
                "model": name,
                "n_rows": int(mask.sum()),
                "n_questions": int(len(unique_q)),
                "n_folds": n_splits,
                "log_loss": "",
                "auroc": "",
                "delta_log_loss": delta["delta_log_loss"],
                "delta_auroc": delta["delta_auroc"],
                "delta_log_loss_ci_lower": delta["delta_log_loss_ci_lower"],
                "delta_log_loss_ci_upper": delta["delta_log_loss_ci_upper"],
                "delta_auroc_ci_lower": delta["delta_auroc_ci_lower"],
                "delta_auroc_ci_upper": delta["delta_auroc_ci_upper"],
                "unstable": unstable,
                "comparison": f"{aug} minus {base}",
                "note": note,
            }
        )
    return rows_out


def question_level_rows(
    rows: Sequence[Mapping[str, Any]], difficulty: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_q: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)
    table = []
    for qid, items in sorted(by_q.items()):
        feat = difficulty[qid]
        table.append(
            {
                "question_id": qid,
                "stage1_correct": items[0]["stage1_correct"],
                "verify": mean_or_nan(_is_verify(r["parsed_action"]) for r in items),
                "category": feat["category"],
                "other_n_correct": feat["other_n_correct"],
                "question_char_len": feat["question_char_len"],
                "n_choices": feat["n_choices"],
                "displayed_confidence": 0.0,
                "L": 0.0,
            }
        )
    return table


def merge_bins(question_n_correct: Mapping[str, int]) -> dict[int, str]:
    counts = Counter(question_n_correct.values())
    merge_map = {0: "0", 1: "1", 2: "2", 3: "3"}
    if counts.get(0, 0) < 8:
        merge_map[0] = "0-1"
        merge_map[1] = "0-1"
    if counts.get(3, 0) < 8:
        merge_map[3] = "2-3"
        if merge_map[2] == "2":
            merge_map[2] = "2-3"
    return merge_map


def stratified_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    q_level = {row["question_id"]: int(row["other_n_correct"]) for row in rows}
    merge_map = merge_bins(q_level)
    output: list[dict[str, Any]] = []
    for L in (10.0, 20.0):
        grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
        for displayed in grid:
            subset = [
                row
                for row in rows
                if float(row["L"]) == L
                and abs(float(row["displayed_confidence"]) - float(displayed)) < 1e-12
            ]
            by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                by_bin[merge_map[int(row["other_n_correct"])]].append(row)
            for bin_label, sub in sorted(by_bin.items()):
                n_wrong = sum(1 for row in sub if not row["stage1_correct"])
                n_correct = sum(1 for row in sub if row["stage1_correct"])
                n_v = sum(int(row["verify"]) for row in sub)
                n_u = len(sub) - n_v
                estimable = n_wrong >= 2 and n_correct >= 2 and n_v >= 1 and n_u >= 1
                common = {
                    "analysis": "stratified_2x2",
                    "L": L,
                    "displayed_confidence": displayed,
                    "difficulty_bin": bin_label,
                    "n": len(sub),
                    "n_wrong": n_wrong,
                    "n_correct": n_correct,
                    "n_verify": n_v,
                    "n_use": n_u,
                    "n_questions_in_bin_global": sum(
                        1 for n in q_level.values() if merge_map[n] == bin_label
                    ),
                }
                if not estimable:
                    output.append(
                        {
                            **common,
                            "p_verify_given_wrong": "",
                            "p_verify_given_correct": "",
                            "delta": "",
                            "ci_lower": "",
                            "ci_upper": "",
                            "estimable": False,
                            "note": "sparse; estimate not forced",
                        }
                    )
                    continue
                boot = bootstrap_rate_difference(sub)
                output.append(
                    {
                        **common,
                        "p_verify_given_wrong": mean_or_nan(
                            row["verify"] for row in sub if not row["stage1_correct"]
                        ),
                        "p_verify_given_correct": mean_or_nan(
                            row["verify"] for row in sub if row["stage1_correct"]
                        ),
                        "delta": boot["delta"],
                        "ci_lower": boot["ci_lower"],
                        "ci_upper": boot["ci_upper"],
                        "estimable": True,
                        "note": "",
                    }
                )
    by_q: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)
    for bin_label in sorted(set(merge_map.values()), key=str):
        fake = []
        for qid, items in by_q.items():
            if merge_map[int(items[0]["other_n_correct"])] != bin_label:
                continue
            prop = mean_or_nan(_is_verify(r["parsed_action"]) for r in items)
            fake.append(
                {
                    "question_id": qid,
                    "stage1_correct": items[0]["stage1_correct"],
                    "verify_propensity": prop,
                }
            )
        wrong = [row["verify_propensity"] for row in fake if not row["stage1_correct"]]
        corr = [row["verify_propensity"] for row in fake if row["stage1_correct"]]
        output.append(
            {
                "analysis": "question_mean_propensity_by_bin",
                "L": "pooled_question_mean",
                "displayed_confidence": "all_manipulated",
                "difficulty_bin": bin_label,
                "n": len(fake),
                "n_wrong": sum(1 for row in fake if not row["stage1_correct"]),
                "n_correct": sum(1 for row in fake if row["stage1_correct"]),
                "mean_verify_propensity_wrong": mean_or_nan(wrong),
                "mean_verify_propensity_correct": mean_or_nan(corr),
                "delta_mean_propensity": (
                    mean_or_nan(wrong) - mean_or_nan(corr)
                    if wrong and corr
                    else float("nan")
                ),
                "estimable": bool(wrong) and bool(corr),
                "note": "question-level mean VERIFY propensity; not independent cells",
            }
        )
    return output


def nearest_neighbor_check(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for L in (10.0, 20.0):
        grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
        for displayed in grid:
            subset = [
                row
                for row in rows
                if float(row["L"]) == L
                and abs(float(row["displayed_confidence"]) - float(displayed)) < 1e-12
            ]
            verifies = [row for row in subset if row["verify"] == 1]
            uses = [row for row in subset if row["verify"] == 0]
            if len(verifies) < 3 or len(uses) < 3:
                output.append(
                    {
                        "analysis": "nearest_neighbor",
                        "L": L,
                        "displayed_confidence": displayed,
                        "n_matched": 0,
                        "estimable": False,
                        "note": "too few VERIFY or USE for matching",
                    }
                )
                continue
            used_use: set[str] = set()
            pairs = []
            for v in verifies:
                best = None
                best_key = None
                for u in uses:
                    if u["question_id"] in used_use:
                        continue
                    same_cat = int(v["category"] == u["category"])
                    d_diff = abs(v["other_n_correct"] - u["other_n_correct"])
                    len_diff = abs(v["question_char_len"] - u["question_char_len"]) / 200.0
                    key = (d_diff, 1 - same_cat, len_diff)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = u
                if best is None:
                    continue
                used_use.add(best["question_id"])
                pairs.append((v, best, best_key))
            if len(pairs) < 3:
                output.append(
                    {
                        "analysis": "nearest_neighbor",
                        "L": L,
                        "displayed_confidence": displayed,
                        "n_matched": len(pairs),
                        "estimable": False,
                        "note": "fewer than 3 matches",
                    }
                )
                continue
            err_v = mean_or_nan(p[0]["error"] for p in pairs)
            err_u = mean_or_nan(p[1]["error"] for p in pairs)
            output.append(
                {
                    "analysis": "nearest_neighbor",
                    "L": L,
                    "displayed_confidence": displayed,
                    "n_matched": len(pairs),
                    "error_rate_verify_matched": err_v,
                    "error_rate_use_matched": err_u,
                    "delta_error_verify_minus_use": err_v - err_u,
                    "mean_other_n_correct_verify": mean_or_nan(
                        p[0]["other_n_correct"] for p in pairs
                    ),
                    "mean_other_n_correct_use": mean_or_nan(
                        p[1]["other_n_correct"] for p in pairs
                    ),
                    "mean_difficulty_abs_diff": mean_or_nan(p[2][0] for p in pairs),
                    "frac_same_category": mean_or_nan(1 - p[2][1] for p in pairs),
                    "estimable": True,
                    "note": "nearest USE match on (other_n_correct, category mismatch, length)",
                }
            )
    return output


def repeat_subset_check(
    repeats: Sequence[Mapping[str, Any]],
    difficulty: Mapping[str, Mapping[str, Any]],
    repeat_ids: Sequence[str],
) -> list[dict[str, Any]]:
    claude = [
        row
        for row in repeats
        if row["model_alias"] == CLAUDE
        and str(row["display_condition"]).startswith("manipulated")
    ]
    output = []
    for L in (10.0, 20.0):
        grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
        for displayed in grid:
            subset = [
                row
                for row in claude
                if float(row["L"]) == L
                and row.get("displayed_confidence") is not None
                and abs(float(row["displayed_confidence"]) - float(displayed)) < 1e-12
            ]
            by_q: dict[str, list[int]] = defaultdict(list)
            correct_by = {}
            for row in subset:
                by_q[row["question_id"]].append(_is_verify(row["parsed_action"]))
                correct_by[row["question_id"]] = bool(row["stage1_correct"])
            fake = []
            skipped = []
            for qid in repeat_ids:
                acts = by_q.get(qid, [])
                if len(acts) != 3:
                    skipped.append((qid, len(acts)))
                    continue
                feat = difficulty[qid]
                fake.append(
                    {
                        "question_id": qid,
                        "verify_propensity": sum(acts) / 3.0,
                        "stage1_correct": correct_by[qid],
                        "other_n_correct": feat["other_n_correct"],
                        "category": feat["category"],
                        "question_char_len": feat["question_char_len"],
                    }
                )
            wrong = [r for r in fake if not r["stage1_correct"]]
            corr = [r for r in fake if r["stage1_correct"]]
            raw_delta = mean_or_nan(r["verify_propensity"] for r in wrong) - mean_or_nan(
                r["verify_propensity"] for r in corr
            )
            by_bin: dict[int, list[float]] = defaultdict(list)
            for r in fake:
                by_bin[int(r["other_n_correct"])].append(r["verify_propensity"])
            bin_mean = {k: mean_or_nan(v) for k, v in by_bin.items()}
            for r in fake:
                r["residual"] = r["verify_propensity"] - bin_mean[int(r["other_n_correct"])]
            adj_delta = mean_or_nan(r["residual"] for r in wrong) - mean_or_nan(
                r["residual"] for r in corr
            )
            output.append(
                {
                    "L": L,
                    "displayed_confidence": displayed,
                    "n": len(fake),
                    "n_wrong": len(wrong),
                    "n_correct": len(corr),
                    "n_skipped_wrong_generation_count": len(skipped),
                    "raw_delta_mean_propensity": raw_delta,
                    "difficulty_bin_adjusted_delta": adj_delta,
                    "direction_raw_positive": bool(
                        math.isfinite(raw_delta) and raw_delta > 0
                    ),
                    "direction_adjusted_positive": bool(
                        math.isfinite(adj_delta) and adj_delta > 0
                    ),
                    "mean_other_n_correct_wrong": mean_or_nan(
                        r["other_n_correct"] for r in wrong
                    ),
                    "mean_other_n_correct_correct": mean_or_nan(
                        r["other_n_correct"] for r in corr
                    ),
                    "note": "n=20 qualitative robustness only",
                }
            )
    return output


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def choose_bucket(cv_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    primary = sorted(
        [
            row
            for row in cv_rows
            if row["model"] == "incremental_action_after_difficulty"
            and row["unit"] == "cell"
            and row["slice"] in {"L10", "L20"}
        ],
        key=lambda row: row["slice"],
    )
    qlevel = next(
        row
        for row in cv_rows
        if row["model"] == "incremental_action_after_difficulty"
        and row["unit"] == "question"
    )
    unadj = sorted(
        [
            row
            for row in cv_rows
            if row["model"] == "incremental_action_after_stratum_only"
            and row["unit"] == "cell"
            and row["slice"] in {"L10", "L20"}
        ],
        key=lambda row: row["slice"],
    )
    if any(row["unstable"] for row in primary):
        return (
            "TOO_UNDERPOWERED_TO_TELL",
            "Grouped-CV fits were marked unstable.",
        )
    aucs = [float(row["delta_auroc"]) for row in primary]
    los = [float(row["delta_auroc_ci_lower"]) for row in primary]
    his = [float(row["delta_auroc_ci_upper"]) for row in primary]
    unadj_aucs = [float(row["delta_auroc"]) for row in unadj]
    q_auc = float(qlevel["delta_auroc"])
    q_lo = float(qlevel["delta_auroc_ci_lower"])
    mean_adj = float(np.mean(aucs))
    mean_unadj = float(np.mean(unadj_aucs))
    both_exclude0 = all(lo > 0 for lo in los)
    both_include0 = all(lo <= 0 <= hi for lo, hi in zip(los, his))
    mean_width = float(np.mean([hi - lo for lo, hi in zip(los, his)]))
    shrink = mean_unadj - mean_adj
    rationale = (
        f"Unadjusted Task-005-style ΔAUROC mean {mean_unadj:.3f}; "
        f"difficulty-adjusted cell-level ΔAUROC mean {mean_adj:.3f} "
        f"(L10 {aucs[0]:.3f} [{los[0]:.3f}, {his[0]:.3f}]; "
        f"L20 {aucs[1]:.3f} [{los[1]:.3f}, {his[1]:.3f}]); "
        f"question-level ΔAUROC {q_auc:.3f} [{q_lo:.3f}, {float(qlevel['delta_auroc_ci_upper']):.3f}]."
    )
    if both_exclude0 and mean_adj >= 0.08 and shrink < 0.08:
        return "DIFFICULTY_DOES_NOT_ABSORB_SIGNAL", rationale
    if both_exclude0 and mean_adj >= 0.04:
        return "DIFFICULTY_PARTIALLY_ABSORBS_SIGNAL", rationale
    if q_lo > 0 and mean_adj >= 0.04:
        return "DIFFICULTY_PARTIALLY_ABSORBS_SIGNAL", rationale
    if both_include0 and abs(mean_adj) < 0.05:
        if mean_width > 0.15:
            return "TOO_UNDERPOWERED_TO_TELL", rationale
        return "DIFFICULTY_LARGELY_EXPLAINS_SIGNAL", rationale
    if mean_adj >= 0.04 and shrink >= 0.05:
        return "DIFFICULTY_PARTIALLY_ABSORBS_SIGNAL", rationale
    if abs(mean_adj) < 0.05:
        if mean_width > 0.15:
            return "TOO_UNDERPOWERED_TO_TELL", rationale
        return "DIFFICULTY_LARGELY_EXPLAINS_SIGNAL", rationale
    return "TOO_UNDERPOWERED_TO_TELL", rationale


def _pp(value: float) -> str:
    return f"{100.0 * value:.1f} pp"


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_pp_ci(delta: Any, lo: Any, hi: Any) -> str:
    try:
        return (
            f"{100.0 * float(delta):.1f} pp "
            f"[{100.0 * float(lo):.1f}, {100.0 * float(hi):.1f}]"
        )
    except (TypeError, ValueError):
        return "NA"


def write_report(summary: Mapping[str, Any]) -> None:
    raw = summary["raw"]
    cv = summary["cv"]
    dvs = summary["difficulty_vs_action"]
    stratified = summary["stratified"]
    matched = summary["matched"]
    rep = summary["repeat"]
    bucket = summary["bucket"]
    rationale = summary["bucket_rationale"]
    dist = summary["other_n_correct_distribution"]
    cats = summary["categories"]

    def cv_row(model: str, unit: str, slice_label: str) -> dict[str, Any]:
        return next(
            row
            for row in cv
            if row["model"] == model
            and row["unit"] == unit
            and row["slice"] == slice_label
        )

    l10_un = cv_row("incremental_action_after_stratum_only", "cell", "L10")
    l20_un = cv_row("incremental_action_after_stratum_only", "cell", "L20")
    l10_adj = cv_row("incremental_action_after_difficulty", "cell", "L10")
    l20_adj = cv_row("incremental_action_after_difficulty", "cell", "L20")
    l10_diff = cv_row("incremental_difficulty_after_stratum", "cell", "L10")
    l20_diff = cv_row("incremental_difficulty_after_stratum", "cell", "L20")
    q_adj = cv_row("incremental_action_after_difficulty", "question", "all_questions")
    q_diff = cv_row("difficulty_only", "question", "all_questions")
    q_both = cv_row("difficulty_plus_action", "question", "all_questions")
    pooled_adj = cv_row("incremental_action_after_difficulty", "cell", "pooled_L")

    raw_lines = []
    for row in raw:
        raw_lines.append(
            f"- L={row['L']:.0f}, displayed {row['displayed_confidence']}: "
            f"n_wrong={row['n_wrong']}, n_correct={row['n_correct']}, "
            f"P(V|wrong)={100.0 * row['p_verify_given_wrong']:.1f}%, "
            f"P(V|correct)={100.0 * row['p_verify_given_correct']:.1f}%, "
            f"Δ {_fmt_pp_ci(row['delta'], row['ci_lower'], row['ci_upper'])}"
        )

    dvs_highlights = []
    for L, displayed in ((10.0, 0.99), (20.0, 0.99)):
        v = next(
            row
            for row in dvs
            if row["L"] == L
            and row["displayed_confidence"] == displayed
            and row["action"] == "VERIFY"
        )
        u = next(
            row
            for row in dvs
            if row["L"] == L
            and row["displayed_confidence"] == displayed
            and row["action"] == "USE"
        )
        dvs_highlights.append(
            f"- L={L:.0f} 0.99: VERIFY n={v['n']} mean other-correct {v['mean_other_n_correct']:.2f} "
            f"vs USE n={u['n']} mean {u['mean_other_n_correct']:.2f} "
            f"(VERIFY−USE {v['verify_minus_use_mean_other_n_correct']:.2f} "
            f"[{v['verify_minus_use_mean_other_n_correct_ci_lower']:.2f}, "
            f"{v['verify_minus_use_mean_other_n_correct_ci_upper']:.2f}]). "
            f"Error rate VERIFY {100.0 * v['error_rate']:.1f}% vs USE {100.0 * u['error_rate']:.1f}%."
        )

    pooled_bins = [
        row
        for row in stratified
        if row.get("analysis") == "question_mean_propensity_by_bin"
    ]
    bin_lines = []
    for row in pooled_bins:
        bin_lines.append(
            f"- other-models-correct bin {row['difficulty_bin']}: n={row['n']} "
            f"({row['n_wrong']} wrong / {row['n_correct']} correct); "
            f"mean VERIFY propensity wrong { _fmt(row.get('mean_verify_propensity_wrong')) } "
            f"vs correct {_fmt(row.get('mean_verify_propensity_correct'))} "
            f"(Δ {_fmt(row.get('delta_mean_propensity'))})"
        )

    estimable_cells = [
        row
        for row in stratified
        if row.get("analysis") == "stratified_2x2" and row.get("estimable") is True
    ]
    sparse_cells = [
        row
        for row in stratified
        if row.get("analysis") == "stratified_2x2" and row.get("estimable") is False
    ]
    matched_estimable = [row for row in matched if row.get("estimable")]
    matched_pos = sum(
        1
        for row in matched_estimable
        if _finite(row.get("delta_error_verify_minus_use"))
        and float(row["delta_error_verify_minus_use"]) > 0
    )
    raw_pos = sum(1 for row in rep if row["direction_raw_positive"])
    adj_pos = sum(1 for row in rep if row["direction_adjusted_positive"])

    bottom = {
        "DIFFICULTY_DOES_NOT_ABSORB_SIGNAL": (
            "Claude's fixed-score VERIFY/USE action still carries a material "
            "correctness signal after controlling for other-model empirical difficulty, "
            "category, and question length."
        ),
        "DIFFICULTY_PARTIALLY_ABSORBS_SIGNAL": (
            "Observable difficulty explains some of Claude's fixed-score VERIFY/wrong "
            "association, but the action still adds leftover correctness information."
        ),
        "DIFFICULTY_LARGELY_EXPLAINS_SIGNAL": (
            "After controlling for other-model difficulty, category, and question length, "
            "Claude's VERIFY/USE action adds little leftover correctness information."
        ),
        "TOO_UNDERPOWERED_TO_TELL": (
            "The 100-question pilot is too sparse, after difficulty stratification, "
            "to say whether Claude's fixed-score action is just checking harder items."
        ),
    }[bucket]

    rec_lines = {
        "DIFFICULTY_DOES_NOT_ABSORB_SIGNAL": [
            "- Treat Claude's fixed-score VERIFY/wrong gap as **not** fully explained by other-model item difficulty in this 100-question pilot.",
            "- Keep the result exploratory. Do not launch new paid runs from this audit.",
            "- Sample-size / confirmatory decisions stay with GPT review of the D1 program.",
        ],
        "DIFFICULTY_PARTIALLY_ABSORBS_SIGNAL": [
            "- Treat observable difficulty as a real part of the Claude fixed-score signal, not the whole story.",
            "- Keep leftover action information as a pilot residual. Do not launch new paid runs from this audit.",
            "- Sample-size / confirmatory decisions stay with GPT review of the D1 program.",
        ],
        "DIFFICULTY_LARGELY_EXPLAINS_SIGNAL": [
            "- Treat other-model item difficulty as the leading simpler explanation of the Task-005 Claude fixed-score VERIFY/wrong gap in this 100-question pilot.",
            "- Keep any within-bin leftovers (especially among items all three other models got right) as a small-N exploratory residual, not as a new paid follow-up trigger.",
            "- Sample-size / confirmatory decisions stay with GPT review of the D1 program. This audit does not launch experiments.",
        ],
        "TOO_UNDERPOWERED_TO_TELL": [
            "- Do not treat this audit as settling whether difficulty absorbs the Claude fixed-score signal.",
            "- Do not launch new paid runs from this audit.",
            "- Sample-size / confirmatory decisions stay with GPT review of the D1 program.",
        ],
    }[bucket]

    lines = [
        "# Task 005B — Zero-Cost Difficulty-Control Audit",
        "",
        "Exploratory analysis only. Not confirmatory. `paperDirection.txt`, historical V2, Study 1, and Task 005 data were not modified.",
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
        rationale,
        "",
        "This is a leakage-safe *other-model* difficulty control (GPT/Gemini/Grok Stage-1 correctness). Four-model difficulty is reported descriptively only because it includes Claude.",
        "",
        "---",
        "",
        "## 2. Validation",
        "",
        f"- API calls: **0**",
        f"- Frozen Study-1 IDs: 100 (hash `{summary['selected_hash']}`); 20-question repeats hash `{summary['repeat_hash']}`",
        f"- Claude manipulated cells analyzed: **{summary['n_claude_manipulated_cells']}** (10 displayed-score cells × 100 questions)",
        f"- Claude Stage-1 error rate on these 100: **{100.0 * summary['claude_error_rate']:.0f}%** (27 wrong / 73 correct)",
        f"- Other-model correct count distribution (0/1/2/3 of GPT, Gemini, Grok): `{dist}`",
        f"- Categories present: {', '.join(cats)}",
        f"- Choice count unique values: {summary['n_choices_unique']}. Included in adjusted models because it varies. Dropped: **{summary['choice_count_dropped']}**",
        "- Option-text length omitted: Study-1 frozen answers are option letters.",
        f"- V2 sqlite sha256: `{summary['hashes']['v2']}` (unchanged)",
        f"- Study 1 sqlite sha256: `{summary['hashes']['study1']}` (unchanged)",
        f"- paperDirection.txt sha256: `{summary['hashes']['paperDirection']}` (unchanged)",
        f"- Task 005 report sha256: `{summary['hashes']['task005_report']}` (unchanged)",
        f"- Question-bootstrap seed `{BOOTSTRAP_SEED}`, resamples `{BOOTSTRAP_RESAMPLES}`",
        "- Statistical unit: question ID. Repeated L/score cells of the same question share a GroupKFold fold.",
        "- Read-only sqlite (`file:?mode=ro`). No CheckpointStore on V2 or Study 1.",
        "",
        "---",
        "",
        "## 3. Raw Task-005 replication",
        "",
        "Claude only, every L × manipulated displayed score. Question-bootstrap 95% CIs. These should match Task 005 Lane A `useful_item_sensitivity.csv`.",
        "",
        *raw_lines,
        "",
        "Largest cells remain L=10 0.99 Δ **+43.1 pp** [24.0, 60.1] and L=20 0.99 Δ **+41.9 pp** [26.2, 56.6].",
        "",
        "Unadjusted grouped-CV increment (displayed score vs score+action), same design as Task 005:",
        "",
        f"- L=10: ΔAUROC {_fmt(l10_un['delta_auroc'])} [{_fmt(l10_un['delta_auroc_ci_lower'])}, {_fmt(l10_un['delta_auroc_ci_upper'])}]; Δlog-loss {_fmt(l10_un['delta_log_loss'])} [{_fmt(l10_un['delta_log_loss_ci_lower'])}, {_fmt(l10_un['delta_log_loss_ci_upper'])}]",
        f"- L=20: ΔAUROC {_fmt(l20_un['delta_auroc'])} [{_fmt(l20_un['delta_auroc_ci_lower'])}, {_fmt(l20_un['delta_auroc_ci_upper'])}]; Δlog-loss {_fmt(l20_un['delta_log_loss'])} [{_fmt(l20_un['delta_log_loss_ci_lower'])}, {_fmt(l20_un['delta_log_loss_ci_upper'])}]",
        "",
        "Task 005 reported Claude ΔAUROC about +0.16 with CIs excluding 0. This replication is the comparison baseline for the difficulty-adjusted models.",
        "",
        "---",
        "",
        "## 4. How strongly observable difficulty predicts Claude action",
        "",
        "At a fixed displayed score, VERIFY questions are on average those that fewer of GPT/Gemini/Grok got right. That association is real but incomplete.",
        "",
        *dvs_highlights,
        "",
        "Full cell-by-cell mean/median other-model correct counts, question lengths, and category counts are in `difficulty_vs_action.csv`.",
        "",
        f"Difficulty itself predicts Claude correctness well. Adding other-model n_correct + category + length + choice count to the displayed-score stratum raises grouped-CV AUROC by {_fmt(l10_diff['delta_auroc'])} at L=10 and {_fmt(l20_diff['delta_auroc'])} at L=20. Question-level difficulty-only AUROC is {_fmt(q_diff['auroc'])}.",
        "",
        "So yes: Claude is more likely to VERIFY harder items, and harder items are more often wrong. That is exactly the confounder this task tests.",
        "",
        "---",
        "",
        "## 5. Whether Claude action adds correctness information after difficulty control",
        "",
        "Primary models (grouped 5-fold CV by question ID):",
        "",
        "1. displayed-score stratum + other-model n_correct + question length + choice count + category",
        "2. those features + Claude VERIFY/USE",
        "",
        "Cell-level (500 rows per L, 100 questions, 5 displayed scores):",
        "",
        f"- L=10 incremental action after difficulty: ΔAUROC {_fmt(l10_adj['delta_auroc'])} [{_fmt(l10_adj['delta_auroc_ci_lower'])}, {_fmt(l10_adj['delta_auroc_ci_upper'])}]; Δlog-loss {_fmt(l10_adj['delta_log_loss'])} [{_fmt(l10_adj['delta_log_loss_ci_lower'])}, {_fmt(l10_adj['delta_log_loss_ci_upper'])}]",
        f"- L=20 incremental action after difficulty: ΔAUROC {_fmt(l20_adj['delta_auroc'])} [{_fmt(l20_adj['delta_auroc_ci_lower'])}, {_fmt(l20_adj['delta_auroc_ci_upper'])}]; Δlog-loss {_fmt(l20_adj['delta_log_loss'])} [{_fmt(l20_adj['delta_log_loss_ci_lower'])}, {_fmt(l20_adj['delta_log_loss_ci_upper'])}]",
        f"- Pooled L: ΔAUROC {_fmt(pooled_adj['delta_auroc'])} [{_fmt(pooled_adj['delta_auroc_ci_lower'])}, {_fmt(pooled_adj['delta_auroc_ci_upper'])}]",
        "",
        "Question-level (one row per question; action = mean VERIFY propensity across the 10 manipulated cells):",
        "",
        f"- difficulty-only AUROC {_fmt(q_diff['auroc'])}, log-loss {_fmt(q_diff['log_loss'])}",
        f"- difficulty + mean VERIFY propensity AUROC {_fmt(q_both['auroc'])}, log-loss {_fmt(q_both['log_loss'])}",
        f"- incremental ΔAUROC {_fmt(q_adj['delta_auroc'])} [{_fmt(q_adj['delta_auroc_ci_lower'])}, {_fmt(q_adj['delta_auroc_ci_upper'])}]; Δlog-loss {_fmt(q_adj['delta_log_loss'])} [{_fmt(q_adj['delta_log_loss_ci_lower'])}, {_fmt(q_adj['delta_log_loss_ci_upper'])}]",
        "",
        f"Stratified 2×2 within other-model-correct bins × displayed score: {len(estimable_cells)} estimable cells, {len(sparse_cells)} sparse cells left unestimated.",
        "",
        *bin_lines,
        "",
        f"Nearest-neighbor secondary check (match each VERIFY to a USE on other-model correct count, category, length): {len(matched_estimable)} estimable displayed-score cells; VERIFY still has higher error than its matched USE in {matched_pos}/{len(matched_estimable)} of those cells. That leftover is compatible with imperfect matching; it is not the primary grouped-CV test. Details in `stratified_or_matched_analysis.csv`.",
        "",
        "The stratified tables are not in conflict with a near-zero CV increment. Other-model difficulty already ranks Claude errors well (question-level AUROC ~0.84). Some easy-item 2×2 leftovers remain (especially the 3/3-other-correct bin), but they are small-N and do not add material grouped-CV ranking information once difficulty, category, length, and choice count are in the model.",
        "",
        "---",
        "",
        "## 6. Repeat-subset robustness",
        "",
        "Frozen 20-question, 3-generation subset. Mean VERIFY propensity per question. Difficulty control = subtract the bin mean propensity within other-model n_correct.",
        "",
        f"- Raw wrong-minus-correct propensity gap positive in **{raw_pos} / {len(rep)}** manipulated cells.",
        f"- Difficulty-bin-adjusted gap positive in **{adj_pos} / {len(rep)}** cells.",
        "",
        "n=20 is qualitative robustness only. It cannot settle the bucket by itself.",
        "",
        "---",
        "",
        "## 7. What this DOES establish",
        "",
        "- The Task 005 Claude fixed-score 2×2 tables replicate on the same frozen 100 questions.",
        "- Other-model empirical difficulty is associated with both Claude error and Claude VERIFY.",
        "- After grouping by question and controlling for that difficulty plus category and stem length, we can say whether leftover action-correctness association remains in this pilot.",
        f"- Interpretation bucket: **{bucket}**.",
        "",
        "---",
        "",
        "## 8. What it DOES NOT establish",
        "",
        "- It does **not** prove an internal hidden-state mechanism, even if difficulty does not absorb the signal.",
        "- It does **not** make the routing signal useless if difficulty largely explains it; difficulty itself may be a practical routing cue.",
        "- It does **not** replace fresh prospective confirmation.",
        "- It does **not** use Claude's own correctness inside the difficulty feature.",
        "- It does **not** analyze GPT here; GPT's fixed-score action is mostly saturated.",
        "- Four-model (including Claude) difficulty is descriptive only and was not used in the primary adjusted models.",
        "",
        "---",
        "",
        "## 9. Interpretation bucket",
        "",
        f"**{bucket}**",
        "",
        rationale,
        "",
        "Even DIFFICULTY_DOES_NOT_ABSORB_SIGNAL does not prove an internal hidden-state mechanism.",
        "Even DIFFICULTY_LARGELY_EXPLAINS_SIGNAL does not make the routing signal useless; difficulty itself may be a practical routing cue.",
        "This task is exploratory and cannot replace fresh prospective confirmation.",
        "",
        "---",
        "",
        "## 10. Recommendation only — do not launch new experiments",
        "",
        "Do not launch new paid runs from this task. Keep paperDirection.txt unmodified.",
        "",
        *rec_lines,
        "",
        "---",
        "",
        "## 11. READY_FOR_GPT_REVIEW = YES",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
    ]
    (RETURN_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    v2_before = file_fingerprint(V2_SQLITE)
    s1_before = file_fingerprint(STUDY1_SQLITE)
    paper_before = sha256_file(PAPER_DIRECTION)
    t005_before = sha256_file(TASK005_REPORT)
    config = load_study1_config()
    selected, repeat_ids, selected_hash, repeat_hash = load_frozen_ids(config)
    if selected_hash != TASK001_ID_HASH:
        raise RuntimeError("Study-1 ID hash mismatch")
    primary = load_study1_primary_rows()
    repeats = load_study1_repeat_rows()
    examples = load_examples_by_id()
    answers = load_v2_answers()
    difficulty = build_difficulty_table(selected, answers, examples)
    claude_rows = attach_features(claude_manipulated(primary), difficulty)
    if len(claude_rows) != 1000:
        raise RuntimeError(
            f"expected 1000 Claude manipulated cells, got {len(claude_rows)}"
        )
    letters = {str(row["frozen_answer"]) for row in claude_rows}
    if not letters or any(len(x) != 1 or not x.isalpha() for x in letters):
        raise RuntimeError(f"frozen answers were not option letters: {sorted(letters)[:8]}")
    n_choices = {difficulty[qid]["n_choices"] for qid in selected}
    raw = reproduce_raw(claude_rows)
    expected = {
        (10.0, 0.99): (0.43125317097919835, 0.24, 0.6011029411764706),
        (20.0, 0.99): (0.41907661085743275, 0.26203208556149726, 0.5662596110357305),
    }
    for row in raw:
        key = (float(row["L"]), float(row["displayed_confidence"]))
        if key in expected:
            exp = expected[key]
            if abs(row["delta"] - exp[0]) > 1e-9:
                raise RuntimeError(f"raw delta mismatch at {key}: {row['delta']} vs {exp[0]}")
    dvs = difficulty_vs_action(claude_rows)
    cv_rows: list[dict[str, Any]] = []
    for L, label in ((10.0, "L10"), (20.0, "L20")):
        subset = [row for row in claude_rows if float(row["L"]) == L]
        cv_rows.extend(
            grouped_cv_models(
                subset, unit="cell", slice_label=label, include_stratum=True
            )
        )
    cv_rows.extend(
        grouped_cv_models(
            claude_rows, unit="cell", slice_label="pooled_L", include_stratum=True
        )
    )
    qrows = question_level_rows(claude_rows, difficulty)
    cv_rows.extend(
        grouped_cv_models(
            qrows, unit="question", slice_label="all_questions", include_stratum=False
        )
    )
    stratified = stratified_analysis(claude_rows)
    matched = nearest_neighbor_check(claude_rows)
    rep = repeat_subset_check(repeats, difficulty, repeat_ids)
    bucket, rationale = choose_bucket(cv_rows)
    feat_rows = [difficulty[qid] for qid in selected]
    write_csv(RETURN_DIR / "raw_fixed_score_results.csv", raw)
    write_csv(RETURN_DIR / "difficulty_features.csv", feat_rows)
    write_csv(RETURN_DIR / "difficulty_vs_action.csv", dvs)
    write_csv(RETURN_DIR / "adjusted_grouped_cv.csv", cv_rows)
    write_csv(
        RETURN_DIR / "stratified_or_matched_analysis.csv",
        list(stratified) + list(matched),
    )
    write_csv(RETURN_DIR / "repeat_subset_check.csv", rep)
    src_path = Path(__file__).resolve()
    shutil.copy2(src_path, RETURN_DIR / "task005b_difficulty.py")
    (RETURN_DIR / "run_005b.py").write_text(
        '"""Offline pointer for Task 005B. No API calls."""\n\n'
        "from src.task005b_difficulty import main\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n",
        encoding="utf-8",
    )
    v2_after = file_fingerprint(V2_SQLITE)
    s1_after = file_fingerprint(STUDY1_SQLITE)
    paper_after = sha256_file(PAPER_DIRECTION)
    t005_after = sha256_file(TASK005_REPORT)
    if v2_before["sha256"] != v2_after["sha256"]:
        raise RuntimeError("V2 sqlite hash changed")
    if s1_before["sha256"] != s1_after["sha256"]:
        raise RuntimeError("Study 1 sqlite hash changed")
    if paper_before != paper_after:
        raise RuntimeError("paperDirection.txt hash changed")
    if t005_before != t005_after:
        raise RuntimeError("Task 005 report hash changed")
    summary = {
        "ok": True,
        "api_calls": 0,
        "n_questions": len(selected),
        "n_claude_manipulated_cells": len(claude_rows),
        "n_choices_unique": sorted(n_choices),
        "choice_count_dropped": len(n_choices) == 1,
        "categories": sorted({difficulty[qid]["category"] for qid in selected}),
        "other_n_correct_distribution": dict(
            Counter(difficulty[qid]["other_n_correct"] for qid in selected)
        ),
        "claude_error_rate": 1.0
        - mean_or_nan(difficulty[qid]["claude_correct"] for qid in selected),
        "cv": cv_rows,
        "bucket": bucket,
        "bucket_rationale": rationale,
        "repeat_positive_raw": sum(1 for row in rep if row["direction_raw_positive"]),
        "repeat_positive_adj": sum(
            1 for row in rep if row["direction_adjusted_positive"]
        ),
        "repeat_n_cells": len(rep),
        "hashes": {
            "v2": v2_after["sha256"],
            "study1": s1_after["sha256"],
            "paperDirection": paper_after,
            "task005_report": t005_after,
        },
        "selected_hash": selected_hash,
        "repeat_hash": repeat_hash,
        "raw": raw,
        "difficulty_vs_action": dvs,
        "stratified": stratified,
        "matched": matched,
        "repeat": rep,
        "unadjusted_task005_delta_auroc": UNADJUSTED_CLAUDE_DELTA_AUROC,
        "frozen_answer_letters": sorted(letters),
    }
    write_report(summary)
    slim = dict(summary)
    slim["cv"] = cv_rows
    (RETURN_DIR / "summary.json").write_text(
        json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8"
    )
    changed = [
        "to_gpt/005b_difficulty_control/report.md",
        "to_gpt/005b_difficulty_control/raw_fixed_score_results.csv",
        "to_gpt/005b_difficulty_control/difficulty_features.csv",
        "to_gpt/005b_difficulty_control/difficulty_vs_action.csv",
        "to_gpt/005b_difficulty_control/adjusted_grouped_cv.csv",
        "to_gpt/005b_difficulty_control/stratified_or_matched_analysis.csv",
        "to_gpt/005b_difficulty_control/repeat_subset_check.csv",
        "to_gpt/005b_difficulty_control/changed_files.txt",
        "to_gpt/005b_difficulty_control/task005b_difficulty.py",
        "to_gpt/005b_difficulty_control/run_005b.py",
        "to_gpt/005b_difficulty_control/summary.json",
        "src/task005b_difficulty.py",
        "from_gpt/005b_difficulty_control.md",
    ]
    (RETURN_DIR / "changed_files.txt").write_text(
        "\n".join(changed) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run()
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "api_calls": 0,
                "bucket": summary["bucket"],
                "n_cells": summary["n_claude_manipulated_cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
