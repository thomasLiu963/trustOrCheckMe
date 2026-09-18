"""Task 007 Lane C: score control vs empirical item-difficulty / prioritization."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .study1_analysis import MODEL_LABELS, _pyplot, mean_or_nan
from .study1_sample import load_v2b_examples
from .task005_lane_a import load_v2_answers
from .task006_common import SCORE_CONDITIONS, STAKES_FAMILIES
from .task007_common import (
    ALL_V2_MODELS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CORRECT_KEY,
    GPT_CLAUDE,
    LANE_C_DIR,
    OTHER_BY_TARGET,
    write_csv,
)

HARD_BINS = {0, 1}
MEDIUM_BINS = {2}
EASY_BINS = {3}


def difficulty_bin(n_other: int) -> str:
    if n_other in HARD_BINS:
        return "hard_0to1"
    if n_other in MEDIUM_BINS:
        return "medium_2"
    return "easy_3"


def build_difficulty_rows(question_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    examples = {row.example_id: row for row in load_v2b_examples()}
    answers = load_v2_answers()
    out: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for qid in question_ids:
        example = examples[qid]
        flags: dict[str, int] = {}
        for alias in ALL_V2_MODELS:
            rec = answers.get((qid, alias))
            if rec is None:
                missing.append(f"{qid}/{alias}")
                continue
            flags[alias] = int(bool(rec["is_correct"]))
        if len(flags) != 4:
            continue
        out[qid] = {
            "question_id": qid,
            "category": example.category,
            "n_choices": len(example.choices),
            "question_char_len": len(example.question),
            "gpt_correct": flags["openai_gpt56_sol"],
            "claude_correct": flags["anthropic_sonnet5"],
            "gemini_correct": flags["google_gemini38_flash"],
            "grok_correct": flags["xai_grok420_nonreasoning"],
        }
    if missing:
        raise RuntimeError(f"difficulty join incomplete: {missing[:8]}")
    return out


def attach_difficulty(
    table: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in table:
        feat = features[row["question_id"]]
        others = OTHER_BY_TARGET[row["model_alias"]]
        other_n = int(sum(int(feat[CORRECT_KEY[alias]]) for alias in others))
        target_key = CORRECT_KEY[row["model_alias"]]
        item = dict(row)
        item.update(
            {
                "other_n_correct": other_n,
                "other_frac_wrong": 1.0 - other_n / 3.0,
                "difficulty_bin": difficulty_bin(other_n),
                "target_correct": int(feat[target_key]),
                "category": feat["category"],
                "n_choices": feat["n_choices"],
                "question_char_len": feat["question_char_len"],
            }
        )
        if int(item["stage1_correct"]) != int(item["target_correct"]):
            raise RuntimeError(
                f"target correctness leak/mismatch {row['question_id']}/{row['model_alias']}"
            )
        output.append(item)
    return output


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _ci(vals: Sequence[float]) -> tuple[float, float]:
    clean = sorted(v for v in vals if math.isfinite(v))
    if not clean:
        return float("nan"), float("nan")
    return clean[int(0.025 * (len(clean) - 1))], clean[int(0.975 * (len(clean) - 1))]


def difficulty_by_condition(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    aliases = sorted({row["model_alias"] for row in rows})
    for model in aliases:
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                cell = [
                    row
                    for row in rows
                    if row["model_alias"] == model
                    and row["family"] == family
                    and row["score_condition"] == condition
                ]
                by_q = {row["question_id"]: row for row in cell}
                qids = sorted(by_q)
                verifies = [row for row in cell if row["verify"] == 1]
                uses = [row for row in cell if row["verify"] == 0]
                y = np.array([int(row["verify"]) for row in cell], dtype=int)
                hard = np.array([float(row["other_frac_wrong"]) for row in cell])
                auroc = _safe_auc(y, hard)
                rng = random.Random(BOOTSTRAP_SEED)
                aucs = []
                gaps = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    draw = [qids[rng.randrange(len(qids))] for _ in qids]
                    sub = [by_q[qid] for qid in draw]
                    yy = np.array([int(r["verify"]) for r in sub])
                    hh = np.array([float(r["other_frac_wrong"]) for r in sub])
                    aucs.append(_safe_auc(yy, hh))
                    hard_v = mean_or_nan(
                        r["verify"] for r in sub if r["difficulty_bin"] == "hard_0to1"
                    )
                    easy_v = mean_or_nan(
                        r["verify"] for r in sub if r["difficulty_bin"] == "easy_3"
                    )
                    if math.isfinite(hard_v) and math.isfinite(easy_v):
                        gaps.append(hard_v - easy_v)
                auc_lo, auc_hi = _ci(aucs)
                gap = mean_or_nan(
                    r["verify"] for r in cell if r["difficulty_bin"] == "hard_0to1"
                ) - mean_or_nan(
                    r["verify"] for r in cell if r["difficulty_bin"] == "easy_3"
                )
                gap_lo, gap_hi = _ci(gaps)
                for bin_name in ("hard_0to1", "medium_2", "easy_3"):
                    sub = [row for row in cell if row["difficulty_bin"] == bin_name]
                    output.append(
                        {
                            "model_alias": model,
                            "family": family,
                            "score_condition": condition,
                            "difficulty_bin": bin_name,
                            "n": len(sub),
                            "verify_rate": mean_or_nan(row["verify"] for row in sub),
                            "mean_other_n_correct_verify": mean_or_nan(
                                row["other_n_correct"] for row in verifies
                            ),
                            "mean_other_n_correct_use": mean_or_nan(
                                row["other_n_correct"] for row in uses
                            ),
                            "auroc_difficulty_for_verify": auroc,
                            "auroc_ci_lower": auc_lo,
                            "auroc_ci_upper": auc_hi,
                            "hard_minus_easy_verify": gap,
                            "hard_minus_easy_ci_lower": gap_lo,
                            "hard_minus_easy_ci_upper": gap_hi,
                            "n_verify": len(verifies),
                            "n_use": len(uses),
                        }
                    )
    return output


def interaction_models(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            visible = [
                row
                for row in rows
                if row["model_alias"] == model
                and row["family"] == family
                and row["score_condition"] in {"displayed_0.70", "displayed_0.90", "displayed_0.99"}
            ]
            by_q: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in visible:
                by_q[row["question_id"]].append(row)
            qids = sorted(by_q)
            y = np.array([int(row["verify"]) for row in visible], dtype=int)
            score = np.array([float(row["displayed_confidence"]) for row in visible])
            diff = np.array([float(row["other_n_correct"]) for row in visible])
            X = np.column_stack([score, diff, score * diff])
            unstable = False
            try:
                clf = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf.fit(X, y)
                coefs = clf.coef_[0]
            except ValueError:
                unstable = True
                coefs = np.array([float("nan")] * 3)
            rng = random.Random(BOOTSTRAP_SEED)
            boots = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw = [qids[rng.randrange(len(qids))] for _ in qids]
                sub = [row for qid in draw for row in by_q[qid]]
                yy = np.array([int(row["verify"]) for row in sub], dtype=int)
                if len(np.unique(yy)) < 2:
                    continue
                ss = np.array([float(row["displayed_confidence"]) for row in sub])
                dd = np.array([float(row["other_n_correct"]) for row in sub])
                XX = np.column_stack([ss, dd, ss * dd])
                try:
                    m = LogisticRegression(max_iter=2000, solver="lbfgs")
                    m.fit(XX, yy)
                    boots.append(m.coef_[0][2])
                except ValueError:
                    continue
            lo, hi = _ci(boots)
            output.append(
                {
                    "model_alias": model,
                    "family": family,
                    "n_rows": len(visible),
                    "n_questions": len(qids),
                    "coef_score": float(coefs[0]),
                    "coef_other_n_correct": float(coefs[1]),
                    "coef_score_x_difficulty": float(coefs[2]),
                    "interaction_ci_lower": lo,
                    "interaction_ci_upper": hi,
                    "unstable": unstable,
                    "note": "positive interaction: higher displayed score makes VERIFY more associated with easier items (higher other_n_correct). Clustered question bootstrap.",
                }
            )
    return output


def marginal_curves(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in sorted({row["model_alias"] for row in rows}):
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                for n_corr in (0, 1, 2, 3):
                    sub = [
                        row
                        for row in rows
                        if row["model_alias"] == model
                        and row["family"] == family
                        and row["score_condition"] == condition
                        and int(row["other_n_correct"]) == n_corr
                    ]
                    output.append(
                        {
                            "model_alias": model,
                            "family": family,
                            "score_condition": condition,
                            "other_n_correct": n_corr,
                            "difficulty_bin": difficulty_bin(n_corr),
                            "n": len(sub),
                            "verify_rate": mean_or_nan(row["verify"] for row in sub),
                        }
                    )
    return output


def switch_set_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            by_q: dict[str, dict[float, Mapping[str, Any]]] = defaultdict(dict)
            for row in rows:
                if row["model_alias"] != model or row["family"] != family:
                    continue
                if row["displayed_confidence"] is None:
                    continue
                by_q[row["question_id"]][float(row["displayed_confidence"])] = row
            for lo, hi in ((0.70, 0.90), (0.90, 0.99)):
                remain_v, switch_vu, remain_u, switch_uv = [], [], [], []
                for qid, scores in by_q.items():
                    if lo not in scores or hi not in scores:
                        continue
                    a = int(scores[lo]["verify"])
                    b = int(scores[hi]["verify"])
                    rec = scores[lo]
                    if a == 1 and b == 1:
                        remain_v.append(rec)
                    elif a == 1 and b == 0:
                        switch_vu.append(rec)
                    elif a == 0 and b == 0:
                        remain_u.append(rec)
                    else:
                        switch_uv.append(rec)
                output.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "from_score": lo,
                        "to_score": hi,
                        "n_remain_verify": len(remain_v),
                        "n_switch_verify_to_use": len(switch_vu),
                        "n_remain_use": len(remain_u),
                        "n_switch_use_to_verify": len(switch_uv),
                        "mean_other_n_correct_remain_verify": mean_or_nan(
                            r["other_n_correct"] for r in remain_v
                        ),
                        "mean_other_n_correct_switch_v_to_u": mean_or_nan(
                            r["other_n_correct"] for r in switch_vu
                        ),
                        "mean_other_n_correct_remain_use": mean_or_nan(
                            r["other_n_correct"] for r in remain_u
                        ),
                        "mean_other_n_correct_switch_u_to_v": mean_or_nan(
                            r["other_n_correct"] for r in switch_uv
                        ),
                        "mean_error_remain_verify": mean_or_nan(
                            1 - int(r["stage1_correct"]) for r in remain_v
                        ),
                        "mean_error_switch_v_to_u": mean_or_nan(
                            1 - int(r["stage1_correct"]) for r in switch_vu
                        ),
                        "preferentially_keeps_harder": bool(
                            remain_v
                            and switch_vu
                            and mean_or_nan(r["other_n_correct"] for r in remain_v)
                            < mean_or_nan(r["other_n_correct"] for r in switch_vu)
                        ),
                    }
                )
    return output


def error_catching(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    aliases = sorted({row["model_alias"] for row in rows})
    for model in aliases:
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                cell = [
                    row
                    for row in rows
                    if row["model_alias"] == model
                    and row["family"] == family
                    and row["score_condition"] == condition
                ]
                n_wrong = sum(1 for row in cell if not row["stage1_correct"])
                n_corr = sum(1 for row in cell if row["stage1_correct"])
                output.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "score_condition": condition,
                        "n": len(cell),
                        "verify_rate": mean_or_nan(row["verify"] for row in cell),
                        "n_wrong": n_wrong,
                        "n_correct": n_corr,
                        "frac_errors_verified": mean_or_nan(
                            row["verify"] for row in cell if not row["stage1_correct"]
                        ),
                        "frac_correct_verified": mean_or_nan(
                            row["verify"] for row in cell if row["stage1_correct"]
                        ),
                    }
                )
    return output


def _bucket_for_model(
    model: str,
    by_cond: Sequence[Mapping[str, Any]],
    switches: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    vis = [
        row
        for row in by_cond
        if row["model_alias"] == model
        and row["difficulty_bin"] == "hard_0to1"
        and row["score_condition"].startswith("displayed_")
    ]
    gaps = {}
    for row in vis:
        gaps[row["family"], row["score_condition"]] = float(row["hard_minus_easy_verify"])
    keep_harder = [
        row
        for row in switches
        if row["model_alias"] == model and row.get("preferentially_keeps_harder")
    ]
    n_switch = [row for row in switches if row["model_alias"] == model]
    high_gaps = [
        gaps[k]
        for k in gaps
        if k[1] == "displayed_0.99" and math.isfinite(gaps[k])
    ]
    low_gaps = [
        gaps[k]
        for k in gaps
        if k[1] == "displayed_0.70" and math.isfinite(gaps[k])
    ]
    if not high_gaps or not low_gaps:
        return "UNIDENTIFIABLE", "hard/easy gaps undefined in some visible cells"
    mean_high = float(np.mean(high_gaps))
    mean_low = float(np.mean(low_gaps))
    keep_frac = len(keep_harder) / len(n_switch) if n_switch else 0.0
    if mean_high <= 0.02 and mean_low > 0.10:
        return (
            "SCORE_ATTENUATES_DIFFICULTY_SENSITIVITY",
            f"hard−easy VERIFY gap {mean_low:.2f} at 0.70 vs {mean_high:.2f} at 0.99",
        )
    if mean_high > mean_low + 0.05 and keep_frac >= 0.5:
        return (
            "SCORE_SHARPENS_DIFFICULTY_PRIORITY",
            f"hard−easy gap rises {mean_low:.2f} → {mean_high:.2f}; switch-set keeps harder items in {len(keep_harder)}/{len(n_switch)} transitions",
        )
    if mean_high > 0.05 and keep_frac >= 0.5:
        return (
            "SCORE_SHIFTS_BUDGET_PRESERVES_DIFFICULTY_PRIORITY",
            f"hard items still preferentially verified at 0.99 (gap {mean_high:.2f}); switch-set keeps harder items in {len(keep_harder)}/{len(n_switch)} transitions",
        )
    if abs(mean_high - mean_low) < 0.08 and mean_high > 0.05:
        return (
            "SCORE_SHIFTS_BUDGET_PRESERVES_DIFFICULTY_PRIORITY",
            f"difficulty gap remains similar ({mean_low:.2f} → {mean_high:.2f}) while rates change",
        )
    return (
        "MIXED_OR_MODEL_SPECIFIC",
        f"hard−easy 0.70={mean_low:.2f}, 0.99={mean_high:.2f}; keep-harder transitions {len(keep_harder)}/{len(n_switch)}",
    )


def write_figures(marginal: Sequence[Mapping[str, Any]]) -> None:
    plt = _pyplot()
    fig_dir = LANE_C_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for model in GPT_CLAUDE:
        for family in STAKES_FAMILIES:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            xs = [0, 1, 2, 3]
            for condition, marker in (
                ("hidden", "o"),
                ("displayed_0.70", "s"),
                ("displayed_0.90", "^"),
                ("displayed_0.99", "D"),
            ):
                ys = []
                for n_corr in xs:
                    hits = [
                        row
                        for row in marginal
                        if row["model_alias"] == model
                        and row["family"] == family
                        and row["score_condition"] == condition
                        and int(row["other_n_correct"]) == n_corr
                    ]
                    ys.append(hits[0]["verify_rate"] if hits else float("nan"))
                ax.plot(xs, ys, marker=marker, label=condition)
            ax.set_xlabel("Other-model correct count (lower = harder)")
            ax.set_ylabel("VERIFY rate")
            ax.set_title(f"{MODEL_LABELS[model]} {family}")
            ax.set_xticks(xs)
            ax.legend(frameon=False)
            fig.tight_layout()
            stem = f"{model}_{family}_verify_vs_difficulty"
            fig.savefig(fig_dir / f"{stem}.pdf")
            fig.savefig(fig_dir / f"{stem}.png", dpi=140)
            plt.close(fig)


def run_lane_c(merged_gpt_claude: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    LANE_C_DIR.mkdir(parents=True, exist_ok=True)
    qids = sorted({row["question_id"] for row in merged_gpt_claude})
    feats = build_difficulty_rows(qids)
    feat_rows = []
    for qid, feat in feats.items():
        for model in GPT_CLAUDE:
            others = OTHER_BY_TARGET[model]
            other_n = int(sum(int(feat[CORRECT_KEY[alias]]) for alias in others))
            feat_rows.append(
                {
                    **feat,
                    "target_model": model,
                    "other_models": ",".join(others),
                    "other_n_correct": other_n,
                    "difficulty_bin": difficulty_bin(other_n),
                }
            )
    attached = attach_difficulty(merged_gpt_claude, feats)
    by_cond = difficulty_by_condition(attached)
    interactions = interaction_models(attached)
    curves = marginal_curves(attached)
    switches = switch_set_analysis(attached)
    catching = error_catching(attached)
    buckets = {}
    lines = ["# Lane C model decision buckets", "", "Exploratory. Not confirmatory.", ""]
    for model in GPT_CLAUDE:
        bucket, reason = _bucket_for_model(model, by_cond, switches)
        buckets[model] = {"bucket": bucket, "reason": reason}
        lines += [f"## {MODEL_LABELS[model]}", "", f"**{bucket}**", "", reason, ""]
    write_csv(LANE_C_DIR / "difficulty_features.csv", feat_rows)
    write_csv(LANE_C_DIR / "difficulty_by_condition.csv", by_cond)
    write_csv(LANE_C_DIR / "interaction_models.csv", interactions)
    write_csv(LANE_C_DIR / "marginal_curves.csv", curves)
    write_csv(LANE_C_DIR / "switch_set_analysis.csv", switches)
    write_csv(LANE_C_DIR / "error_catching_by_condition.csv", catching)
    (LANE_C_DIR / "model_decision_buckets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_figures(curves)
    return {
        "n_questions": len(qids),
        "buckets": buckets,
        "by_cond": by_cond,
        "interactions": interactions,
        "switches": switches,
        "catching": catching,
        "attached": attached,
    }
