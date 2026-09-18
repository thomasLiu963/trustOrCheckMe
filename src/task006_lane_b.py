"""Task 006 Lane B: exploratory matched-budget routing on existing q1/q2/hidden.

Zero API calls. Read-only on V2 and q2 sqlite.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .study1_analysis import MODEL_LABELS, _write_csv
from .study1_schemas import STUDY1_MODEL_ALIASES
from .task005_common import Q2_SQLITE, PRIMARY_FAMILY
from .task005_lane_a import (
    crossfit_isotonic,
    error_catch_from_weights,
    fractional_verify_weights,
    load_v2_answers,
    load_v2_confidence,
    load_v2_primary_verification,
)
from .task006_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANONICAL_HIDDEN,
    LANE_B_DIR,
    load_json_records,
)


BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)

ROUTER_RESOURCES = {
    "raw_q1": "one Stage-2 confidence report (q1)",
    "calibrated_q1": "one Stage-2 confidence report + grouped-CV isotonic fit (not an extra model call)",
    "raw_q2": "one independent Stage-2 re-elicitation (q2); conceptually one extra self-assessment call",
    "mean_q1_q2": "two confidence reports (q1 and q2)",
    "single_hidden_ai_L10": (
        "one verification-decision elicitation in the predeclared hidden "
        "AI-system L=10 cell; deployment-fairer than the aggregate"
    ),
    "hidden_aggregate": (
        "fraction VERIFY across historical hidden owner×L cells; multi-elicitation "
        "upper bound / diagnostic, not a one-call practical router"
    ),
    "combined_q1_q2_single_hidden": (
        "q1 + q2 + one hidden verification judgment; grouped-CV logistic, "
        "OOF predicted P(error); three elicitations"
    ),
}


def _load_q2() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for record in load_json_records(Q2_SQLITE, stage="confidence"):
        qid = str(record.get("question_id") or record.get("example_id"))
        alias = str(record.get("model_alias"))
        q2 = record.get("q2")
        if q2 is None:
            continue
        out[(qid, alias)] = float(q2)
    return out


def _single_hidden_map(
    v2_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    owner = CANONICAL_HIDDEN["decision_owner"]
    L = float(CANONICAL_HIDDEN["L"])
    for row in v2_rows:
        if row["decision_owner"] != owner:
            continue
        if float(row["L"]) != L:
            continue
        if row["confidence_visibility"] != "hidden":
            continue
        key = (str(row["question_id"]), str(row["model_alias"]))
        out[key] = int(str(row["parsed_action"]) == "VERIFY_FIRST")
    return out


def _hidden_frac_map(
    v2_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in v2_rows:
        if row["confidence_visibility"] != "hidden":
            continue
        if row["model_alias"] not in STUDY1_MODEL_ALIASES:
            continue
        grouped[(str(row["question_id"]), str(row["model_alias"]))].append(
            int(str(row["parsed_action"]) == "VERIFY_FIRST")
        )
    return {key: sum(vals) / len(vals) for key, vals in grouped.items() if vals}


def build_routing_table() -> list[dict[str, Any]]:
    answers = load_v2_answers()
    conf = load_v2_confidence()
    v2_rows = load_v2_primary_verification()
    q2 = _load_q2()
    single = _single_hidden_map(v2_rows)
    frac = _hidden_frac_map(v2_rows)
    table: list[dict[str, Any]] = []
    for alias in STUDY1_MODEL_ALIASES:
        qids = sorted(
            {
                qid
                for (qid, model) in answers
                if model == alias
            }
        )
        for qid in qids:
            ans = answers.get((qid, alias))
            c = conf.get((qid, alias))
            if ans is None or c is None:
                continue
            if (qid, alias) not in q2 or (qid, alias) not in single:
                continue
            table.append(
                {
                    "question_id": qid,
                    "model_alias": alias,
                    "stage1_correct": bool(ans["is_correct"]),
                    "q1": float(c["probability_correct"]),
                    "q2": float(q2[(qid, alias)]),
                    "single_hidden_verify": int(single[(qid, alias)]),
                    "hidden_verify_fraction": float(frac.get((qid, alias), float("nan"))),
                }
            )
    return table


def _oof_combined(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    y = np.array([int(not row["stage1_correct"]) for row in rows], dtype=int)
    qids = np.array([row["question_id"] for row in rows])
    X = np.column_stack(
        [
            [float(row["q1"]) for row in rows],
            [float(row["q2"]) for row in rows],
            [float(row["single_hidden_verify"]) for row in rows],
        ]
    )
    oof = np.full(len(rows), np.nan)
    gkf = GroupKFold(n_splits=5)
    dummy = np.zeros(len(rows))
    for train_idx, test_idx in gkf.split(dummy, y, groups=qids):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof


def _router_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    q1 = np.array([float(row["q1"]) for row in rows])
    q2 = np.array([float(row["q2"]) for row in rows])
    hid_s = np.array([float(row["single_hidden_verify"]) for row in rows])
    hid_f = np.array([float(row["hidden_verify_fraction"]) for row in rows])
    correct = np.array([int(row["stage1_correct"]) for row in rows], dtype=int)
    qids = np.array([row["question_id"] for row in rows])
    cal = crossfit_isotonic(q1, correct, qids, seed=BOOTSTRAP_SEED)
    combined = _oof_combined(rows)
    return {
        "raw_q1": q1,
        "calibrated_q1": cal,
        "raw_q2": q2,
        "mean_q1_q2": (q1 + q2) / 2.0,
        "single_hidden_ai_L10": 1.0 - hid_s,
        "hidden_aggregate": -hid_f,
        "combined_q1_q2_single_hidden": -combined,
    }


def _catch(score: np.ndarray, wrong: np.ndarray, n_verify: float) -> float:
    return error_catch_from_weights(fractional_verify_weights(score, n_verify), wrong)


def _bootstrap_catches(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, np.ndarray],
    budgets: Sequence[float],
) -> dict[tuple[str, float], tuple[float, float, float]]:
    n = len(rows)
    qids = [row["question_id"] for row in rows]
    by_q = {qid: i for i, qid in enumerate(qids)}
    q_list = list(by_q)
    rng = random.Random(BOOTSTRAP_SEED)
    wrong = np.array([not row["stage1_correct"] for row in rows])
    point: dict[tuple[str, float], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [q_list[rng.randrange(len(q_list))] for _ in q_list]
        idx = np.array([by_q[qid] for qid in draw])
        w = wrong[idx]
        for name, score in scores.items():
            s = score[idx]
            for frac in budgets:
                point[(name, frac)].append(_catch(s, w, frac * len(idx)))
    out: dict[tuple[str, float], tuple[float, float, float]] = {}
    for key, vals in point.items():
        finite = sorted(v for v in vals if math.isfinite(v))
        if not finite:
            out[key] = (float("nan"), float("nan"), float("nan"))
            continue
        mean = float(np.mean(finite))
        out[key] = (
            mean,
            finite[int(0.025 * (len(finite) - 1))],
            finite[int(0.975 * (len(finite) - 1))],
        )
        _ = n
    return out


def run_lane_b() -> dict[str, Any]:
    LANE_B_DIR.mkdir(parents=True, exist_ok=True)
    (LANE_B_DIR / "figures").mkdir(parents=True, exist_ok=True)
    table = build_routing_table()
    catch_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    resource_rows = [
        {
            "router": name,
            "conceptual_signals": desc,
            "deployment_fair": name
            not in {"hidden_aggregate", "combined_q1_q2_single_hidden"}
            or name == "combined_q1_q2_single_hidden",
            "note": (
                "combined uses three elicitations; hidden_aggregate uses many historical cells"
                if name == "hidden_aggregate"
                else ""
            ),
        }
        for name, desc in ROUTER_RESOURCES.items()
    ]
    resource_rows[5]["deployment_fair"] = False
    resource_rows[6]["deployment_fair"] = False
    # combined is 3-call, fairer than aggregate but not 1-call
    resource_rows[6]["deployment_fair"] = False

    summaries: dict[str, Any] = {}
    for alias in STUDY1_MODEL_ALIASES:
        rows = [row for row in table if row["model_alias"] == alias]
        if len(rows) != 500:
            summaries[alias] = {"n": len(rows), "unstable": True}
            continue
        scores = _router_scores(rows)
        wrong = np.array([not row["stage1_correct"] for row in rows])
        n_wrong = int(wrong.sum())
        boots = _bootstrap_catches(rows, scores, BUDGETS)
        raw_q1_at = {}
        for frac in BUDGETS:
            n_v = frac * len(rows)
            raw_catch = _catch(scores["raw_q1"], wrong, n_v)
            raw_q1_at[frac] = raw_catch
            for name, score in scores.items():
                point = _catch(score, wrong, n_v)
                mean, lo, hi = boots[(name, frac)]
                catch_rows.append(
                    {
                        "model_alias": alias,
                        "model_label": MODEL_LABELS[alias],
                        "router": name,
                        "budget_fraction": frac,
                        "n": len(rows),
                        "n_wrong": n_wrong,
                        "n_verify": n_v,
                        "errors_caught_fraction": point,
                        "errors_caught_count": point * n_wrong,
                        "boot_mean": mean,
                        "ci_lower": lo,
                        "ci_upper": hi,
                    }
                )
                if name != "raw_q1":
                    gain_rows.append(
                        {
                            "model_alias": alias,
                            "router": name,
                            "budget_fraction": frac,
                            "gain_vs_raw_q1": point - raw_catch,
                            "raw_q1_catch": raw_catch,
                            "router_catch": point,
                        }
                    )
        summaries[alias] = {
            "n": 500,
            "n_wrong": n_wrong,
            "catch_at_30_raw_q1": raw_q1_at[0.3],
            "catch_at_30_combined": _catch(scores["combined_q1_q2_single_hidden"], wrong, 0.3 * 500),
            "catch_at_30_single_hidden": _catch(scores["single_hidden_ai_L10"], wrong, 0.3 * 500),
            "catch_at_30_q2": _catch(scores["raw_q2"], wrong, 0.3 * 500),
            "catch_at_30_hidden_agg": _catch(scores["hidden_aggregate"], wrong, 0.3 * 500),
        }

    _write_csv(LANE_B_DIR / "error_catch_by_budget.csv", catch_rows)
    _write_csv(LANE_B_DIR / "pairwise_gains.csv", gain_rows)
    _write_csv(LANE_B_DIR / "resource_accounting.csv", resource_rows)
    (LANE_B_DIR / "router_definitions.md").write_text(
        "\n".join(
            [
                "# Router definitions",
                "",
                "Exploratory only. Same 500 historical V2-B questions. Not a frozen confirmatory router.",
                "",
                f"Canonical single hidden cell: `{json.dumps(CANONICAL_HIDDEN)}`",
                "",
                *[f"- **{name}:** {desc}" for name, desc in ROUTER_RESOURCES.items()],
                "",
                "Learned/calibrated routers use GroupKFold by question ID. Calibration and logistic fits never see test-question rows.",
                "Ranking uses Checkpoint A fractional inclusion at the cutoff (lowest rank-score first).",
                "Primary metric: fraction of actual Stage-1 wrong answers caught at a fixed verification budget.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        for alias in STUDY1_MODEL_ALIASES:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for name in ROUTER_RESOURCES:
                xs = []
                ys = []
                for row in catch_rows:
                    if row["model_alias"] == alias and row["router"] == name:
                        xs.append(row["budget_fraction"])
                        ys.append(row["errors_caught_fraction"])
                if xs:
                    ax.plot(xs, ys, marker="o", label=name.replace("_", " "))
            ax.set_xlabel("verification budget (fraction of 500)")
            ax.set_ylabel("fraction of errors caught")
            ax.set_title(f"{MODEL_LABELS[alias]} exploratory routing (V2-B, n=500)")
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(LANE_B_DIR / "figures" / f"catch_curve_{alias}.png", dpi=140)
            fig.savefig(LANE_B_DIR / "figures" / f"catch_curve_{alias}.pdf")
            plt.close(fig)
    except Exception:
        pass

    def _gain(alias: str, router: str, frac: float = 0.3) -> float:
        hits = [
            r
            for r in gain_rows
            if r["model_alias"] == alias
            and r["router"] == router
            and r["budget_fraction"] == frac
        ]
        return float(hits[0]["gain_vs_raw_q1"]) if hits else float("nan")

    report = [
        "# Lane B — exploratory matched-budget routing",
        "",
        "Zero API calls. Historical V2-B 500 questions × Study-1 GPT/Claude. Exploratory; same questions motivated the hypotheses.",
        "",
        "## Canonical single hidden router",
        "",
        f"Predeclared `{CANONICAL_HIDDEN}` without peeking at catch rates.",
        "",
        "## Error catch at 30% budget vs raw q1",
        "",
    ]
    for alias in STUDY1_MODEL_ALIASES:
        s = summaries.get(alias, {})
        report += [
            f"### {MODEL_LABELS[alias]}",
            "",
            f"- n={s.get('n')} questions, n_wrong={s.get('n_wrong')}",
            f"- raw q1 catch@30%: {s.get('catch_at_30_raw_q1')}",
            f"- q2 catch@30%: {s.get('catch_at_30_q2')} (gain {_gain(alias, 'raw_q2'):+.3f})",
            f"- single hidden AI L=10 catch@30%: {s.get('catch_at_30_single_hidden')} (gain {_gain(alias, 'single_hidden_ai_L10'):+.3f})",
            f"- hidden aggregate catch@30%: {s.get('catch_at_30_hidden_agg')} (gain {_gain(alias, 'hidden_aggregate'):+.3f}; not deployment-fair)",
            f"- combined q1+q2+single-hidden catch@30%: {s.get('catch_at_30_combined')} (gain {_gain(alias, 'combined_q1_q2_single_hidden'):+.3f})",
            "",
        ]
    (LANE_B_DIR / "routing_report.md").write_text("\n".join(report), encoding="utf-8")
    (LANE_B_DIR / "summary.json").write_text(
        json.dumps({"n_rows": len(table), "summaries": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"n_rows": len(table), "summaries": summaries, "catch_rows": catch_rows, "gain_rows": gain_rows}
