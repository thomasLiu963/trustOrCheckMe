"""Task 007 orchestrator: full qualitative 80 + Gemini/Grok 20 + decision packet."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .study1_analysis import MODEL_LABELS
from .study1_schemas import load_study1_config
from .task006_common import EXPECTED_REPEAT_IDS, QUAL_SQLITE, REPEAT_HASH
from .task006_lane_a import family_template_hash
from .task006_prompts import (
    format_displayed_probability,
    parse_qualitative_response,
)
from .task007_analyze import (
    classify_score_response,
    rate_table,
    saturation_table,
    score_response_table,
    split_pilot_vs_remaining,
    stakes_table,
)
from .task007_common import (
    CLAUDE_ENDPOINT,
    GEMINI_ENDPOINT,
    GEMINI_GROK,
    GPT_CLAUDE,
    GPT_ENDPOINT,
    GROK_ENDPOINT,
    LANE_A_CAP,
    LANE_A_DIR,
    LANE_A_EXPERIMENT,
    LANE_A_PROVIDER_CAP,
    LANE_A_RUN_ID,
    LANE_A_SQLITE,
    LANE_B_CAP,
    LANE_B_DIR,
    LANE_B_EXPERIMENT,
    LANE_B_PROVIDER_CAP,
    LANE_B_RUN_ID,
    LANE_B_SQLITE,
    LANE_C_DIR,
    LANE_D_DIR,
    PAPER_DIRECTION,
    PROMPT_HASH_MODERATE,
    PROMPT_HASH_STRONGER,
    PROMPT_VERSION,
    RETURN_DIR,
    TASK001_ID_HASH,
    TASK_ID,
    TOTAL_CAP,
    coerce_qual_row,
    load_csv,
    load_historical_aliases,
    protected_fingerprints,
    remaining_80_ids,
    sha256_file,
    write_csv,
)
from .task007_lane_c import (
    attach_difficulty,
    build_difficulty_rows,
    difficulty_by_condition,
    run_lane_c,
)
from .task007_lane_d import run_lane_d
from .task007_lane_e import choose_spine, write_lane_e
from .task007_run import (
    build_qual_calls,
    results_to_rows,
    run_paid_grid,
    validate_calls,
)


TASK006_RESULTS = (
    PROJECT_ROOT
    / "to_gpt"
    / "006_qualitative_pilot_parallel_prep"
    / "lane_A_qualitative_pilot"
    / "results.csv"
)
SRC_FILES = (
    "task007.py",
    "task007_common.py",
    "task007_run.py",
    "task007_analyze.py",
    "task007_lane_c.py",
    "task007_lane_d.py",
    "task007_lane_e.py",
)
ENDPOINT_BY_ALIAS = {
    "openai_gpt56_sol": GPT_ENDPOINT,
    "anthropic_sonnet5": CLAUDE_ENDPOINT,
    "google_gemini38_flash": GEMINI_ENDPOINT,
    "xai_grok420_nonreasoning": GROK_ENDPOINT,
}


def _pp(value: Any, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "NA"


def _delta(value: Any) -> str:
    try:
        return f"{100.0 * float(value):+.1f} pp"
    except (TypeError, ValueError):
        return "NA"


def _copy_scripts() -> None:
    src = PROJECT_ROOT / "src"
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        LANE_A_DIR: (
            "task007_run.py",
            "task007_analyze.py",
            "task007_common.py",
            "task006_prompts.py",
        ),
        LANE_B_DIR: (
            "task007_run.py",
            "task007_analyze.py",
            "task007_common.py",
            "task006_prompts.py",
        ),
        LANE_C_DIR: ("task007_lane_c.py", "task007_common.py", "task007_analyze.py"),
        LANE_D_DIR: ("task007_lane_d.py", "task007_common.py"),
    }
    for dest, names in mapping.items():
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(src / name, dest / name)
    shutil.copy2(src / "task007.py", RETURN_DIR / "task007.py")
    shutil.copy2(src / "task007_lane_e.py", RETURN_DIR / "task007_lane_e.py")


def _list_changed(root: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            paths.append(str(path.relative_to(PROJECT_ROOT)))
    for name in SRC_FILES:
        rel = f"src/{name}"
        if rel not in paths:
            paths.append(rel)
    paths.append("docs/D1_DECISION_LOG.md")
    paths.append(f"from_gpt/{TASK_ID}.md")
    for sqlite in (LANE_A_SQLITE, LANE_B_SQLITE):
        if sqlite.exists():
            paths.append(
                str(sqlite.relative_to(PROJECT_ROOT))
                + "  (gitignored new sqlite; not V2/Study1/q2/006)"
            )
    return paths


def _env_present(name: str) -> bool:
    return bool(os.getenv(name))


def _rate_line(rate_rows: Sequence[Mapping[str, Any]], model: str, family: str) -> str:
    bits = []
    for cond in ("hidden", "displayed_0.70", "displayed_0.90", "displayed_0.99"):
        hits = [
            row
            for row in rate_rows
            if row["model_alias"] == model
            and row["family"] == family
            and row["score_condition"] == cond
        ]
        if hits:
            bits.append(
                f"{cond} {_pp(hits[0]['verify_rate'])} "
                f"({hits[0]['n_verify']}/{hits[0]['n']}; "
                f"CI {_pp(hits[0]['ci_lower'])}–{_pp(hits[0]['ci_upper'])})"
            )
    return "; ".join(bits) if bits else "not available"


def _score_line(score_rows: Sequence[Mapping[str, Any]], model: str, family: str) -> str:
    rows = [
        row
        for row in score_rows
        if row["model_alias"] == model and row["family"] == family
    ]
    if not rows:
        return "not available"
    parts = []
    for row in rows:
        parts.append(
            f"{row['from_score']}→{row['to_score']} {_delta(row['mean_delta_verify'])} "
            f"[{_delta(row['ci_lower'])}, {_delta(row['ci_upper'])}] "
            f"(dec {row['n_decreased']}/inc {row['n_increased']}/same {row['n_unchanged']})"
        )
    return "; ".join(parts)


def preflight_lane_b_model(alias: str) -> dict[str, Any]:
    errors: list[str] = []
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    if alias not in models.models:
        return {"ok": False, "model_alias": alias, "errors": [f"{alias} missing from models.yaml"]}
    spec = models.models[alias]
    expected = ENDPOINT_BY_ALIAS[alias]
    if spec.api_model != expected:
        errors.append(f"endpoint {spec.api_model} != {expected}")
    env_needed = {
        "google_gemini38_flash": ("GOOGLE_CLOUD_PROJECT",),
        "xai_grok420_nonreasoning": ("XAI_API_KEY",),
    }[alias]
    for name in env_needed:
        if not _env_present(name):
            errors.append(f"missing env {name}")
    try:
        create_adapter(alias, spec)
    except Exception as error:
        errors.append(f"adapter: {error}")
    try:
        hist = load_historical_aliases(EXPECTED_REPEAT_IDS, [alias])
        missing = [qid for qid in EXPECTED_REPEAT_IDS if (qid, alias) not in hist]
        if missing:
            errors.append(f"missing Stage-1 for {missing[:3]}")
        else:
            empty = [
                qid
                for qid in EXPECTED_REPEAT_IDS
                if not hist[(qid, alias)].get("frozen_answer")
            ]
            if empty:
                errors.append(f"empty frozen answer for {empty[:3]}")
    except Exception as error:
        errors.append(f"historical Stage-1: {error}")
    try:
        parse_qualitative_response('{"action":"VERIFY_FIRST"}')
        parse_qualitative_response('{"action":"USE_UNVERIFIED"}')
    except Exception as error:
        errors.append(f"parser: {error}")
    return {
        "ok": not errors,
        "model_alias": alias,
        "endpoint": spec.api_model,
        "provider": spec.provider,
        "errors": errors,
        "class": "BLOCKED" if errors else "READY",
    }


def build_preflight() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    selected, remaining, remaining_hash, selected_hash, repeat_hash = remaining_80_ids()
    moderate = family_template_hash("moderate")
    stronger = family_template_hash("stronger")
    lane_a_calls = build_qual_calls(
        question_ids=remaining,
        aliases=GPT_CLAUDE,
        sample_hash=remaining_hash,
        experiment_version=LANE_A_EXPERIMENT,
    )
    lane_a_val = validate_calls(
        lane_a_calls,
        expected_n=LANE_A_CAP,
        expected_ids=remaining,
        expected_hash=remaining_hash,
        aliases=GPT_CLAUDE,
        endpoints={alias: ENDPOINT_BY_ALIAS[alias] for alias in GPT_CLAUDE},
    )
    lane_b_status = {alias: preflight_lane_b_model(alias) for alias in GEMINI_GROK}
    ready_b = [alias for alias, row in lane_b_status.items() if row["ok"]]
    lane_b_calls = []
    lane_b_val: dict[str, Any] = {"ok": True, "n_calls": 0, "skipped": True}
    if ready_b:
        lane_b_calls = build_qual_calls(
            question_ids=list(EXPECTED_REPEAT_IDS),
            aliases=ready_b,
            sample_hash=REPEAT_HASH,
            experiment_version=LANE_B_EXPERIMENT,
        )
        lane_b_val = validate_calls(
            lane_b_calls,
            expected_n=160 * len(ready_b),
            expected_ids=list(EXPECTED_REPEAT_IDS),
            expected_hash=REPEAT_HASH,
            aliases=ready_b,
            endpoints={alias: ENDPOINT_BY_ALIAS[alias] for alias in ready_b},
        )
    if not TASK006_RESULTS.exists():
        lane_a_val = dict(lane_a_val)
        lane_a_val["ok"] = False
        lane_a_val["errors"] = list(lane_a_val.get("errors") or []) + [
            f"missing Task 006 results at {TASK006_RESULTS}"
        ]
    summary = {
        "task": TASK_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash_moderate": moderate,
        "prompt_hash_stronger": stronger,
        "frozen_moderate_ok": moderate == PROMPT_HASH_MODERATE,
        "frozen_stronger_ok": stronger == PROMPT_HASH_STRONGER,
        "task006_20_ids": list(EXPECTED_REPEAT_IDS),
        "task006_20_hash": repeat_hash,
        "study1_100_hash": selected_hash,
        "study1_100_n": len(selected),
        "remaining_80_ids": remaining,
        "remaining_80_hash": remaining_hash,
        "endpoints": dict(ENDPOINT_BY_ALIAS),
        "lane_a_cap": LANE_A_CAP,
        "lane_b_cap": LANE_B_CAP,
        "total_cap": TOTAL_CAP,
        "reuse_task006_gpt_claude_20": True,
        "paperDirection_untouched": True,
        "paperDirection_sha256": sha256_file(PAPER_DIRECTION),
        "lane_a_validation": lane_a_val,
        "lane_b_model_status": lane_b_status,
        "lane_b_ready_aliases": ready_b,
        "lane_b_validation": lane_b_val,
        "visible_tokens": {
            "0.70": format_displayed_probability(0.70),
            "0.90": format_displayed_probability(0.90),
            "0.99": format_displayed_probability(0.99),
        },
        "will_not_modify_paperDirection": True,
        "ok": bool(lane_a_val.get("ok")) and (not ready_b or bool(lane_b_val.get("ok"))),
    }
    print("=== TASK 007 PREFLIGHT ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return {
        **summary,
        "selected": selected,
        "remaining": remaining,
        "lane_a_calls": lane_a_calls,
        "lane_b_calls": lane_b_calls,
    }


def write_validation(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def analyze_lane_a(
    new_rows: Sequence[Mapping[str, Any]],
    remaining: Sequence[str],
) -> dict[str, Any]:
    LANE_A_DIR.mkdir(parents=True, exist_ok=True)
    old = [coerce_qual_row(row, source="task006") for row in load_csv(TASK006_RESULTS)]
    if len(old) != 320:
        raise RuntimeError(f"Task 006 results.csv has {len(old)} rows, expected 320")
    merged = list(old) + list(new_rows)
    keys = {
        (row["question_id"], row["model_alias"], row["family"], row["score_condition"])
        for row in merged
    }
    if len(merged) != 1600 or len(keys) != 1600:
        raise RuntimeError(
            f"merged qualitative cells {len(merged)} unique {len(keys)} != 1600"
        )
    rates = rate_table(merged, GPT_CLAUDE)
    scores = score_response_table(merged, GPT_CLAUDE)
    stakes = stakes_table(merged, GPT_CLAUDE)
    split = split_pilot_vs_remaining(merged, remaining, EXPECTED_REPEAT_IDS, GPT_CLAUDE)
    write_csv(LANE_A_DIR / "results_new80.csv", new_rows)
    write_csv(LANE_A_DIR / "results_merged100.csv", merged)
    write_csv(LANE_A_DIR / "score_response.csv", scores)
    write_csv(LANE_A_DIR / "stakes_effects.csv", stakes)
    write_csv(LANE_A_DIR / "pilot_vs_remaining80.csv", split)
    return {
        "merged": merged,
        "new_rows": new_rows,
        "rate_rows": rates,
        "score_rows": scores,
        "stakes_rows": stakes,
        "split_rows": split,
        "n_merged": len(merged),
        "n_new": len(new_rows),
        "n_old": len(old),
    }


def analyze_lane_b(table: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> dict[str, Any]:
    LANE_B_DIR.mkdir(parents=True, exist_ok=True)
    rates = rate_table(table, aliases)
    scores = score_response_table(table, aliases)
    stakes = stakes_table(table, aliases)
    sat = saturation_table(table, aliases)
    classes = {alias: classify_score_response(scores, alias) for alias in GEMINI_GROK}
    for alias in GEMINI_GROK:
        if alias not in aliases:
            classes[alias] = {
                "model_alias": alias,
                "class": "BLOCKED",
                "max_abs_070_to_099": float("nan"),
                "reason": "model failed Lane B preflight; no substitution",
            }
    feat = build_difficulty_rows(list(EXPECTED_REPEAT_IDS)) if table else {}
    attached = attach_difficulty(table, feat) if table else []
    desc = difficulty_by_condition(attached) if attached else []
    write_csv(LANE_B_DIR / "results.csv", table)
    write_csv(LANE_B_DIR / "score_response.csv", scores)
    write_csv(LANE_B_DIR / "saturation.csv", sat)
    write_csv(LANE_B_DIR / "difficulty_descriptive.csv", desc)
    write_csv(LANE_B_DIR / "stakes_effects.csv", stakes)
    write_csv(LANE_B_DIR / "rate_table.csv", rates)
    return {
        "table": table,
        "rate_rows": rates,
        "score_rows": scores,
        "sat_rows": sat,
        "classes": classes,
        "n": len(table),
    }


def decide_hold(
    lane_a: Mapping[str, Any] | None,
    lane_c: Mapping[str, Any] | None,
    validation_ok: bool,
) -> str | None:
    if not validation_ok:
        return "Major data validation issue: Lane A preflight or merge failed."
    if not lane_a or int(lane_a.get("n_merged") or 0) != 1600:
        return "Merged GPT/Claude qualitative grid is not 1,600 cells."
    collapse = []
    for row in lane_a.get("split_rows") or []:
        if row.get("kind") != "direction_replication":
            continue
        if row["model_alias"] != "openai_gpt56_sol":
            continue
        p = float(row.get("pilot20_0.70_to_0.99") or 0)
        r = float(row.get("remaining80_0.70_to_0.99") or 0)
        if not row.get("same_sign") and abs(p) >= 0.20 and abs(r) < 0.05:
            collapse.append(row["family"])
    if len(collapse) == 2:
        return (
            "Full-100 qualitative GPT 0.70→0.99 effect collapsed on the remaining 80 "
            f"in both families ({collapse})."
        )
    if lane_c:
        buckets = {
            spec.get("bucket")
            for spec in (lane_c.get("buckets") or {}).values()
        }
        if buckets == {"UNIDENTIFIABLE"}:
            return "Lane C difficulty pattern is UNIDENTIFIABLE for both GPT and Claude."
    return None


def write_root_report(summary: Mapping[str, Any]) -> None:
    a = summary.get("lane_a") or {}
    b = summary.get("lane_b") or {}
    c = summary.get("lane_c") or {}
    d = summary.get("lane_d") or {}
    fp = summary.get("fingerprints") or {}
    pre = summary.get("preflight") or {}
    spine_id, spine_note = choose_spine(summary)
    hold = summary.get("hold_revision")
    gpt_b = ((c.get("buckets") or {}).get("openai_gpt56_sol") or {})
    claude_b = ((c.get("buckets") or {}).get("anthropic_sonnet5") or {})
    classes = b.get("classes") or summary.get("lane_b_classes") or {}
    lines = [
        "# Task 007 — Full Qualitative Expansion, Four-Model Pilot, and Paper-Direction Decision Packet",
        "",
        "Exploratory only. Not confirmatory. `paperDirection.txt` was not modified.",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
        "---",
        "",
        "## 1. Plain-English executive bottom line",
        "",
    ]
    if hold:
        lines += [
            f"**HOLD paperDirection revision:** {hold}",
            "",
        ]
    else:
        lines += [
            f"On the full exploratory 100-question GPT/Claude qualitative grid, displayed confidence still changes verification without L/C arithmetic. Lane C classifies GPT as **{gpt_b.get('bucket', 'UNASSIGNED')}** and Claude as **{claude_b.get('bucket', 'UNASSIGNED')}**. Best-supported paper spine: **Spine {spine_id}** ({spine_note}). Gemini/Grok 20-q classes: "
            + "; ".join(
                f"{MODEL_LABELS.get(alias, alias)}={(classes.get(alias) or {}).get('class', 'BLOCKED')}"
                for alias in GEMINI_GROK
            )
            + ".",
            "",
            "Do not treat this as confirmatory. Do not call cross-model difficulty a production router feature. Do not infer hidden-state suppression.",
            "",
        ]
    paid_a = int(summary.get("lane_a_new_calls") or 0)
    paid_b = int(summary.get("lane_b_new_calls") or 0)
    lines += [
        "## 2. Exact paid scientific calls",
        "",
        f"- Lane A GPT/Claude remaining-80: **{paid_a}** / {LANE_A_CAP} (sqlite `{LANE_A_SQLITE}`)",
        f"- Lane B Gemini/Grok frozen-20: **{paid_b}** / up to {LANE_B_CAP} (sqlite `{LANE_B_SQLITE}`)",
        f"- Total new scientific calls: **{paid_a + paid_b}** / max {TOTAL_CAP}",
        f"- Task 006 GPT/Claude 20-q cells reused: **320** (not rerun)",
        f"- Merged GPT/Claude qualitative cells: **{a.get('n_merged', 0)}** / 1,600",
        f"- Lane A provider attempts: {summary.get('lane_a_provider_attempts', 0)}",
        f"- Lane B provider attempts: {summary.get('lane_b_provider_attempts', 0)}",
        "- Lanes C/D/E API calls: **0**",
        f"- Confirmatory study launched: **no**",
        "",
        "## 3. GPT full-100 qualitative replication",
        "",
        f"- Moderate: {_rate_line(a.get('rate_rows') or [], 'openai_gpt56_sol', 'moderate')}",
        f"- Stronger: {_rate_line(a.get('rate_rows') or [], 'openai_gpt56_sol', 'stronger')}",
        f"- Paired deltas moderate: {_score_line(a.get('score_rows') or [], 'openai_gpt56_sol', 'moderate')}",
        f"- Paired deltas stronger: {_score_line(a.get('score_rows') or [], 'openai_gpt56_sol', 'stronger')}",
        "",
        "## 4. Claude full-100 qualitative replication",
        "",
        f"- Moderate: {_rate_line(a.get('rate_rows') or [], 'anthropic_sonnet5', 'moderate')}",
        f"- Stronger: {_rate_line(a.get('rate_rows') or [], 'anthropic_sonnet5', 'stronger')}",
        f"- Paired deltas moderate: {_score_line(a.get('score_rows') or [], 'anthropic_sonnet5', 'moderate')}",
        f"- Paired deltas stronger: {_score_line(a.get('score_rows') or [], 'anthropic_sonnet5', 'stronger')}",
        "",
        "## 5. Gemini/Grok 20-question qualitative pilot",
        "",
    ]
    for alias in GEMINI_GROK:
        spec = classes.get(alias) or {"class": "BLOCKED", "reason": "not run"}
        lines.append(
            f"- **{MODEL_LABELS.get(alias, alias)}** `{ENDPOINT_BY_ALIAS[alias]}`: "
            f"**{spec.get('class')}**. {spec.get('reason', '')}"
        )
        if alias in (b.get("aliases") or []):
            lines.append(f"  - Moderate: {_rate_line(b.get('rate_rows') or [], alias, 'moderate')}")
            lines.append(f"  - Stronger: {_rate_line(b.get('rate_rows') or [], alias, 'stronger')}")
            lines.append(
                f"  - Paired moderate: {_score_line(b.get('score_rows') or [], alias, 'moderate')}"
            )
    lines += [
        "",
        "n=20; no paper-ready claims. No q2 on Gemini/Grok.",
        "",
        "## 6. Does score conditioning preserve, attenuate, or sharpen empirical difficulty prioritization?",
        "",
        f"- GPT: **{gpt_b.get('bucket', 'UNASSIGNED')}**. {gpt_b.get('reason', '')}",
        f"- Claude: **{claude_b.get('bucket', 'UNASSIGNED')}**. {claude_b.get('reason', '')}",
        "- Difficulty is leave-one-target-out other-model correctness, not a production feature.",
        "",
        "## 7. Switch-set result",
        "",
    ]
    for row in c.get("switches") or []:
        lines.append(
            f"- {MODEL_LABELS.get(row['model_alias'], row['model_alias'])} {row['family']} "
            f"{row['from_score']}→{row['to_score']}: remain-V {row['n_remain_verify']}, "
            f"V→U {row['n_switch_verify_to_use']}, remain-U {row['n_remain_use']}, "
            f"U→V {row['n_switch_use_to_verify']}; mean other-correct remain-V "
            f"{row['mean_other_n_correct_remain_verify']} vs V→U "
            f"{row['mean_other_n_correct_switch_v_to_u']}; keeps-harder="
            f"{row['preferentially_keeps_harder']}"
        )
    if not c.get("switches"):
        lines.append("- Not available.")
    lines += [
        "",
        "## 8. Error-catching consequences",
        "",
        "Within-condition VERIFY rate vs fraction of target errors verified. Not a matched-budget claim.",
        "",
    ]
    for row in c.get("catching") or []:
        if row["score_condition"] not in {"displayed_0.70", "displayed_0.99"}:
            continue
        lines.append(
            f"- {MODEL_LABELS.get(row['model_alias'], row['model_alias'])} {row['family']} "
            f"{row['score_condition']}: VERIFY {_pp(row['verify_rate'])}; "
            f"errors caught {_pp(row['frac_errors_verified'])}; "
            f"corrects verified {_pp(row['frac_correct_verified'])}"
        )
    lines += [
        "",
        "## 9. Model-specific decision buckets",
        "",
        f"- GPT: **{gpt_b.get('bucket', 'UNASSIGNED')}**",
        f"- Claude: **{claude_b.get('bucket', 'UNASSIGNED')}**",
        f"- Gemini 20-q: **{(classes.get('google_gemini38_flash') or {}).get('class', 'BLOCKED')}**",
        f"- Grok 20-q: **{(classes.get('xai_grok420_nonreasoning') or {}).get('class', 'BLOCKED')}**",
        "",
        "## 10. Prospective four-model study readiness",
        "",
        f"- Unseen pool hash `{d.get('pool_hash')}`; D1/D2/D3 cell counts written; no confirmatory sample frozen.",
        f"- Design matrix rows: {d.get('n_design_rows')}. Routing plan is scaffolding only.",
        "",
        "## 11. Best-supported current paper spine",
        "",
        f"**Spine {spine_id}.** {spine_note}",
        "",
        "## 12. Which prior candidate story was weakened/killed",
        "",
        "- Hidden VERIFY as leftover internal correctness knowledge after difficulty control was already weakened by Tasks 005B/005C; Task 007 does not revive it.",
        "- 'A Score Is Not the State' as a mechanism headline is not supported. Score is a causal control input; the residual is mostly empirical difficulty plus a small leftover at most.",
        "- Treating the multi-cell hidden aggregate as a fair one-call production router remains forbidden.",
        "",
        "## 13. What is now ready to go into paperDirection.txt",
        "",
        "Nothing until GPT reviews `paper_direction_decision_packet.md`. The packet contains KEEP/REWRITE/DELETE/ADD recommendations only.",
        "",
        "## 14. What still needs fresh confirmation",
        "",
        "- A leakage-safe unseen sample (not these 100 development questions).",
        "- Fair matched-budget routing with predeclared baselines.",
        "- Four-model Stage-1/q1/qualitative pipeline on the same items.",
        "- Gemini/Grok beyond n=20.",
        "",
        "## 15. Exact cost / runtime / retries",
        "",
        f"- Lane A wall-clock seconds: {summary.get('lane_a_wall', 0)}",
        f"- Lane B wall-clock seconds: {summary.get('lane_b_wall', 0)}",
        f"- Lane A estimated USD: {summary.get('lane_a_usd', 0)}",
        f"- Lane B estimated USD: {summary.get('lane_b_usd', 0)}",
        f"- Lane A this-process new/reused: {summary.get('lane_a_this_process_new', 0)}/{summary.get('lane_a_reused', 0)}",
        f"- Lane B this-process new/reused: {summary.get('lane_b_this_process_new', 0)}/{summary.get('lane_b_reused', 0)}",
        f"- Lane A failures: {summary.get('lane_a_n_fail', 0)}",
        f"- Lane B failures: {summary.get('lane_b_n_fail', 0)}",
        f"- V2 sha256: `{((fp.get('v2') or {}).get('sha256'))}`",
        f"- Study 1 sha256: `{((fp.get('study1') or {}).get('sha256'))}`",
        f"- q2 sha256: `{((fp.get('q2') or {}).get('sha256'))}`",
        f"- Task 006 sqlite sha256: `{((fp.get('task006_sqlite') or {}).get('sha256'))}`",
        f"- paperDirection sha256: `{((fp.get('paperDirection') or {}).get('sha256'))}`",
        f"- Prompt hashes: moderate `{pre.get('prompt_hash_moderate')}`; stronger `{pre.get('prompt_hash_stronger')}`",
        "",
        "## 16. READY_FOR_GPT_REVIEW",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
    ]
    (RETURN_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_support_files(summary: Mapping[str, Any]) -> None:
    cost = [
        "# Task 007 cost summary",
        "",
        f"- Lane A scientific: {summary.get('lane_a_new_calls', 0)} / {LANE_A_CAP}",
        f"- Lane B scientific: {summary.get('lane_b_new_calls', 0)} / up to {LANE_B_CAP}",
        f"- Total scientific: {int(summary.get('lane_a_new_calls') or 0) + int(summary.get('lane_b_new_calls') or 0)} / {TOTAL_CAP}",
        f"- Lane A provider attempts: {summary.get('lane_a_provider_attempts', 0)} / {LANE_A_PROVIDER_CAP}",
        f"- Lane B provider attempts: {summary.get('lane_b_provider_attempts', 0)} / {LANE_B_PROVIDER_CAP}",
        f"- Lane A estimated USD: {summary.get('lane_a_usd', 0)}",
        f"- Lane B estimated USD: {summary.get('lane_b_usd', 0)}",
        f"- Lane A this-process new/reused: {summary.get('lane_a_this_process_new', 0)}/{summary.get('lane_a_reused', 0)}",
        f"- Lane B this-process new/reused: {summary.get('lane_b_this_process_new', 0)}/{summary.get('lane_b_reused', 0)}",
        "- Lanes C/D/E: $0",
        "- Task 006 320 GPT/Claude cells: reused, $0 this task",
        "",
    ]
    (RETURN_DIR / "cost_summary.md").write_text("\n".join(cost) + "\n", encoding="utf-8")
    log = [
        "# Task 007 parallel execution log",
        "",
        f"- Started: {summary.get('started')}",
        f"- Finished: {summary.get('finished')}",
        "- Isolation: Lane A sqlite `results/study1_qualitative_full80/qualitative_full80.sqlite3`; Lane B sqlite `results/study1_qualitative_gemini_grok20/qualitative_gemini_grok20.sqlite3`",
        "- Did not write V2, Study 1, q2, Task 006 sqlite, Tasks 004–006 outputs, or paperDirection.txt",
        "- Lane A and Lane B ran concurrently when both were authorized",
        f"- Remaining 80 hash: `{((summary.get('preflight') or {}).get('remaining_80_hash'))}`",
        f"- Lane B ready aliases: {(summary.get('preflight') or {}).get('lane_b_ready_aliases')}",
        f"- Hold revision: {summary.get('hold_revision')}",
        "",
    ]
    (RETURN_DIR / "parallel_execution_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    (RETURN_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": TASK_ID,
                "started": summary.get("started"),
                "finished": summary.get("finished"),
                "lane_a_cap": LANE_A_CAP,
                "lane_b_cap": LANE_B_CAP,
                "lane_a_new_calls": summary.get("lane_a_new_calls"),
                "lane_b_new_calls": summary.get("lane_b_new_calls"),
                "fingerprints": summary.get("fingerprints"),
                "remaining_80_hash": (summary.get("preflight") or {}).get("remaining_80_hash"),
                "hold_revision": summary.get("hold_revision"),
                "paperDirection_modified": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (RETURN_DIR / "changed_files.txt").write_text(
        "\n".join(_list_changed(RETURN_DIR)) + "\n", encoding="utf-8"
    )


def append_decision_log(summary: Mapping[str, Any]) -> None:
    path = PROJECT_ROOT / "docs" / "D1_DECISION_LOG.md"
    spine_id, spine_note = choose_spine(summary)
    text = path.read_text(encoding="utf-8")
    entry = f"""
---

## 2026-09-18 — Task 007 full qualitative 100 + four-model pilot + paper-direction packet

**Date:** 2026-09-18

**Previous state:**  
Task 006 qualitative 20-q GPT/Claude pilot gated A-GO-FULL-QUALITATIVE. Tasks 005B/005C showed that much of the natural-verification residual is empirical item difficulty. `paperDirection.txt` was still unmodified.

**Evidence:**  
Numbered task `from_gpt/007_full_qualitative_four_model_decision_packet.md`.

**Decision:**

1. Lane A completes the remaining 80 Study-1 questions for GPT/Claude with the frozen Task-006 qualitative prompts (1,280 new cells). Task-006's original 320 cells are reused, not rerun. Merged grid = 1,600 exploratory cells. New sqlite `results/study1_qualitative_full80/qualitative_full80.sqlite3`.
2. Lane B runs Gemini/Grok on the frozen 20-question subset with the same prompts (up to 320 cells) if adapters/historical Stage-1 pass. Separate sqlite `results/study1_qualitative_gemini_grok20/qualitative_gemini_grok20.sqlite3`. No q2. No endpoint substitution.
3. Lane C tests whether displayed score merely shifts the verification budget or changes difficulty-sensitive prioritization. Cross-model other-correct is an evaluation proxy, not a production feature.
4. Lane D prepares, and does not run, a prospective four-model pipeline and D1/D2/D3 cost matrix on the Task-006 unseen pool.
5. Lane E writes a paper-direction decision packet. **Do not edit `paperDirection.txt` in this task.**
6. Do not launch a confirmatory study. Do not infer hidden-state suppression.

**Lane C buckets / spine:** GPT/Claude buckets and recommended Spine {spine_id} ({spine_note}) are in `to_gpt/007_full_qualitative_four_model_decision_packet/`. Hold revision: {summary.get('hold_revision') or 'no'}.

**Affected study already frozen?**  
Historical V2 remains HISTORICAL-FROZEN. Study 1 / Tasks 003–006 remain exploratory and are not overwritten.

**Exploratory vs confirmatory relative to this decision:**  
All Task 007 outputs are EXPLORATORY.

**Current next engineering task:**  
Return `to_gpt/007_full_qualitative_four_model_decision_packet/` for GPT review. Do not edit `paperDirection.txt` until GPT accepts the packet. Do not launch the confirmatory study.

**Status:**  
Task 007 = AUTHORIZED / EXPLORATORY.  
Confirmatory study = NOT AUTHORIZED.  
paperDirection.txt = UNMODIFIED.
"""
    if "Task 007 full qualitative 100" not in text:
        path.write_text(text.rstrip() + "\n" + entry + "\n", encoding="utf-8")


def _call_manifest(calls: Sequence[Any], source: str) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "question_id": row.question_id,
            "model_alias": row.model_alias,
            "model_endpoint": row.model_endpoint,
            "family": row.family,
            "score_condition": row.score_condition,
            "displayed_confidence": row.displayed_confidence,
            "request_key": row.request_key,
            "prompt_hash": row.prompt_hash,
            "prompt_version": PROMPT_VERSION,
        }
        for row in calls
    ]


def _validation_md(
    *,
    title: str,
    preflight: Mapping[str, Any],
    extra: Sequence[str],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- prompt_version: `{PROMPT_VERSION}`",
        f"- moderate hash: `{preflight.get('prompt_hash_moderate')}` (frozen match={preflight.get('frozen_moderate_ok')})",
        f"- stronger hash: `{preflight.get('prompt_hash_stronger')}` (frozen match={preflight.get('frozen_stronger_ok')})",
        f"- Task-006 20 hash: `{preflight.get('task006_20_hash')}`",
        f"- Study-1 100 hash: `{preflight.get('study1_100_hash')}`",
        f"- remaining 80 hash: `{preflight.get('remaining_80_hash')}`",
        f"- visible tokens: `{preflight.get('visible_tokens')}`",
        f"- reuse Task-006 GPT/Claude 20: {preflight.get('reuse_task006_gpt_claude_20')}",
        f"- paperDirection untouched: {preflight.get('paperDirection_untouched')}",
        f"- paperDirection sha256: `{preflight.get('paperDirection_sha256')}`",
        "",
    ]
    lines.extend(extra)
    lines.append("")
    return "\n".join(lines)


async def _run_paid(preflight: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    lane_a_task = run_paid_grid(
        calls=preflight["lane_a_calls"],
        aliases=GPT_CLAUDE,
        sqlite_path=LANE_A_SQLITE,
        experiment_version=LANE_A_EXPERIMENT,
        run_id=LANE_A_RUN_ID,
        scientific_cap=LANE_A_CAP,
        provider_cap=LANE_A_PROVIDER_CAP,
    )
    ready_b = list(preflight.get("lane_b_ready_aliases") or [])
    if ready_b:
        lane_b_task = run_paid_grid(
            calls=preflight["lane_b_calls"],
            aliases=ready_b,
            sqlite_path=LANE_B_SQLITE,
            experiment_version=LANE_B_EXPERIMENT,
            run_id=LANE_B_RUN_ID,
            scientific_cap=len(preflight["lane_b_calls"]),
            provider_cap=LANE_B_PROVIDER_CAP,
            concurrency=1,
            max_transient_retries=5,
            backoff_base_seconds=3.0,
            backoff_max_seconds=45.0,
        )
        lane_a, lane_b = await asyncio.gather(lane_a_task, lane_b_task)
        return lane_a, lane_b
    lane_a = await lane_a_task
    return lane_a, None


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    started = datetime.now(UTC).isoformat()
    before = protected_fingerprints()
    preflight = build_preflight()
    if not preflight["lane_a_validation"].get("ok"):
        raise SystemExit(
            "Task 007 Lane A preflight failed: "
            + json.dumps(preflight["lane_a_validation"].get("errors"))
        )
    remaining = preflight["remaining"]
    LANE_A_DIR.mkdir(parents=True, exist_ok=True)
    LANE_B_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(LANE_A_DIR / "call_manifest.csv", _call_manifest(preflight["lane_a_calls"], "lane_a"))
    write_csv(LANE_B_DIR / "call_manifest.csv", _call_manifest(preflight["lane_b_calls"], "lane_b"))
    write_validation(
        LANE_A_DIR / "validation.md",
        _validation_md(
            title="Lane A validation",
            preflight=preflight,
            extra=[
                f"- planned calls: {len(preflight['lane_a_calls'])}",
                f"- remaining 80 IDs: {remaining}",
                f"- endpoints: GPT `{GPT_ENDPOINT}` reasoning.effort=none; Claude `{CLAUDE_ENDPOINT}` thinking disabled",
                f"- preflight ok: {preflight['lane_a_validation'].get('ok')}",
            ],
        ),
    )
    write_validation(
        LANE_B_DIR / "validation.md",
        _validation_md(
            title="Lane B validation",
            preflight=preflight,
            extra=[
                f"- model status: {json.dumps(preflight.get('lane_b_model_status'))}",
                f"- ready aliases: {preflight.get('lane_b_ready_aliases')}",
                f"- planned calls: {len(preflight['lane_b_calls'])}",
                "- no q2; no endpoint substitution",
            ],
        ),
    )
    lane_d = run_lane_d()
    lane_a_run, lane_b_run = asyncio.run(_run_paid(preflight))
    if not lane_a_run.get("ok"):
        new_rows, failures = results_to_rows(lane_a_run, source="task007a")
        write_csv(LANE_A_DIR / "failures.csv", failures)
        write_csv(LANE_A_DIR / "results_new80.csv", new_rows)
        raise SystemExit(
            f"Task 007 Lane A incomplete: successful={lane_a_run.get('successful')} "
            f"failed={lane_a_run.get('failed')}"
        )
    new_rows, a_fail = results_to_rows(lane_a_run, source="task007a")
    write_csv(LANE_A_DIR / "failures.csv", a_fail)
    lane_a = analyze_lane_a(new_rows, remaining)
    lane_b_table: list[dict[str, Any]] = []
    b_fail: list[dict[str, Any]] = []
    ready_b = list(preflight.get("lane_b_ready_aliases") or [])
    if lane_b_run is None:
        write_csv(LANE_B_DIR / "failures.csv", [])
        write_csv(LANE_B_DIR / "results.csv", [])
        lane_b = analyze_lane_b([], [])
        lane_b["aliases"] = []
    else:
        lane_b_table, b_fail = results_to_rows(lane_b_run, source="task007b")
        write_csv(LANE_B_DIR / "failures.csv", b_fail)
        if not lane_b_run.get("ok"):
            write_csv(LANE_B_DIR / "results.csv", lane_b_table)
            print(
                json.dumps(
                    {
                        "warning": "Lane B incomplete; continuing with available cells",
                        "successful": lane_b_run.get("successful"),
                        "failed": lane_b_run.get("failed"),
                    }
                ),
                flush=True,
            )
        ran_aliases = sorted({row["model_alias"] for row in lane_b_table}) or ready_b
        lane_b = analyze_lane_b(lane_b_table, ran_aliases)
        lane_b["aliases"] = ran_aliases
    lane_c = run_lane_c(lane_a["merged"])
    after = protected_fingerprints()
    for name in ("v2", "study1", "q2", "task006_sqlite", "paperDirection"):
        if (before.get(name) or {}).get("sha256") != (after.get(name) or {}).get("sha256"):
            raise RuntimeError(f"{name} hash changed during Task 007")
    if QUAL_SQLITE.exists() and before["task006_sqlite"]["sha256"] != after["task006_sqlite"]["sha256"]:
        raise RuntimeError("Task 006 sqlite hash changed")
    packet_summary = {
        "lane_c": lane_c,
        "lane_b_classes": lane_b.get("classes"),
        "hold_revision": None,
    }
    hold = decide_hold(lane_a, lane_c, True)
    packet_summary["hold_revision"] = hold
    write_lane_e(packet_summary)
    finished = datetime.now(UTC).isoformat()
    summary = {
        "started": started,
        "finished": finished,
        "preflight": {k: v for k, v in preflight.items() if k not in {"lane_a_calls", "lane_b_calls", "selected", "remaining"}},
        "lane_a": lane_a,
        "lane_b": lane_b,
        "lane_b_classes": lane_b.get("classes"),
        "lane_c": {
            "buckets": lane_c.get("buckets"),
            "switches": lane_c.get("switches"),
            "catching": lane_c.get("catching"),
        },
        "lane_d": lane_d,
        "fingerprints": after,
        "hold_revision": hold,
        "lane_a_new_calls": lane_a_run.get("successful"),
        "lane_a_this_process_new": lane_a_run.get("new_scientific_calls"),
        "lane_a_reused": lane_a_run.get("reused"),
        "lane_b_new_calls": 0 if lane_b_run is None else lane_b_run.get("successful"),
        "lane_b_this_process_new": 0 if lane_b_run is None else lane_b_run.get("new_scientific_calls"),
        "lane_b_reused": 0 if lane_b_run is None else lane_b_run.get("reused"),
        "lane_a_provider_attempts": lane_a_run.get("provider_attempts"),
        "lane_b_provider_attempts": 0 if lane_b_run is None else lane_b_run.get("provider_attempts"),
        "lane_a_wall": lane_a_run.get("wall_clock_seconds"),
        "lane_b_wall": 0 if lane_b_run is None else lane_b_run.get("wall_clock_seconds"),
        "lane_a_usd": lane_a_run.get("estimated_cost_usd"),
        "lane_b_usd": 0 if lane_b_run is None else lane_b_run.get("estimated_cost_usd"),
        "lane_a_n_fail": len(a_fail),
        "lane_b_n_fail": len(b_fail),
    }
    _copy_scripts()
    write_root_report(summary)
    write_support_files(summary)
    append_decision_log(summary)
    print(
        json.dumps(
            {
                "ok": True,
                "lane_a_new": summary["lane_a_new_calls"],
                "lane_b_new": summary["lane_b_new_calls"],
                "n_merged": lane_a.get("n_merged"),
                "gpt_bucket": (lane_c.get("buckets") or {}).get("openai_gpt56_sol"),
                "claude_bucket": (lane_c.get("buckets") or {}).get("anthropic_sonnet5"),
                "gemini_grok": {
                    alias: (lane_b.get("classes") or {}).get(alias, {}).get("class")
                    for alias in GEMINI_GROK
                },
                "hold_revision": hold,
                "paperDirection_sha256": after["paperDirection"]["sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
