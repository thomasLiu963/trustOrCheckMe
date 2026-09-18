"""Task 006 orchestrator: qualitative 20-question pilot + parallel zero-cost lanes."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .study1_analysis import MODEL_LABELS
from .task006_common import (
    CLAUDE_ENDPOINT,
    EXPECTED_REPEAT_IDS,
    GPT_ENDPOINT,
    LANE_A_DIR,
    LANE_B_DIR,
    LANE_C_DIR,
    LANE_D_DIR,
    QUAL_SQLITE,
    REPEAT_HASH,
    RETURN_DIR,
    SCIENTIFIC_CAP,
    TASK_ID,
    assert_006_write_target,
    protected_fingerprints,
    sha256_file,
)
from .task006_lane_a import analyze_lane_a, run_lane_a
from .task006_lane_b import run_lane_b
from .task006_lane_c import run_lane_c
from .task006_lane_d import run_lane_d


SRC_FILES = (
    "task006.py",
    "task006_common.py",
    "task006_prompts.py",
    "task006_lane_a.py",
    "task006_lane_b.py",
    "task006_lane_c.py",
    "task006_lane_d.py",
)


def _copy_scripts() -> None:
    src = PROJECT_ROOT / "src"
    mapping = {
        LANE_A_DIR: ("task006_lane_a.py", "task006_prompts.py", "task006_common.py"),
        LANE_B_DIR: ("task006_lane_b.py", "task006_common.py"),
        LANE_C_DIR: ("task006_lane_c.py", "task006_common.py"),
        LANE_D_DIR: ("task006_lane_d.py", "task006_common.py"),
    }
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    for dest, names in mapping.items():
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(src / name, dest / name)
    shutil.copy2(src / "task006.py", RETURN_DIR / "task006.py")


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
    if QUAL_SQLITE.exists():
        paths.append(str(QUAL_SQLITE.relative_to(PROJECT_ROOT)) + "  (gitignored new sqlite; not V2/Study1/q2)")
    return paths


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


def write_root_report(
    *,
    lane_a_run: Mapping[str, Any],
    lane_a_analysis: Mapping[str, Any] | None,
    lane_b: Mapping[str, Any],
    lane_c: Mapping[str, Any],
    lane_d: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
) -> None:
    pre = lane_a_run.get("preflight") or {}
    analysis = lane_a_analysis or {}
    gate = analysis.get("gate") or {}
    rate_rows = analysis.get("rate_rows") or []
    score_rows = analysis.get("score_rows") or []
    stakes_rows = analysis.get("stakes_rows") or []
    sat_rows = analysis.get("sat_rows") or []
    routing_rows = analysis.get("routing_rows") or []
    b_sum = lane_b.get("summaries") or {}

    def rates(model: str, family: str) -> str:
        bits = []
        for cond in ("hidden", "displayed_0.70", "displayed_0.90", "displayed_0.99"):
            hits = [
                r
                for r in rate_rows
                if r["model_alias"] == model
                and r["family"] == family
                and r["score_condition"] == cond
            ]
            if hits:
                bits.append(f"{cond} {_pp(hits[0]['verify_rate'])} ({hits[0]['n_verify']}/{hits[0]['n']})")
        return "; ".join(bits) if bits else "not available"

    def score_line(model: str, family: str) -> str:
        rows = [
            r
            for r in score_rows
            if r["model_alias"] == model and r["family"] == family
        ]
        if not rows:
            return "not available"
        parts = []
        for r in rows:
            parts.append(
                f"{r['from_score']}→{r['to_score']} {_delta(r['mean_delta_verify'])} "
                f"(dec {r['n_decreased']}/inc {r['n_increased']}/same {r['n_unchanged']})"
            )
        return "; ".join(parts)

    gpt_intermediate = [
        r for r in sat_rows if r["model_alias"] == "openai_gpt56_sol" and r["useful_intermediate_10_90"]
    ]
    identifiable = [r for r in routing_rows if r.get("label") == "identifiable_pilot_only"]

    paid = 0 if lane_a_run.get("dry_run") else int(lane_a_run.get("new_scientific_calls") or 0)
    lines = [
        "# Task 006 — Qualitative-Stakes Pilot + Parallel Paper-Critical Prep",
        "",
        "Exploratory only. Not confirmatory. `paperDirection.txt` was not modified.",
        "",
        "READY_FOR_GPT_REVIEW = YES" if not lane_a_run.get("dry_run") and analysis else "READY_FOR_GPT_REVIEW = NO (Lane A dry-run only)",
        "",
        "---",
        "",
        "## 1. Plain-English executive bottom line",
        "",
    ]
    if lane_a_run.get("dry_run"):
        lines += [
            "Lane A paid calls have not been made. Preflight/prompt freeze completed. Lanes B/C/D are zero-cost and complete.",
            "",
        ]
    else:
        lines += [
            f"Qualitative 20-question pilot gate: **{gate.get('gate', 'UNASSIGNED')}**. {gate.get('reason', '')}",
            "",
            "Numerical L/C and expected-value language were absent from the paid prompts. Displayed-score tokens used the historical Study-1 sentence and `.12g` formatter.",
            "",
        ]
    lines += [
        "Exploratory historical routing still shows a material catch-rate gap versus raw q1 at equal verification budget, especially when a hidden verification judgment is available. That remains exploratory on the same 500 questions.",
        "",
        f"A fresh unseen MMLU-Pro pool of {lane_c.get('n_eligible')} questions is prepared with candidate N=200…1000 lists. Gemini/Grok adapters exist; open-weight inference does not.",
        "",
        "## 2. Exact paid scientific calls made",
        "",
        f"- Target: **{SCIENTIFIC_CAP}**",
        f"- Made: **{paid if not lane_a_run.get('dry_run') else 0}**",
        f"- Provider attempts: {lane_a_run.get('provider_attempts', 0) if not lane_a_run.get('dry_run') else 0}",
        f"- Failures: {analysis.get('n_fail', 0) if analysis else 0}",
        f"- Lanes B/C/D API calls: **0**",
        f"- New sqlite: `{QUAL_SQLITE}`",
        "",
        "## 3. Prompt-freeze validation",
        "",
        f"- Frozen 20 IDs hash: `{REPEAT_HASH}`",
        f"- GPT `{GPT_ENDPOINT}` reasoning.effort=none; Claude `{CLAUDE_ENDPOINT}` thinking=disabled",
        f"- Moderate family hash: `{pre.get('prompt_hash_moderate')}`",
        f"- Stronger family hash: `{pre.get('prompt_hash_stronger')}`",
        f"- Visible tokens: `{pre.get('visible_tokens')}`",
        f"- Preflight ok: **{pre.get('ok')}**; audit errors: {pre.get('n_errors')}",
        "- Task-005 Lane D drafts were **not** used (they contained forbidden 'justified' / 'not as a default' language).",
        "",
        "## 4. GPT qualitative score-response result",
        "",
        f"- Moderate: {rates('openai_gpt56_sol', 'moderate')}",
        f"- Stronger: {rates('openai_gpt56_sol', 'stronger')}",
        f"- Paired deltas (pilot): {score_line('openai_gpt56_sol', 'moderate')}",
        f"- Stronger paired deltas: {score_line('openai_gpt56_sol', 'stronger')}",
        "",
        "## 5. Claude qualitative score-response result",
        "",
        f"- Moderate: {rates('anthropic_sonnet5', 'moderate')}",
        f"- Stronger: {rates('anthropic_sonnet5', 'stronger')}",
        f"- Paired deltas (pilot): {score_line('anthropic_sonnet5', 'moderate')}",
        "",
        "## 6. Stronger-vs-moderate stakes manipulation check",
        "",
    ]
    if stakes_rows:
        for row in stakes_rows:
            if row["score_condition"] == "hidden":
                continue
            lines.append(
                f"- {MODEL_LABELS[row['model_alias']]} {row['score_condition']}: "
                f"stronger−moderate {_delta(row['mean_stronger_minus_moderate'])} "
                f"(stronger-only {row['n_stronger_only']}, moderate-only {row['n_moderate_only']})"
            )
    else:
        lines.append("- Not available (Lane A not analyzed).")
    lines += [
        "",
        f"- GPT visible-cell mean stakes effect: {_delta(gate.get('stakes_mean_gpt_visible'))}; labeled active={gate.get('stakes_active')}",
        "",
        "## 7. Saturation / measurability result",
        "",
        f"- GPT visible cells in the provisional 10–90% VERIFY band: {len(gpt_intermediate)} / {len([r for r in sat_rows if r['model_alias']=='openai_gpt56_sol'])}",
        "- Full cell table: `lane_A_qualitative_pilot/saturation_diagnostic.csv`",
        "",
        "## 8. Pilot wrong-vs-correct routing where identifiable",
        "",
        f"- Identifiable (both actions present) visible cells: {len(identifiable)}",
        "- Saturated cells labeled `ITEM_ROUTING_NOT_IDENTIFIABLE_DUE_TO_ACTION_SATURATION`.",
        "- n=20; do not overinterpret.",
        "",
        "## 9. Lane-A gate",
        "",
        f"**{gate.get('gate', 'UNASSIGNED')}**",
        "",
        gate.get("reason", "Lane A analysis not complete."),
        "",
        "## 10. Exploratory routing result",
        "",
        "Historical V2-B, GPT and Claude, grouped-CV where learned. Primary metric: error catch at 10–50% verification budgets.",
        "",
    ]
    for alias, label in MODEL_LABELS.items():
        s = b_sum.get(alias) or {}
        lines += [
            f"- **{label}** catch@30% raw q1={s.get('catch_at_30_raw_q1')}; "
            f"q2={s.get('catch_at_30_q2')}; "
            f"single hidden AI/L10={s.get('catch_at_30_single_hidden')}; "
            f"hidden aggregate={s.get('catch_at_30_hidden_agg')} (not deployment-fair); "
            f"combined q1+q2+single-hidden={s.get('catch_at_30_combined')}",
        ]
    lines += [
        "",
        "Mandatory-looking baselines for a later fresh study, if GPT agrees: raw q1, calibrated q1, q1+q2, and one hidden verification judgment. Do not freeze the router from this exploratory table.",
        "",
        "## 11. Fresh unseen candidate-pool readiness",
        "",
        f"- Source MMLU-Pro test revision `{lane_c.get('revision')}`: {lane_c.get('n_source')} items",
        f"- Eligible after excluding 500 V2-B IDs: **{lane_c.get('n_eligible')}**",
        f"- Pool hash: `{lane_c.get('pool_hash')}`",
        "- Candidate stratified lists: N=200,400,600,800,1000 with predeclared seeds. Not the confirmatory sample.",
        "",
        "## 12. Gemini/Grok/open-model feasibility",
        "",
    ]
    for row in lane_d.get("closed") or []:
        lines.append(
            f"- {row['label']}: **{row['status']}** (`{row['api_model']}`); historical V2 Stage-1/2 present; no Study 1/q2 replication"
        )
    lines += [
        f"- Open-weight: **{lane_d.get('open_status')}** (no inference adapter in `src/`)",
        "",
        "## 13. What the combined Task-006 evidence DOES establish",
        "",
        "- Whether a non-numerical qualitative verification prompt still produces systematic displayed-score responsiveness in this 20-question GPT/Claude pilot.",
        "- Whether stronger vs moderate consequence wording is behaviorally active at n=20.",
        "- Whether GPT binary actions become measurable (non-saturated) without L/C arithmetic.",
        "- That historical matched-budget routing still looks like a paper-critical payoff worth prospective confirmation, with resource labels that prevent treating the hidden aggregate as a one-call router.",
        "- That a leakage-safe unseen MMLU-Pro pool and candidate N plans exist without peeking at new model outcomes.",
        "- That Gemini/Grok closed-model replication is adapter-ready; open-weight is not.",
        "",
        "## 14. What it DOES NOT establish",
        "",
        "- A confirmatory effect size, a paper-ready qualitative cliff, or a frozen confirmatory sample.",
        "- That GPT has 'zero internal evidence' if qualitative actions remain saturated.",
        "- A hidden-state / suppression mechanism.",
        "- That q2 or hidden should be the production router.",
        "- Gemini, Grok, or open-model qualitative behavior (not run).",
        "- Task 005B difficulty-control conclusions (separate packet).",
        "",
        "## 15. Recommended next action(s), recommendation only",
        "",
        "Do not execute these in Task 006.",
        "",
        "1. GPT reviews Task 005B and Task 006 together before any scale-up.",
        "2. If Lane A is A-GO-FULL-QUALITATIVE or A-GO-NARROW, freeze a larger qualitative experiment in a later numbered task using only the useful family/conditions.",
        "3. If A-FAIL-QUALITATIVE or A-REVISE-PROMPT, do not scale this wording; GPT should rewrite once.",
        "4. Do not launch a confirmatory 600–1000 study from these diagnostics.",
        "5. Do not run Gemini/Grok/open models until GPT authorizes a numbered replication task.",
        "6. Keep `paperDirection.txt` unmodified.",
        "",
        "## 16. Exact cost / runtime / retries",
        "",
        f"- Lane A scientific calls: {0 if lane_a_run.get('dry_run') else lane_a_run.get('new_scientific_calls', 0)}",
        f"- Lane A provider attempts: {0 if lane_a_run.get('dry_run') else lane_a_run.get('provider_attempts', 0)}",
        f"- Lane A wall-clock seconds: {lane_a_run.get('wall_clock_seconds', 0) if not lane_a_run.get('dry_run') else 0}",
        f"- Lane A estimated USD: {analysis.get('cost_usd', 0) if analysis else 0}",
        "- Lanes B/C/D: 0 calls",
        f"- V2 sha256: `{fingerprints['v2']['sha256']}`",
        f"- Study 1 sha256: `{fingerprints['study1']['sha256']}`",
        f"- paperDirection sha256: `{fingerprints['paperDirection']['sha256']}`",
        "",
        "## 17. READY_FOR_GPT_REVIEW",
        "",
        "READY_FOR_GPT_REVIEW = YES" if not lane_a_run.get("dry_run") and analysis else "READY_FOR_GPT_REVIEW = NO",
        "",
    ]
    (RETURN_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_support_files(
    *,
    lane_a_run: Mapping[str, Any],
    lane_b: Mapping[str, Any],
    lane_c: Mapping[str, Any],
    lane_d: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    started: str,
    finished: str,
) -> None:
    cost = [
        "# Task 006 cost summary",
        "",
        f"- Lane A scientific: {0 if lane_a_run.get('dry_run') else lane_a_run.get('new_scientific_calls', 0)} / {SCIENTIFIC_CAP}",
        f"- Lane A provider attempts: {0 if lane_a_run.get('dry_run') else lane_a_run.get('provider_attempts', 0)}",
        f"- Lane A estimated USD: see lane_A_qualitative_pilot/cost_summary.md",
        "- Lanes B/C/D: $0",
        "",
    ]
    (RETURN_DIR / "cost_summary.md").write_text("\n".join(cost), encoding="utf-8")
    log = [
        "# Task 006 parallel execution log",
        "",
        f"- Started: {started}",
        f"- Finished: {finished}",
        "- Isolation: new sqlite `results/study1_qualitative_pilot/qualitative_pilot.sqlite3`",
        "- Did not write V2, Study 1, q2, Task 005, Task 005B, or paperDirection.txt",
        "- Lane order: B, C, D (zero-cost) in this process; Lane A prompt audit before paid calls",
        f"- Frozen 20 IDs: {list(EXPECTED_REPEAT_IDS)}",
        f"- Repeat hash: {REPEAT_HASH}",
        f"- Lane B rows: {lane_b.get('n_rows')}",
        f"- Lane C eligible: {lane_c.get('n_eligible')}",
        f"- Lane D open status: {lane_d.get('open_status')}",
        f"- Protected hashes after run: V2 `{fingerprints['v2']['sha256'][:12]}…` Study1 `{fingerprints['study1']['sha256'][:12]}…`",
        "",
    ]
    (RETURN_DIR / "parallel_execution_log.md").write_text("\n".join(log), encoding="utf-8")
    manifest = {
        "task": TASK_ID,
        "started": started,
        "finished": finished,
        "scientific_cap": SCIENTIFIC_CAP,
        "lane_a_dry_run": bool(lane_a_run.get("dry_run")),
        "lane_a_new_calls": lane_a_run.get("new_scientific_calls"),
        "fingerprints": fingerprints,
        "repeat_hash": REPEAT_HASH,
    }
    (RETURN_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (RETURN_DIR / "changed_files.txt").write_text(
        "\n".join(_list_changed(RETURN_DIR)) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-cost", action="store_true", help="Lanes B/C/D only")
    parser.add_argument("--dry-run", action="store_true", help="Lane A prompt audit, 0 paid calls")
    parser.add_argument("--yes", action="store_true", help="authorize Lane A paid calls")
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args(argv)
    assert_006_write_target(QUAL_SQLITE)
    started = datetime.now(UTC).isoformat()
    before = protected_fingerprints()
    lane_b = run_lane_b()
    lane_c = run_lane_c()
    lane_d = run_lane_d()
    allow_paid = bool(args.yes) and not args.zero_cost and not args.dry_run
    lane_a_run = asyncio.run(
        run_lane_a(
            allow_paid=allow_paid,
            max_calls=args.max_calls if allow_paid else None,
        )
    )
    analysis = None
    if allow_paid and lane_a_run.get("ok") and not lane_a_run.get("dry_run"):
        analysis = analyze_lane_a(lane_a_run)
    after = protected_fingerprints()
    if before["v2"]["sha256"] != after["v2"]["sha256"]:
        raise RuntimeError("V2 sqlite hash changed")
    if before["study1"]["sha256"] != after["study1"]["sha256"]:
        raise RuntimeError("Study 1 sqlite hash changed")
    if before["paperDirection"]["sha256"] != after["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt hash changed")
    _copy_scripts()
    write_root_report(
        lane_a_run=lane_a_run,
        lane_a_analysis=analysis,
        lane_b=lane_b,
        lane_c=lane_c,
        lane_d=lane_d,
        fingerprints=after,
    )
    write_support_files(
        lane_a_run=lane_a_run,
        lane_b=lane_b,
        lane_c=lane_c,
        lane_d=lane_d,
        fingerprints=after,
        started=started,
        finished=datetime.now(UTC).isoformat(),
    )
    print(
        json.dumps(
            {
                "ok": bool(lane_a_run.get("ok")),
                "dry_run": lane_a_run.get("dry_run"),
                "lane_a_errors": lane_a_run.get("errors"),
                "new_scientific_calls": lane_a_run.get("new_scientific_calls"),
                "lane_b_n": lane_b.get("n_rows"),
                "lane_c_eligible": lane_c.get("n_eligible"),
                "gate": (analysis or {}).get("gate", {}).get("gate"),
            },
            indent=2,
        )
    )
    if not lane_a_run.get("ok"):
        raise SystemExit("Task 006 Lane A preflight or run failed")


if __name__ == "__main__":
    main()
