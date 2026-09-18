"""Task 007 Lane E: paperDirection decision packet. Does not edit paperDirection.txt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .study1_analysis import MODEL_LABELS
from .task007_common import RETURN_DIR


def _bucket(summary: Mapping[str, Any], model: str) -> str:
    return ((summary.get("lane_c") or {}).get("buckets") or {}).get(model, {}).get(
        "bucket", "UNIDENTIFIABLE"
    )


def choose_spine(summary: Mapping[str, Any]) -> tuple[str, str]:
    gpt = _bucket(summary, "openai_gpt56_sol")
    claude = _bucket(summary, "anthropic_sonnet5")
    attenuates = "ATTENUATES" in gpt or "ATTENUATES" in claude
    sharpens = "SHARPENS" in gpt or "SHARPENS" in claude
    preserves = "PRESERVES" in gpt or "PRESERVES" in claude
    if gpt == claude and "PRESERVES" in gpt:
        return "B", gpt
    if gpt == claude and "ATTENUATES" in gpt:
        return "A", gpt
    if gpt == claude and "SHARPENS" in gpt:
        return "B", gpt
    if gpt != claude:
        return "C", f"GPT {gpt}; Claude {claude}"
    if preserves and not attenuates:
        return "B", gpt
    if attenuates and not preserves:
        return "A", gpt
    return "C", f"GPT {gpt}; Claude {claude}; sharpens={sharpens}"


def write_lane_e(summary: Mapping[str, Any]) -> None:
    spine_id, spine_note = choose_spine(summary)
    gpt_b = _bucket(summary, "openai_gpt56_sol")
    claude_b = _bucket(summary, "anthropic_sonnet5")
    gg = summary.get("lane_b_classes") or {}
    rows = [
        ("Displayed confidence changes verification", "Study 1 numerical cliff; Task 006/007 qualitative grids", "003/006/007", "SUPPORTED_STRONGLY_BUT_NOT_CONFIRMATORY", "Displayed score causally changes VERIFY/USE", "The model lost its internal evidence"),
        ("GPT numerical threshold cliff", "Study 1 0.89→0.91 / 0.94→0.96 near-deterministic GPT flips; Task 004 stability", "003/004", "SUPPORTED_STRONGLY_BUT_NOT_CONFIRMATORY", "Under explicit L/C, GPT's binary action pins to the displayed number", "GPT has no item-specific evidence"),
        ("Score effect survives without explicit L/C arithmetic", "Task 006 n=20 GO; Task 007 full-100 qualitative replication", "006/007", "SUPPORTED_EXPLORATORY", "Qualitative displayed confidence still changes verification", "The Study-1 cliff was only arithmetic instruction-following, full stop"),
        ("Verification contains information beyond q1", "Checkpoint A / Task 005 hidden vs q1", "005", "SUPPORTED_EXPLORATORY", "Hidden VERIFY can rank errors better than one verbal confidence", "Privileged internal truth signal"),
        ("Verification contains information beyond q1+q2", "Task 005 Lane B Contrast A", "005", "SUPPORTED_EXPLORATORY", "A second confidence sample does not absorb the hidden residual", "q2 proves a hidden state"),
        ("Verification contains information beyond q1+q2+difficulty", "Task 005C Contrast B", "005C", "WEAK", "GPT: small leftover ΔAUROC after difficulty; Claude: largely explained", "Natural verification is a mysterious correctness oracle"),
        ("Claude fixed-score wrong-vs-correct gap", "Task 005 Lane A all 10 cells PILOT_POSITIVE; largest at 0.99", "005", "SUPPORTED_EXPLORATORY", "At a fixed displayed score Claude verifies wrong answers more often", "Claude retains hidden-state evidence of its own errors"),
        ("Difficulty explanation of Claude gap", "Task 005B grouped CV leftover ~0 after other-model difficulty", "005B", "FALSIFIED_OR_LARGELY_EXPLAINED", "Claude's fixed-score gap is largely an empirical-difficulty pattern", "Claude's gap is unused internal knowledge"),
        ("Natural/hidden verification tracks empirical item difficulty", "005B/005C associations; Task 007 Lane C curves", "005C/007", "SUPPORTED_EXPLORATORY", "VERIFY propensity rises on items fewer other models got right", "True difficulty, or a production routing feature"),
        ("Score conditioning vs difficulty sensitivity", f"Task 007 Lane C: GPT {gpt_b}; Claude {claude_b}", "007", "SUPPORTED_EXPLORATORY", "State the Lane C bucket without crowding-out language unless ATTENUATES is actually observed", "Score feedback suppresses hidden evidence"),
        ("Routing gain from a single hidden judgment", "Task 006 Lane B: GPT single AI/L10 ≈ raw q1; Claude worse than q1", "006", "WEAK", "One predeclared hidden cell is not a proven router upgrade", "Deploy the hidden bit as the production policy"),
        ("Routing gain from multi-elicitation hidden aggregate", "Checkpoint A / Task 005/006 aggregate catch > raw q1 for GPT", "005/006", "SUPPORTED_EXPLORATORY", "Multi-cell hidden fraction is an upper-bound diagnostic", "Fair one-call production router"),
        ("Cross-model generality", f"Gemini/Grok 20-q qualitative: {gg}", "007-B", "WEAK" if gg else "UNTESTED", "Describe only the 20-question qualitative pattern", "Four-model law of verification"),
        ("Hidden-state mechanism", "No probes, steering, or activations were run", "none", "UNTESTED", "Do not claim a mechanism", "Suppressed internal uncertainty / hidden state"),
        ("Practical matched-budget payoff", "Historical V2-B exploratory catch tables only", "005/006", "WEAK", "Historical matched-budget gaps are exploratory", "This router is ready to deploy"),
    ]
    lines = [
        "# Paper-direction decision packet (recommendation only)",
        "",
        "Do **not** treat this file as an edit to `paperDirection.txt`. GPT reviews it first.",
        "",
        f"Best-supported spine from Task 007 Lane C: **Spine {spine_id}** ({spine_note}).",
        "",
        "## E.1 Evidence ledger",
        "",
        "| claim | current evidence | strongest task | status | allowed wording | wording to avoid |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    spines = {
        "A": {
            "title": "Spine A — score attenuates item sensitivity",
            "lay": "Showing a high confidence number does not just make the model check less often. It also makes the remaining checks less tied to which items look empirically hard.",
            "q": "Does feeding back a displayed confidence score reduce the influence of item-level difficulty on verification, even without explicit costs?",
            "attacks": "Reviewers will say you merely saturated the action at high scores, so difficulty 'disappeared' for mechanical reasons.",
            "confirm": "A fresh sample must show the attenuation at intermediate, non-saturated operating points with predeclared hard/easy bins.",
        },
        "B": {
            "title": "Spine B — score mostly shifts the checking budget",
            "lay": "The displayed score mainly decides how many answers get checked. The model still tends to spend the remaining checks on empirically harder items.",
            "q": "When a displayed confidence score changes the verification rate, does item-difficulty prioritization remain intact?",
            "attacks": "Reviewers will say difficulty is just a correlate of the model's own error, or that switch-sets are too small.",
            "confirm": "Prospective four-model study with matched-budget routing and leave-one-model-out difficulty as an evaluation covariate.",
        },
        "C": {
            "title": "Spine C — model-specific / mixed",
            "lay": "Displayed scores change checking, but GPT and Claude (and maybe Gemini/Grok) do not share one difficulty-priority story.",
            "q": "Which parts of score-conditioned verification are shared across models, and which are model-specific?",
            "attacks": "Four-model fishing; post-hoc spines per model.",
            "confirm": "Predeclare GPT vs Claude contrasts and a Gemini/Grok replication rule before the fresh study.",
        },
    }
    lines += ["", "## E.2 Candidate paper spines", ""]
    order = ["B", "A", "C"] if spine_id == "B" else ["A", "B", "C"] if spine_id == "A" else ["C", "B", "A"]
    for i, key in enumerate(order, start=1):
        spec = spines[key]
        lines += [
            f"### {i}. {spec['title']}" + ("  *(best fit to Task 007)*" if key == spine_id else ""),
            "",
            spec["lay"],
            "",
            f"- Central question: {spec['q']}",
            "- Contributions: (1) causal displayed-score control without requiring L/C arithmetic; (2) empirical-difficulty account of natural verification; (3) budget-shift vs attenuation test; (4) matched-budget routing as the payoff, not a hidden-state story; (5) GPT vs Claude (and optional Gemini/Grok) contrast.",
            "- Likely figures: Fig 1 qualitative score-response; Fig 2 VERIFY vs other-model-correct by displayed score; Fig 3 switch-set; Fig 4 matched-budget catch.",
            f"- Reviewer attack: {spec['attacks']}",
            f"- Fresh confirmation must establish: {spec['confirm']}",
            "- Must NOT claim: hidden-state suppression, a deployable cross-model difficulty router, or confirmatory status for Tasks 003–007.",
            "",
        ]
    lines += [
        "## E.3 Proposed edits to paperDirection.txt (diff-style recommendation only)",
        "",
        "### KEEP",
        "",
        "- The opening that D1 studies a displayed uncertainty score as a **control input for selective oversight**.",
        "- Study 1's GPT numerical cliff and Task 004 stability as the causal instrument under explicit L/C.",
        "- Checkpoint A's warning that visible GPT largely follows an overconfident scalar.",
        "- The distinction between exploratory Tasks 003–007 and a later prospective confirmatory study.",
        "- The ban on inferring internal-state suppression from saturated binary actions.",
        "",
        "### REWRITE",
        "",
        "- Executive summary / surviving opening: replace leftover 'does useful answer-specific error information remain expressed' language with the Task-007 question: budget shift vs changed item prioritization, with empirical difficulty as the predeclared axis.",
        "- 'Current evidence and what D1 has actually established': add Tasks 005B/005C/006/007 in one paragraph. State that q2 does not absorb hidden residual, but other-model difficulty absorbs most of Claude's fixed-score gap and most of the hidden-after-q1+q2 residual.",
        "- Recommended next experiments: the 800-call reasoning grid remains BLOCKED in-repo; do not keep it as an imminent paid step. The 1,600-call qualitative experiment is now done (20+80). Next paid science is a GPT-reviewed fresh four-model study, not more historical mining.",
        "- Elevator pitch / title family: drop 'A Score Is Not the State' as a headline if Spine B wins; that title over-promises a state/mechanism result.",
        "",
        "### DELETE/DEMOTE",
        "",
        "- Any remaining implication that hidden VERIFY is leftover internal correctness knowledge after difficulty is controlled.",
        "- Treating the 8-cell hidden aggregate as a fair production router.",
        "- Automatic four-model completeness as a scientific requirement (keep Gemini/Grok as a cheap generalization check, which Task 007-B now is).",
        "- The sequential 'zero-dollar analysis then 1000 q2 then 800 reasoning then 1600 qualitative' itinerary as if it were still future work.",
        "",
        "### ADD",
        "",
        "- A one-page map of Tasks 005B/005C/006/007 and the Lane C bucket.",
        "- Explicit allowed-claim / forbidden-claim table from E.1.",
        "- Prospective design options D1/D2/D3 with cell counts, marked not frozen.",
        "- Statement that cross-model difficulty is an evaluation proxy.",
        "",
        "Quote-level anchors in the current file: 'The surviving opening is therefore narrower'; 'D1 should study what happens when an externalized scalar uncertainty report becomes a control input'; 'A Score Is Not the State'. Do not rewrite those sentences here.",
        "",
        "## GO / HOLD",
        "",
    ]
    hold = summary.get("hold_revision")
    if hold:
        lines += [
            "**HOLD revision** until the issue in the Task 007 root report is resolved.",
            "",
            str(hold),
            "",
        ]
    else:
        lines += [
            "**GO toward paperDirection revision after GPT review.** Task 007 produced a coherent exploratory pattern that can be stated without hidden-state claims. Do not edit the file until GPT accepts this packet.",
            "",
        ]
    (RETURN_DIR / "paper_direction_decision_packet.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _ = MODEL_LABELS
