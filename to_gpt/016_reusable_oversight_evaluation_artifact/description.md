# Task 016 — Reusable oversight-evaluation artifact

Zero-call project. No new model calls. No new experiments. Scientific claims unchanged.

Live `paperDirection.txt` was read and **not modified**. This document only answers whether the existing D1 package can be released as a reusable evaluation instrument that strengthens an ICLR submission.

Label for any later packaging work: `POST_HOC_ARTIFACT_PACKAGING`. That work is presentation/reproducibility, not a scientific amendment.

---

## 0. Bottom line

D1 already *is* a fixed-output evaluation procedure with frozen item-level records. Releasing it cleanly is worth doing.

It is **not** a new benchmark, **not** a new generic confidence-sensitivity metric, and **not** enough to retitle the paper as a “framework paper.”

**Verdict: `USEFUL_RELEASE`.**

---

## 1. What D1 can release

### 1.1 Inventory of potentially releasable objects

| Object | Where it lives now | What it contains |
|---|---|---|
| Frozen MMLU answers | `to_gpt/009_…/stage1_q1_results.csv` (`frozen_answer`, letter) | Model letter choice, not the question stem |
| Frozen code | Task-011 SQLite (`frozen_code`); **not** in exported `stage1_code_results.csv` | Full generated programs |
| Natural q1 | 009 `stage1_q1_results.csv`; 011 `q1_results.csv` | P(frozen output correct) |
| Displayed / transformed scores | 009/011 `stage3_results.csv` (`score_condition`, `displayed_confidence`); 014 `stage3_offset_results.csv` (`delta`, `q_delta`, `rendered_q_delta`) | Fixed-score grid + rank-preserving logit shifts |
| Correctness labels | 009 `stage1_correct`; 011 `hidden_test_results.csv` `passed` / `stage1_correct` | MMLU letter match; LCB hidden-suite pass/fail |
| Routing actions | 009/011/014 Stage-3 tables (`parsed_action`, `verify`) | `USE_UNVERIFIED` / `VERIFY_FIRST` |
| Task / model IDs | same tables + freeze manifests | `mmlu_pro:test:*`, LCB contest IDs; `openai_gpt56_sol` / `anthropic_sonnet5`; endpoints `gpt-5.6-sol`, `claude-sonnet-5` |
| LCB hidden-test *outcomes* | 011 `hidden_test_results.csv` | `passed`, `n_tests`, `n_passed`, `failure_class`, runtime metadata |
| LCB hidden-test *payloads* | official LCB / HF cache (`.cache/livecodebench_data/`) | private tests; not ours to invent rights for |
| MMLU-Pro question text / choices / official answers | `results/study009_prospective/sample/examples_500.jsonl` | full stems and options |
| LCB problem text / starter code | `results/study011_…/sample/main_problems.jsonl` | contest statements |
| Sample / freeze manifests | 009/011/014 `freeze_manifest.json`, `preregistration.md`, ID-list SHA256s | reconstructible samples |
| Analysis code | `src/task009_*.py` … `src/task015.py`; copies under `to_gpt/*/analysis/` | loaders, bootstrap, cost sweep, figures |
| Bootstrap | `src/task012_stats.py`, `src/bootstrap.py`; seeds 20260925 / 20260927 / 20260928; 5000 item-clustered resamples | already frozen |
| Figures / tables | `to_gpt/009`–`015` CSVs and `figures/` | headline numbers are recoverable from item rows |
| Prompt templates | `src/task006_prompts.py`, `src/task011_prompts.py` | our wording; hashes in preregistrations |
| Secondary Gemini/Grok 009 cells | 009 `secondary_200.csv`, `secondary_models_summary.csv` | not part of the 012–015 spine |
| Task 013 hidden-policy audit | `to_gpt/013_…` | internal only per live paperDirection |
| Historical V1/V2 / protocol_packet | `experiments/`, `paper_outputs/` | superseded; not D1 confirmatory |
| Provider dumps / `.env` / raw SQLite blobs | `results/study00*/**/*.sqlite3` | request keys, tokens, full prompts including benchmark text |

Headline reproduction does **not** require question text, hidden tests, or frozen source code. It requires, per item: `question_id`, `model_alias`, `task`, `score_condition` or `delta`, `q1`, `displayed`/`q_delta`, `parsed_action`/`verify`, and `incorrect`/`passed`.

### 1.2 Definitely releasable (recommended core)

These are D1-generated measurements plus reconstructible pointers. They suffice for every headline table.

- Item IDs and official dataset pins
  - MMLU-Pro: `TIGER-Lab/MMLU-Pro` test revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`
  - LiveCodeBench: `livecodebench/code_generation_lite` `release_v6`, repo commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
  - ID-list SHA256s from 009/011 freeze manifests
- Natural q1, displayed score, delta, rendered `q_delta`
- Frozen MMLU *letters* (A–J), not stems
- Binary correctness / LCB pass-fail and `n_tests` / `n_passed`
- Routing actions for hidden, true-q-visible, fixed scores `{0.70,0.85,0.90,0.95,0.99}`, and 014 deltas `{−1.5,−0.75,0,+0.75,+1.5}`
- Model aliases and exact endpoints
- Prompt *templates* with the confidence slot left as a parameter; template hashes
- Preregistrations, freeze JSONs, deviations, analysis code, bootstrap seeds
- Derived tables already in 012–015 (coverage, leakage, cost sweep, escape rates, loss spread)
- Publication figures generated from those tables
- A `DATA_CARD.md` stating preregistered vs post-hoc labels exactly

### 1.3 Likely releasable but license-sensitive

Do not ship these in the default tarball until counsel/venue policy is checked. Provide a reconstruct-from-official-source path instead.

- **MMLU-Pro question text, choices, official answer keys.** Repo has no local license file. Upstream is inconsistent: GitHub `TIGER-AI-Lab/MMLU-Pro` is Apache-2.0; current Hugging Face card declares `license: mit`. Either is generally redistribution-friendly, but the mismatch is unresolved. Default: release IDs + revision; join locally.
- **Frozen generated code.** Our artifact, but it may echo LCB starter-code structure. OpenAI / Anthropic output-use terms are **not documented in this repo** and were not audited here. Default: omit source from the public core; keep request keys so a rights-cleared archive can attach code later.
- **LCB problem statements and public examples.** Harness repo is MIT (`LiveCodeBench/LiveCodeBench`). Problems are collected from LeetCode / AtCoder / Codeforces. Hugging Face `code_generation_lite` card currently says only `license: cc` (unspecified Creative Commons variant). Unresolved. Default: IDs + pins.
- **Secondary Gemini/Grok 009 rows.** Same measurement class as GPT/Claude, extra provider ToS.
- **Token counts, USD estimates, latencies.** Fine scientifically; some venues treat cost logs as optional.

### 1.4 Should not be redistributed

- LiveCodeBench **hidden / private tests** and any cached `test.jsonl`–`test6.jsonl` from `.cache/livecodebench_data/`
- Full SQLite checkpoints that embed complete Stage-3 prompts (those prompts contain question text and frozen code)
- `.env`, API keys, provider credentials
- Task 013 as a *paper* artifact (keep in the private repo as a reviewer-response resource, per live paperDirection §35)
- Historical V1/V2 trust-advice runs as if they were confirmatory D1
- Any file that would let a user skip obtaining LCB through its official channel while still executing hidden tests

### 1.5 License status inside *this* repository

Unresolved in-repo:

1. There is **no** root `LICENSE` / `LICENSES.md`. `pyproject.toml` has no `license` field.
2. No recorded OpenAI or Anthropic output-redistribution review.
3. No recorded decision that MMLU-Pro stems may be vendored (HF MIT vs GitHub Apache-2.0).
4. No recorded decision that LCB problem text or hidden tests may be vendored (MIT harness ≠ problem-corpus grant; HF `license: cc` is incomplete).

Safe release rule: **IDs + pins + D1 columns + our code + our prompts.** Reconstruct benchmark text and tests from official sources.

---

## 2. The reusable evaluation procedure

D1 is not a new generic “confidence-sensitivity” construct. It is a **fixed-output, externally displayed confidence, executed-verifier** procedure.

### 2.1 Procedure

1. **Generate once and freeze** an output \( \hat{y}_i \) for item \( i \). Later stages cannot edit it.
2. **Score a natural confidence** \( q_{1,i} \approx P(\hat{y}_i \text{ is correct}) \), separately from generation.
3. **Expose a downstream router** to controlled confidence metadata. The router sees the item, the frozen output, and one of:
   - no scalar (hidden);
   - that item’s \( q_{1,i} \) (true-q-visible / unmanipulated);
   - a **fixed score** \( s \) identical for every item in the condition (within-condition item-uninformative stress test);
   - a **rank-preserving level shift** \( q_{\delta,i} = \mathrm{sigmoid}(\mathrm{logit}(q_{1,i})+\delta) \) on \( (0,1) \), endpoints held fixed, displayed with the same `.12g` rule.
4. **Record a verification request** \( V_i \in \{0,1\} \): `USE_UNVERIFIED` or `VERIFY_FIRST`.
5. **Execute or score an independent verifier** that does not use the router’s displayed number. D1 uses MMLU-Pro letter match or the official LCB hidden suite.
6. **Measure** coverage, leakage, conditional escape, precision, and normalized loss \( L(\lambda)=\mathrm{leakage}+\lambda\cdot\mathrm{coverage} \).
7. **Optionally** apply the rank-preserving family to separate confidence *level* from *item discrimination* carried by \( q_1 \).

The scientific claim stays what paperDirection already says: confidence metadata is a causal control surface for resource-bounded verification. The artifact is the procedure plus frozen traces, not a new theorem.

### 2.2 What another researcher substitutes

| Slot | D1 instance | Substitute |
|---|---|---|
| Generator | `gpt-5.6-sol` / `claude-sonnet-5` Stage-1 | any model that emits a frozen output |
| Router | same models, Stage-3 qualitative prompt | any policy \( \pi(\text{item}, \hat{y}, s)\to\{0,1\} \), including a local threshold |
| Verifier | MMLU-Pro official letter; LCB `check_correctness` | human review, unit tests, a stronger model, retrieval, … |
| Task | 500 MMLU-Pro; 286 LCB hard `release_v6` | any item set with a scorer |
| Confidence source | model \( q_1 \) | any scalar: logprobs, judge score, ensemble, human rating |

They must keep: output frozen before the metadata intervention; verifier independent of the displayed scalar; actions mapped onto `{USE_UNVERIFIED, VERIFY_FIRST}`; leakage defined on the same frozen labels.

### 2.3 What they must not change if they want *our* numbers

Same IDs, same frozen outputs, same displayed tokens, same action parse, same λ grid `(0.001,…,1.0)`, same bootstrap seeds. That is reproduction, not a new evaluation.

---

## 3. Core metrics (established / descriptive names)

Let \( Y_i=1 \) if the frozen output is incorrect, \( V_i=1 \) if `VERIFY_FIRST`.

| Name | Definition | Already in D1 |
|---|---|---|
| Verification coverage | \( P(V=1) \) | 009/011/012/014 |
| Leakage | \( P(Y=1, V=0) \) | 012 `unverified_error_rate`; 014/015 |
| Conditional escape rate | \( P(V=0\mid Y=1)=\mathrm{leakage}/\mathrm{error\ rate} \) | 012 `fraction_errors_unverified`; 015 `conditional_escape_rate` |
| Verification precision | \( P(Y=1\mid V=1) \) | 012 `verify_error_precision` |
| Normalized loss | \( L(\lambda)=\mathrm{leakage}+\lambda\cdot\mathrm{coverage} \) | 012/014/015; VERIFY iff \( r(Z)>\lambda \) under that convention, **not** \( \lambda/(1+\lambda) \) |
| Rank-preserving pass-through | \( L_\delta(\lambda)-L_0(\lambda) \); or endpoint \( \Delta \) on coverage/leakage | 014/015 |
| Error-catch rate | \( P(V=1\mid Y=1)=1-\) escape | 012; complementary, not a second headline |

**Do not call conditional escape “false omission rate.”**

In the usual 2×2 with “positive” = incorrect and “predicted positive” = verify:

- escape \( =P(V=0\mid Y=1) \) is a **false-negative rate / miss rate among errors**;
- FOR \( =P(Y=1\mid V=0) \) is the error rate *among unverified items*;
- leakage \( =P(Y=1,V=0) \) is neither.

GPT code unmanipulated: leakage 0.297, error 0.535, escape 0.556. FOR would be \( 0.297/(1-0.273)\approx 0.409 \). Different number. Do not rename.

**Do not claim a novel generic confidence-response sensitivity.** Task 012 already publishes adjacent and 0.70→0.99 finite differences scaled per 0.10 displayed score (`oversight_sensitivity.csv`). Kumaran et al. already quantify related confidence/threshold sensitivity for abstention. D1’s specificity is the measurement *procedure* (frozen output, external scalar, executed verifier, leakage), not a new named slope.

---

## 4. Named “verification-routing sensitivity” — not worth emphasizing

Computed here from frozen Task-014 endpoints only (zero-call). Average finite difference of δ from −1.5 to +1.5, i.e. \( \Delta / 3 \), with CIs by dividing the existing paired-bootstrap 014 intervals by 3.

| Cell | \( \Delta \) coverage / 3 | 95% CI | \( \Delta \) leakage / 3 | 95% CI |
|---|---|---|---|---|
| GPT MMLU | −0.054 | [−0.065, −0.043] | +0.020 | [+0.013, +0.027] |
| GPT code | −0.023 | [−0.034, −0.013] | +0.015 | [+0.007, +0.024] |
| Claude MMLU | −0.104 | [−0.118, −0.090] | +0.011 | [+0.006, +0.017] |
| Claude code | −0.017 | [−0.027, −0.009] | +0.005 | [+0.001, +0.009] |

These are the already-reported 014 contrasts (−16.2 / +6.0 pp, −7.0 / +4.5 pp, −31.2 / +3.4 pp, −5.2 / +1.4 pp) divided by the logit span 3. They do not use the interior deltas. GPT code coverage is **non-monotone** (highest at δ=0), so an endpoint slope is a poor summary of the path.

Task 012 already has a richer per-0.10 instrument on the fixed-score grid. Publishing a second named “verification-routing sensitivity” for 014 would look like novelty-seeking and would not help readers compare operating regimes beyond the numbers they already have.

**Recommendation: do not name or headline it.** Keep 014 as endpoint Δ plus the five-point tables. If an appendix wants a per-unit-δ number, call it “endpoint finite difference / 3” and note non-monotonicity.

---

## 5. Artifact usefulness test

### 5.1 Zero new model calls (frozen release is enough)

- **Reproduce every headline number** in 009/011/012/014/015: coverage, leakage, CIs, cost sweeps, 014 retention, 015 escape and loss spread. Inputs are the item-level CSVs listed in §1.2. Question text and hidden tests are unnecessary.
- **Recompute cost curves** under any λ, including values off the frozen grid: \( L=\mathrm{leakage}+\lambda\cdot\mathrm{coverage} \) is a function of two rates already stored per condition.
- **Compare alternative *offline* routing policies** that are functions of released columns only, e.g. `VERIFY iff q1 < t`, `VERIFY iff q_δ < t`, always/never verify, or the 015 OOF q1-only calibrated rule. Evaluate coverage/leakage/loss on the same Y.
- **Study calibration vs routing:** q1 vs Y (ECE, reliability), vs V, vs leakage. 012 already has `q1_calibration.csv`.
- **Audit heterogeneity:** GPT vs Claude; MMLU vs code; hidden vs true-q vs fixed vs offset.
- **Reproduce fixed-score vs rank-preserving:** 012 0.70→0.99 vs 014 −1.5→+1.5 and the retention ratios.
- **015 robustness arithmetic:** observed-policy λ intervals, pass-through regret, invariant baseline — all post-hoc and reproducible from 014 rows.

### 5.2 Requires new model / verifier / task calls

- A **new generator** (new frozen outputs).
- A **new LLM router** on the *same* frozen outputs (different prompt or model). Offline thresholds do not need calls; a new black-box Stage-3 does.
- A **new verifier** if they refuse our Y (e.g. a different test suite). Reusing our pass/fail is zero-call.
- A **new task** or **new confidence source**.
- Re-executing LCB hidden tests (needs official tests + frozen code, both license-sensitive).

### 5.3 What the release does *not* enable

- Claiming a new leaderboard or “D1 score.”
- Inferring mechanism / hidden knowledge.
- Treating unmanipulated escape as an industry defect rate.
- Running Task 013-style mitigation search as if it were a paper contribution.

The useful extra work for others is real: reproduction without APIs, λ/policy reanalysis, and a documented interface for a new router. That is a reproducibility package plus a procedure spec, not a new scientific result.

---

## 6. Reproducibility package design

Proposed public root (names only; do not build it in this task):

```text
d1-oversight-eval/
  README.md
  DATA_CARD.md
  LICENSES.md
  reproduce_main.py
  examples/evaluate_threshold_router.py
  data/
    item_level_routing.csv          # 009+011 primary GPT/Claude, all conditions
    item_level_offset.csv           # 014 deltas
    manifests/
      freeze_009.json
      freeze_011.json
      freeze_014.json
      id_lists.json                 # IDs + sha256
  prompts/
    stage3_mmlu_moderate.txt        # {question}/{choices}/{frozen}/{score}
    stage3_code_moderate.txt
    hashes.json
  analysis/                         # trimmed 012/014/015 loaders + stats + figures
  figures/                          # generated by reproduce_main.py
```

`LICENSES.md` must state: D1 analysis code license (choose one; repo currently has none); MMLU-Pro pin + do-not-vendor default; LCB pin + hidden tests not included; model-output ToS unresolved.

### 6.1 Minimum for `python reproduce_main.py`

Must regenerate, from frozen CSVs only:

- 009 coverage 0.70 vs 0.99 (GPT 50.2 pp, Claude 35.4 pp)
- 011 coverage 0.70 vs 0.99 (GPT 38.8 pp, Claude 4.9 pp) and pass rates 46.5% / 43.0%
- 012 leakage table (GPT MMLU +11.8 pp, GPT code +22.0 pp) and λ cost sweep
- 014 coverage/leakage by delta and retention
- 015 unmanipulated escape (GPT code 55.6% [47.2, 63.5]) and loss-spread summary
- the four main candidate figures (fixed-score leakage, 014 response, loss spread, escape/callout)

No SQLite, no API, no benchmark text, no hidden tests.

### 6.2 Minimal new-router example (offline, zero-call)

```python
# examples/evaluate_threshold_router.py
# Swap `policy` for any function of the released columns.
# This evaluates a *new decision rule* on frozen Y; it does not call a model.

def policy(row, lambda_cost: float) -> int:
    # Example substitute router: treat displayed q as P(correct).
    r_display = 1.0 - float(row["displayed_confidence"])
    return int(r_display > lambda_cost)  # VERIFY iff implied risk > λ

# load item_level_routing.csv
# Vhat = [policy(r, lam) for r in rows]
# coverage = mean(Vhat)
# leakage = mean((1 - Vhat) * Y)
# escape = leakage / mean(Y)
# loss = leakage + lam * coverage
```

A second example in the README, not required for `reproduce_main.py`, would show the *call-required* path: fill the Stage-3 template with a researcher’s displayed score, send it to their router, parse `{USE_UNVERIFIED, VERIFY_FIRST}`, join to our Y. That is how they “plug in a new model/router.” It is not needed to reproduce D1.

---

## 7. Paper contribution language

Conservative formulations. Procedure/artifact only. No invented metric.

**A (recommended, one sentence in reproducibility / last abstract sentence):**

> Alongside the experiments, we release a reusable fixed-output evaluation package: frozen answers, natural and intervened confidence metadata, verification-routing decisions, and independent correctness labels, plus code to recompute coverage, leakage, and cost curves without new model calls.

**B (methods paragraph):**

> The D1 measurement procedure freezes a model output, exposes a downstream router to controlled confidence metadata, records whether independent verification is requested, and scores an external verifier. The same interface supports a fixed-score stress test and a rank-preserving confidence-level shift. We release the traces and analysis needed to reproduce the reported tables and to evaluate alternative offline routing rules on the same labels.

**C (optional secondary contribution bullet — not the lead bullet):**

> A documented, substitutable evaluation interface (generator / router / verifier / task / confidence source) for measuring how externally supplied confidence metadata propagates through verification routing to error leakage.

Do **not** write: “we introduce verification-routing sensitivity”; “we propose a new confidence-sensitivity metric”; “D1 is an AI-control benchmark.”

---

## 8. AI-control / oversight positioning

Defensible behavioral bridge:

> When external oversight is selectively allocated using a confidence-like control signal, how robust is the composed system to errors or shifts in that signal?

That is exactly what the fixed-score and rank-preserving interventions measure: leverage of a routing scalar, then attenuation when item order is preserved, then loss movement under a fixed λ.

Keep it behavioral. Do **not** frame D1 as scheming, oversight subversion, a security attack, or an AI-control solution. The router is not hiding from a monitor; the scalar is experimenter-supplied.

**Does this strengthen the paper or feel opportunistic?**

It slightly strengthens *audience* and related-work (oversight allocation, cascades, control-signal robustness) if it stays one paragraph in related work / discussion. It becomes opportunistic if it enters the title, the first abstract sentence, or the contribution list as “AI-control evaluation.” ICLR reviewers in that area will accept the bridge; they will punish a costume change.

Recommendation: use the sentence above in related work. Keep the spine as confidence-metadata-as-control-surface.

---

## 9. Operational hook

Already established (015; matches 012 `fraction_errors_unverified` on GPT code `true_q_visible`):

> In the unmanipulated q1-visible GPT LiveCodeBench condition, 55.6% of incorrect programs were left unverified (95% CI [47.2%, 63.5%]).

This is \( P(\texttt{USE_UNVERIFIED}\mid\text{incorrect}) \). It is **not** an industry defect rate, a DORA comparison, a deployment failure rate, or FOR.

**Where it belongs**

| Location | Recommendation |
|---|---|
| Abstract | Optional **one clause**, not the lead. The lead remains the causal control-surface / leakage result. |
| Introduction | **Yes.** Live paperDirection already has it in the opening paragraph and intro logic. Keep it. |
| Figure 2 callout | **Yes.** Best home: next to the code leakage panel, labeled “unmanipulated q1-visible,” with the CI. |

All three is acceptable only if the abstract clause is short and subordinated. Prefer **introduction + Figure 2**; add the abstract clause if space remains after the +22.0 / +11.8 leakage sentences.

---

## 10. Final verdict

### `USEFUL_RELEASE`

Not `MAJOR_PRESENTATION_UPGRADE`: reviewers will not reclassify D1 from “evaluation / systems study” into “reusable evaluation framework” because of a data dump plus `reproduce_main.py`. The paper’s contribution remains the causal findings (fixed-score leakage, code verifier, rank-preserving attenuation, fixed-objective robustness). ICLR framework papers usually introduce a benchmark, a leaderboard, or a metric others are expected to optimize. D1 should not do that.

Not `NOT_WORTH_EMPHASIZING`: paperDirection already lists a clean artifact as a writing-phase next step. A rights-safe package that reproduces every headline number without APIs, and that documents how to swap router/verifier/task, is standard ICLR hygiene and gives other researchers real zero-call work (λ sweeps, offline policies, heterogeneity). That is worth a reproducibility section and formulation A.

| Question | Answer |
|---|---|
| Changes reviewer perception to “framework paper”? | No. At most: “careful systems eval with a usable release.” |
| Formal contribution bullet? | Optional **secondary** bullet (formulation C). Do not make it Contribution 1–4. |
| Title? | No change. |
| Abstract? | Last sentence / clause only (formulation A). |
| Meaningful work for others? | Yes for reproduction and offline reanalysis; only with new calls for a new LLM router or task. |

No new scientific experiment is justified. After packaging (a later, non-scientific repo task): write the paper.

READY_FOR_GPT_REVIEW = YES
