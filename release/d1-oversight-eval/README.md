# D1 fixed-output oversight evaluation package

This is a **reproducibility and measurement package** for the D1 study of confidence metadata as a control surface for resource-bounded LLM verification.

It is **not** a leaderboard, **not** a new benchmark, and **not** a new generic confidence-sensitivity metric.

No model calls are required to reproduce the headline tables and figures.

## Procedure

D1 measures how an externally supplied confidence-like signal changes verification routing and downstream error leakage while the generated output is held fixed.

1. **Freeze an output.** Generate an answer or program once. Later stages cannot edit it.
2. **Obtain a confidence source.** D1 uses a separately elicited \(q_1 \approx P(\text{frozen output is correct})\).
3. **Expose a router** to controlled confidence metadata, with the frozen output still visible:
   - hidden (no scalar);
   - true-\(q_1\)-visible (unmanipulated);
   - fixed score (same displayed number on every item in the condition);
   - optional rank-preserving shift \(q_\delta=\mathrm{sigmoid}(\mathrm{logit}(q_1)+\delta)\) on \((0,1)\).
4. **Record** `USE_UNVERIFIED` or `VERIFY_FIRST`.
5. **Evaluate with an independent verifier** that does not use the displayed number (MMLU-Pro letter match; official LiveCodeBench hidden suite).
6. **Compute**
   - coverage \(=P(\texttt{VERIFY_FIRST})\)
   - leakage \(=P(\text{incorrect and }\texttt{USE_UNVERIFIED})\)
   - conditional escape \(=P(\texttt{USE_UNVERIFIED}\mid\text{incorrect})\)
   - normalized loss \(=\mathrm{leakage}+\lambda\cdot\mathrm{coverage}\)
7. **Optionally** apply the rank-preserving \(q_\delta\) family to separate confidence *level* from item discrimination carried by \(q_1\).

See `DATA_CARD.md` for what is preregistered vs post-hoc, and `LICENSES.md` for redistribution limits.

## Reproduce headline results

```bash
python -m pip install numpy matplotlib
python reproduce_main.py
```

This reads only `data/item_level_routing.csv` and `data/item_level_offset.csv`. It regenerates derived tables, the main figures, and fails if headline values disagree materially with the frozen reports.

## Offline example

```bash
python examples/evaluate_threshold_router.py --task code --model openai_gpt56_sol --condition true_q_visible --lambda-cost 0.20
```

That script does **not** call a model. It scores a threshold rule `VERIFY iff (1 - displayed_q) > λ` on the frozen labels.

## Using D1 with a New Router

**Entirely offline (no API / model calls)**

- Reproduce every headline number in this package.
- Recompute cost curves for any \(\lambda\).
- Evaluate any routing rule that is a function of released columns (`q1`, displayed score, \(\delta\), \(q_\delta\), item id, model, task) against the frozen correctness labels.
- Compare models/tasks, calibration of \(q_1\) vs correctness, and fixed-score vs rank-preserving tables.

**Requires new API / model / verifier calls**

- A new generator (new frozen outputs).
- A new black-box LLM router: fill `prompts/stage3_*.txt` with *your* item text and displayed score, call your model, parse `USE_UNVERIFIED` / `VERIFY_FIRST`, then join to labels.
- A new task, a new confidence source, or a new verifier if you do not accept the released pass/fail flags.
- Re-running LiveCodeBench hidden tests (not included here; obtain tests from the official LiveCodeBench channel).

To attach a new LLM router to the *same* frozen items you still need the original question text and frozen outputs, which this package deliberately omits. Reconstruct those from the official MMLU-Pro / LiveCodeBench pins in `data/manifests/` if your licenses allow.

## What is included

Item IDs, dataset pins, model endpoints, \(q_1\), displayed / \(q_\delta\) scores, routing actions, correctness / pass-fail, LCB `n_tests` / `n_passed`, prompt templates with placeholders, freeze/preregistration manifests, and analysis code.

## What is not included

MMLU-Pro stems/options/answer keys; LiveCodeBench problem statements, starter code, hidden tests, and generated programs; raw SQLite; filled prompts; credentials; Task 013; historical V1/V2 runs.
