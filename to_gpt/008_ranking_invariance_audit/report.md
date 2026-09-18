# Task 008 — Ranking-invariance audit

Exploratory only. Not confirmatory. `paperDirection.txt` was not modified. API calls: **0**.

**Decision bucket:** `INVARIANCE_SUPPORTED_EXPLORATORY`
**Favored final-paper world:** WORLD 1 (`MODERATELY_FAVORS`)

READY_FOR_GPT_REVIEW = YES

## 1. Plain-English bottom line

Displayed confidence is a strong coverage control on the N=100 qualitative grid. The adversarial hypothesis is that ranking/discrimination stays put while the checking rate moves. Nesting is mostly consistent with a threshold shift (reversals are rare). A shared item propensity estimated from two visible scores predicts the third better than an intercept. Claude’s probability-scale “sharpening” is not automatically a logit-scale ranking rewrite: moderate Claude γ=5.157 [1.496, 9.614], CV Δlog-loss=-0.007. GPT moderate γ=0.316 [-2.119, 4.053], CV Δlog-loss=0.003. Three binary ROC points cannot prove a unique common ranking. The workshop GPT catch drop remains best read as an operating-point/calibration effect.

## 2. Validation

- Grid complete: True; rows=1600; questions=100
- Difficulty: GPT difficulty uses Claude/Gemini/Grok Stage-1 correctness; Claude difficulty uses GPT/Gemini/Grok. Target own correctness is never in difficulty.
- paperDirection sha256: d47eb938a831e728200bf32fb7f47ed16664639a2b1c508d3152892772c2b9da
- V2 sha256: 8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6
- Study 1 sha256: f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd
- See `validation.md`.

## 3. ROC / risk-coverage geometry

### GPT
- moderate hidden: VERIFY 35.0% [26.0, 44.0]; FPR 27.2%; TPR/catch 68.4%; sat=identifiable
- moderate displayed_0.70: VERIFY 59.0% [49.0, 69.0]; FPR 55.6%; TPR/catch 73.7%; sat=identifiable
- moderate displayed_0.90: VERIFY 20.0% [13.0, 28.0]; FPR 13.6%; TPR/catch 47.4%; sat=identifiable
- moderate displayed_0.99: VERIFY 7.0% [2.0, 12.0]; FPR 3.7%; TPR/catch 21.1%; sat=low_coverage_saturated
- stronger hidden: VERIFY 52.0% [42.0, 62.0]; FPR 48.1%; TPR/catch 68.4%; sat=identifiable
- stronger displayed_0.70: VERIFY 92.0% [86.0, 97.0]; FPR 90.1%; TPR/catch 100.0%; sat=high_coverage_saturated
- stronger displayed_0.90: VERIFY 40.0% [31.0, 50.0]; FPR 33.3%; TPR/catch 68.4%; sat=identifiable
- stronger displayed_0.99: VERIFY 16.0% [9.0, 23.0]; FPR 12.3%; TPR/catch 31.6%; sat=identifiable

### Claude
- moderate hidden: VERIFY 65.0% [56.0, 74.0]; FPR 58.9%; TPR/catch 81.5%; sat=identifiable
- moderate displayed_0.70: VERIFY 90.0% [84.0, 95.0]; FPR 86.3%; TPR/catch 100.0%; sat=high_coverage_saturated
- moderate displayed_0.90: VERIFY 79.0% [71.0, 86.0]; FPR 74.0%; TPR/catch 92.6%; sat=identifiable
- moderate displayed_0.99: VERIFY 59.0% [49.0, 68.0]; FPR 50.7%; TPR/catch 81.5%; sat=identifiable
- stronger hidden: VERIFY 71.0% [62.0, 80.0]; FPR 64.4%; TPR/catch 88.9%; sat=identifiable
- stronger displayed_0.70: VERIFY 93.0% [88.0, 98.0]; FPR 90.4%; TPR/catch 100.0%; sat=high_coverage_saturated
- stronger displayed_0.90: VERIFY 82.0% [74.0, 89.0]; FPR 76.7%; TPR/catch 96.3%; sat=identifiable
- stronger displayed_0.99: VERIFY 67.0% [58.0, 76.0]; FPR 60.3%; TPR/catch 85.2%; sat=identifiable

Figures: `figures/roc_space_gpt_claude.png`, `figures/risk_coverage_gpt_claude.png`.

Common-curve geometry (three points; formal test weak):

- Claude moderate: monotone FPR=True, monotone TPR=True, concave=True, identifiable_points=2/3, adjacent UV=0
- Claude stronger: monotone FPR=True, monotone TPR=True, concave=False, identifiable_points=2/3, adjacent UV=0
- GPT moderate: monotone FPR=True, monotone TPR=True, concave=False, identifiable_points=2/3, adjacent UV=0
- GPT stronger: monotone FPR=True, monotone TPR=True, concave=False, identifiable_points=2/3, adjacent UV=2

## 4. Nesting / reversals

As score rises, nested VERIFY sets require U→V = 0. Nesting supports thresholding; it does not prove invariance.

- GPT moderate 0.7→0.9: VV=20 VU=39 UU=41 UV=0; reversal 0.0 pp [0.0, 0.0]
- GPT moderate 0.9→0.99: VV=7 VU=13 UU=80 UV=0; reversal 0.0 pp [0.0, 0.0]
- GPT moderate 0.7→0.99: VV=7 VU=52 UU=41 UV=0; reversal 0.0 pp [0.0, 0.0]
- GPT stronger 0.7→0.9: VV=40 VU=52 UU=8 UV=0; reversal 0.0 pp [0.0, 0.0]
- GPT stronger 0.9→0.99: VV=14 VU=26 UU=58 UV=2; reversal 2.0 pp [0.0, 5.0]
- GPT stronger 0.7→0.99: VV=16 VU=76 UU=8 UV=0; reversal 0.0 pp [0.0, 0.0]
- Claude moderate 0.7→0.9: VV=79 VU=11 UU=10 UV=0; reversal 0.0 pp [0.0, 0.0]
- Claude moderate 0.9→0.99: VV=59 VU=20 UU=21 UV=0; reversal 0.0 pp [0.0, 0.0]
- Claude moderate 0.7→0.99: VV=59 VU=31 UU=10 UV=0; reversal 0.0 pp [0.0, 0.0]
- Claude stronger 0.7→0.9: VV=82 VU=11 UU=7 UV=0; reversal 0.0 pp [0.0, 0.0]
- Claude stronger 0.9→0.99: VV=67 VU=15 UU=18 UV=0; reversal 0.0 pp [0.0, 0.0]
- Claude stronger 0.7→0.99: VV=67 VU=26 UU=7 UV=0; reversal 0.0 pp [0.0, 0.0]

## 5. Shared-ranking vs condition-sensitive model

Shared: `logit P(VERIFY)=α_condition + u_item` (L2 item intercepts). Richer adds difficulty × score. Transferable comparison is grouped-CV on *new questions*, where u_item cannot be reused.

- GPT moderate: CV intercept 0.495; CV +difficulty 0.430; CV +interaction 0.433; Δ vs difficulty 0.003; material_cv=False
- GPT stronger: CV intercept 0.472; CV +difficulty 0.434; CV +interaction 0.433; Δ vs difficulty -0.001; material_cv=False
- Claude moderate: CV intercept 0.519; CV +difficulty 0.494; CV +interaction 0.488; Δ vs difficulty -0.005; material_cv=False
- Claude stronger: CV intercept 0.462; CV +difficulty 0.442; CV +interaction 0.463; Δ vs difficulty 0.020; material_cv=False

## 6. Logit-scale score × difficulty interaction

Primary difficulty = other-models-correct (0–3). Score centered at 0.90. Ridge C=1 on standardized covariates for coefficients; unpenalized grouped-CV for materiality.

- GPT moderate: β(0.90)=-0.990; γ=0.316 [-2.119, 4.053]; slopes 0.70/0.90/0.99 = -1.053 / -0.990 / -0.961; CV Δlog-loss=0.003; material=False; saturated_cell_present=True
- GPT stronger: β(0.90)=-0.828; γ=2.190 [-0.544, 5.808]; slopes 0.70/0.90/0.99 = -1.266 / -0.828 / -0.631; CV Δlog-loss=-0.001; material=False; saturated_cell_present=True
- Claude moderate: β(0.90)=-0.971; γ=5.157 [1.496, 9.614]; slopes 0.70/0.90/0.99 = -2.003 / -0.971 / -0.507; CV Δlog-loss=-0.007; material=False; saturated_cell_present=True
- Claude stronger: β(0.90)=-1.015; γ=5.288 [0.622, 9.312]; slopes 0.70/0.90/0.99 = -2.072 / -1.015 / -0.539; CV Δlog-loss=0.023; material=False; saturated_cell_present=True

## 7. Condition-specific discrimination

AUROC of the binary VERIFY bit for target wrongness. Coarse by construction.

- GPT moderate displayed_0.70: AUROC 0.591 [0.473, 0.701] sat=identifiable
- GPT moderate displayed_0.90: AUROC 0.669 [0.546, 0.789] sat=identifiable
- GPT moderate displayed_0.99: AUROC 0.587 [0.498, 0.686] sat=low_coverage_saturated
- GPT stronger displayed_0.70: AUROC 0.549 [0.519, 0.583] sat=high_coverage_saturated
- GPT stronger displayed_0.90: AUROC 0.675 [0.551, 0.789] sat=identifiable
- GPT stronger displayed_0.99: AUROC 0.596 [0.485, 0.711] sat=identifiable
- Claude moderate displayed_0.70: AUROC 0.568 [0.532, 0.609] sat=high_coverage_saturated
- Claude moderate displayed_0.90: AUROC 0.593 [0.519, 0.660] sat=identifiable
- Claude moderate displayed_0.99: AUROC 0.654 [0.552, 0.745] sat=identifiable
- Claude stronger displayed_0.70: AUROC 0.548 [0.515, 0.584] sat=high_coverage_saturated
- Claude stronger displayed_0.90: AUROC 0.598 [0.536, 0.656] sat=identifiable
- Claude stronger displayed_0.99: AUROC 0.625 [0.531, 0.709] sat=identifiable

## 8. Cross-condition latent-item prediction

- GPT moderate hold displayed_0.70: u AUROC 0.669 [0.611, 0.732]; OOF LL intercept 0.688 / difficulty 0.615 / shared 0.560 / rich 0.540; rich−shared+diff -0.000
- GPT moderate hold displayed_0.90: u AUROC 0.842 [0.777, 0.903]; OOF LL intercept 0.515 / difficulty 0.434 / shared 0.301 / rich 0.288; rich−shared+diff -0.000
- GPT moderate hold displayed_0.99: u AUROC 0.930 [0.893, 0.963]; OOF LL intercept 0.282 / difficulty 0.258 / shared 0.157 / rich 0.198; rich−shared+diff 0.000
- GPT stronger hold displayed_0.70: u AUROC 0.728 [0.677, 0.780]; OOF LL intercept 0.289 / difficulty 0.246 / shared 0.244 / rich 0.219; rich−shared+diff -0.000
- GPT stronger hold displayed_0.90: u AUROC 0.702 [0.624, 0.775]; OOF LL intercept 0.681 / difficulty 0.630 / shared 0.566 / rich 0.559; rich−shared+diff 0.000
- GPT stronger hold displayed_0.99: u AUROC 0.789 [0.692, 0.871]; OOF LL intercept 0.447 / difficulty 0.421 / shared 0.360 / rich 0.366; rich−shared+diff 0.000
- Claude moderate hold displayed_0.70: u AUROC 0.939 [0.902, 0.972]; OOF LL intercept 0.347 / difficulty 0.295 / shared 0.160 / rich 0.149; rich−shared+diff 0.000
- Claude moderate hold displayed_0.90: u AUROC 0.934 [0.893, 0.967]; OOF LL intercept 0.525 / difficulty 0.497 / shared 0.216 / rich 0.230; rich−shared+diff 0.000
- Claude moderate hold displayed_0.99: u AUROC 0.756 [0.679, 0.830]; OOF LL intercept 0.684 / difficulty 0.676 / shared 0.465 / rich 0.485; rich−shared+diff 0.000
- Claude stronger hold displayed_0.70: u AUROC 0.941 [0.907, 0.973]; OOF LL intercept 0.265 / difficulty 0.229 / shared 0.133 / rich 0.126; rich−shared+diff -0.000
- Claude stronger hold displayed_0.90: u AUROC 0.944 [0.907, 0.973]; OOF LL intercept 0.480 / difficulty 0.530 / shared 0.186 / rich 0.297; rich−shared+diff -0.000
- Claude stronger hold displayed_0.99: u AUROC 0.773 [0.683, 0.857]; OOF LL intercept 0.641 / difficulty 0.638 / shared 0.397 / rich 0.412; rich−shared+diff 0.000

## 9. Does Claude “sharpening” survive latent-scale correction?

Task 007’s Claude hard−easy *probability* gap can widen as coverage falls even with a stable logit slope. On the logit scale, moderate Claude γ CI is [1.496, 9.614] and grouped-CV interaction Δlog-loss is -0.007 (material_logit_interaction=False). High-coverage 0.70 cells are saturated, so probability-scale sharpening there is partly a ceiling. Coefficient materiality now requires CV log-loss gain; a gamma CI excluding 0 is not enough under saturation.

## 10. Workshop error-catching reinterpretation

See `workshop_reinterpretation.md`. GPT visibility mostly reduces coverage and tracks the raw-q threshold; matched-budget ranking advantage is uncertain. Calibration explains high-L cost. Claude hidden ranking is the exception.

## 11. Current decision bucket

`INVARIANCE_SUPPORTED_EXPLORATORY`

- GPT: class=invariance; invariance_votes=5; reshape_votes=0; max_reversal=2.0 pp; mean u AUROC=0.777
- Claude: class=invariance; invariance_votes=5; reshape_votes=0; max_reversal=0.0 pp; mean u AUROC=0.881

## 12. Final-paper narrative favored by current evidence

WORLD 1 at strength `MODERATELY_FAVORS`. Full prose in `final_narrative_decision_packet.md`.

## 13. What the prospective study must prove or falsify

- World 1: coverage moves; reversals stay near rerun noise; shared u_i predicts new scores; condition-sensitive models add <0.01 log-loss; matched-coverage catch is stable.
- World 2: at identifiable operating points, ranking/discrimination changes enough to beat those tests.
- World 3: the GPT vs Claude split is predeclared and replicates on fresh items.

## 14. Recommended protocol shape (DO NOT EXECUTE)

- One stakes family (moderate).
- Prefer P2 if the goal is to test invariance (more operating points); P1 if budget-bound.
- Primary GPT+Claude at N=500; Gemini/Grok on 200 of the same items.
- Fresh Stage 1 + q1 mandatory. Drop q2.
- Repeats: 100 items × 3 generations/cell.
- Do not freeze a sample in this task.
- Exact counts: `prospective_design_matrix.csv`.

## 15. READY_FOR_GPT_REVIEW

READY_FOR_GPT_REVIEW = YES

Generated 2026-09-18T09:55:43.261997+00:00
