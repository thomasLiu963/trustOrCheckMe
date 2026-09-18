# Router definitions

Exploratory only. Same 500 historical V2-B questions. Not a frozen confirmatory router.

Canonical single hidden cell: `{"decision_owner": "ai_system", "L": 10.0, "confidence_visibility": "hidden", "rationale": "Predeclared to match Study 1's AI-system / L=10 hidden cell. Not selected by inspecting correctness or catch rates."}`

- **raw_q1:** one Stage-2 confidence report (q1)
- **calibrated_q1:** one Stage-2 confidence report + grouped-CV isotonic fit (not an extra model call)
- **raw_q2:** one independent Stage-2 re-elicitation (q2); conceptually one extra self-assessment call
- **mean_q1_q2:** two confidence reports (q1 and q2)
- **single_hidden_ai_L10:** one verification-decision elicitation in the predeclared hidden AI-system L=10 cell; deployment-fairer than the aggregate
- **hidden_aggregate:** fraction VERIFY across historical hidden owner×L cells; multi-elicitation upper bound / diagnostic, not a one-call practical router
- **combined_q1_q2_single_hidden:** q1 + q2 + one hidden verification judgment; grouped-CV logistic, OOF predicted P(error); three elicitations

Learned/calibrated routers use GroupKFold by question ID. Calibration and logistic fits never see test-question rows.
Ranking uses Checkpoint A fractional inclusion at the cutoff (lowest rank-score first).
Primary metric: fraction of actual Stage-1 wrong answers caught at a fixed verification budget.
