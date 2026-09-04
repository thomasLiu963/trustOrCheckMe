# Pilot evidence summary

This report presents observed evidence only; it does not assign an automatic GO/MODIFY/KILL conclusion.

## Coverage

- anthropic_sonnet5: 200 scored answers; accuracy 0.735; Brier 0.168; monotonicity violation rate 0.020.
- openai_gpt56_sol: 200 scored answers; accuracy 0.845; Brier 0.148; monotonicity violation rate 0.095.
- Requests: answer / stage_1_answer_v1 / anthropic_sonnet5 / failed = 18.
- Requests: answer / stage_1_answer_v1 / anthropic_sonnet5 / success = 182.
- Requests: answer / stage_1_answer_v1 / openai_gpt56_sol / success = 200.
- Requests: answer / stage_1_answer_v2_structured / anthropic_sonnet5 / success = 200.
- Requests: answer / stage_1_answer_v2_structured / openai_gpt56_sol / success = 200.
- Requests: confidence / stage_2_confidence_v1 / anthropic_sonnet5 / success = 1.
- Requests: confidence / stage_2_confidence_v1 / openai_gpt56_sol / success = 1.
- Requests: confidence / stage_2_confidence_v2_structured / anthropic_sonnet5 / failed = 1.
- Requests: confidence / stage_2_confidence_v2_structured / openai_gpt56_sol / success = 1.
- Requests: confidence / stage_2_confidence_v3_structured_compat / anthropic_sonnet5 / success = 200.
- Requests: confidence / stage_2_confidence_v3_structured_compat / openai_gpt56_sol / success = 200.
- Requests: trust / stage_3_direct_trust_v1 / anthropic_sonnet5 / success = 4.
- Requests: trust / stage_3_direct_trust_v1 / openai_gpt56_sol / success = 4.
- Requests: trust / stage_3_direct_trust_v2_structured / anthropic_sonnet5 / success = 800.
- Requests: trust / stage_3_direct_trust_v2_structured / openai_gpt56_sol / success = 800.
- Observed provider cost from configured pricing metadata: $4.1901.

## Stake-sensitive evidence

- anthropic_sonnet5, L=2.000: VERIFY rate 0.425 [0.355, 0.495]; unsafe reliance 0.170; mean regret 0.250.
- anthropic_sonnet5, L=5.000: VERIFY rate 0.450 [0.380, 0.520]; unsafe reliance 0.151; mean regret 0.385.
- anthropic_sonnet5, L=10.000: VERIFY rate 0.475 [0.405, 0.545]; unsafe reliance 0.132; mean regret 0.560.
- anthropic_sonnet5, L=20.000: VERIFY rate 0.530 [0.460, 0.600]; unsafe reliance 0.094; mean regret 0.765.
- openai_gpt56_sol, L=2.000: VERIFY rate 0.140 [0.095, 0.190]; unsafe reliance 0.677; mean regret 0.195.
- openai_gpt56_sol, L=5.000: VERIFY rate 0.175 [0.125, 0.230]; unsafe reliance 0.581; mean regret 0.470.
- openai_gpt56_sol, L=10.000: VERIFY rate 0.170 [0.120, 0.225]; unsafe reliance 0.645; mean regret 1.015.
- openai_gpt56_sol, L=20.000: VERIFY rate 0.175 [0.125, 0.230]; unsafe reliance 0.581; mean regret 1.820.

## Policy evidence

Policy rows and paired examples are supplied in the machine-readable outputs. Calibrated-policy estimates use evaluation-partition examples only; their fitted labels come exclusively from calibration IDs.
- Lowest observed mean cost for anthropic_sonnet5, L=2: oracle (0.275); this is descriptive, not a significance claim.
- Lowest observed mean cost for anthropic_sonnet5, L=5: oracle (0.275); this is descriptive, not a significance claim.
- Lowest observed mean cost for anthropic_sonnet5, L=10: oracle (0.275); this is descriptive, not a significance claim.
- Lowest observed mean cost for anthropic_sonnet5, L=20: oracle (0.275); this is descriptive, not a significance claim.
- Lowest observed mean cost for openai_gpt56_sol, L=2: oracle (0.169); this is descriptive, not a significance claim.
- Lowest observed mean cost for openai_gpt56_sol, L=5: oracle (0.169); this is descriptive, not a significance claim.
- Lowest observed mean cost for openai_gpt56_sol, L=10: oracle (0.169); this is descriptive, not a significance claim.
- Lowest observed mean cost for openai_gpt56_sol, L=20: oracle (0.169); this is descriptive, not a significance claim.
- anthropic_sonnet5, L=2.000, direct minus raw confidence: 0.020 [-0.065, 0.100], paired n=200.
- anthropic_sonnet5, L=2.000, direct minus calibrated confidence: 0.000 [-0.094, 0.094], paired n=160.
- anthropic_sonnet5, L=5.000, direct minus raw confidence: -0.260 [-0.410, -0.120], paired n=200.
- anthropic_sonnet5, L=5.000, direct minus calibrated confidence: -0.281 [-0.463, -0.113], paired n=160.
- anthropic_sonnet5, L=10.000, direct minus raw confidence: -0.100 [-0.345, 0.165], paired n=200.
- anthropic_sonnet5, L=10.000, direct minus calibrated confidence: -0.519 [-0.900, -0.163], paired n=160.
- anthropic_sonnet5, L=20.000, direct minus raw confidence: 0.075 [-0.310, 0.550], paired n=200.
- anthropic_sonnet5, L=20.000, direct minus calibrated confidence: 0.275 [-0.188, 0.863], paired n=160.
- openai_gpt56_sol, L=2.000, direct minus raw confidence: 0.030 [-0.010, 0.075], paired n=200.
- openai_gpt56_sol, L=2.000, direct minus calibrated confidence: 0.013 [-0.044, 0.069], paired n=160.
- openai_gpt56_sol, L=5.000, direct minus raw confidence: -0.075 [-0.195, 0.035], paired n=200.
- openai_gpt56_sol, L=5.000, direct minus calibrated confidence: -0.200 [-0.375, -0.037], paired n=160.
- openai_gpt56_sol, L=10.000, direct minus raw confidence: 0.025 [-0.200, 0.240], paired n=200.
- openai_gpt56_sol, L=10.000, direct minus calibrated confidence: 0.206 [-0.231, 0.681], paired n=160.
- openai_gpt56_sol, L=20.000, direct minus raw confidence: 0.160 [-0.350, 0.710], paired n=200.
- openai_gpt56_sol, L=20.000, direct minus calibrated confidence: 0.963 [0.119, 1.856], paired n=160.

## Interpretation checklist

- Examine effect sizes and confidence intervals, not only point estimates.
- Check missingness and denominator counts before comparing policies.
- Treat oracle results as non-deployable lower bounds.
- Review internal diagnostics before making the research decision.
