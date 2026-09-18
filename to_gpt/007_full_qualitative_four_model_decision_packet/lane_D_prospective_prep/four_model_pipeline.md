# Prospective four-model pipeline (prep only)

Do not run this pipeline in Task 007. GPT must freeze N/conditions later.

## Models

- GPT `gpt-5.6-sol` reasoning.effort=none
- Claude `claude-sonnet-5` thinking disabled
- Gemini `gemini-3.8-flash` thinking_level=LOW (frozen V2 setting)
- Grok `grok-4.20-0309-non-reasoning`

## Unseen pool

- Task 006 eligible pool hash `53b11137658de739537a41032d92aada7f50742ef07e9615da10e070ab37c4fa` (11,532 items after excluding 500 V2-B IDs)
- Candidate stratified lists already exist for N=200/400/600/800/1000; none is the confirmatory sample

## Stages after GPT review

1. Fresh Stage-1 answers for all selected models on the same unseen questions
2. Score correctness from benchmark labels
3. Elicit q1
4. Optionally elicit q2 only if retained as a baseline
5. Hidden qualitative verification (Task-006 frozen wording if still the qualitative family)
6. Selected manipulated displayed-score conditions
7. Leave-one-model-out empirical difficulty for evaluation only
8. Matched-budget routing with deployment-fair baselines

Cross-model difficulty is an evaluation-side proxy, not a production router feature unless separately justified.

## Isolation

- New sqlite, not V2 / Study 1 / q2 / qualitative 006/007 databases
- Do not reuse Task-007 100-question IDs as the confirmatory sample

