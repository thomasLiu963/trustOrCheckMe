# Task 005C validation / leakage audit

- API calls: **0**
- OK: **True**
- Rows: 1000 (expected 1000 = 500 GPT + 500 Claude)
- q2 table: `/Users/thomas/Desktop/trustOrCheckMe/to_gpt/005_parallel_diagnostic_sprint/lane_B_q2/q2_results.csv` (read-only)
- Canonical single hidden action: `ai_system L=10 hidden`
- Unit: question ID
- Bootstrap seed 20260917, resamples 5000

## Frozen-file fingerprints (unchanged if listed)

- v2: `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6`
- study1: `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd`
- q2: `57be401c4f069d5541f3b785e3c51448e49e915727eef178062196ea95272e6a`
- paperDirection: `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36`
- task005_report: `a536d304a7d2581087fa1daba5e9ed92979f488f2348925142b79e28d8f06c69`
- task005b_report: `356fd4fc35aaac473fcc5551eb3d3cd6dc517273cf62851ca7de68d14b41daf2`

## Leakage rules

- Target-model Stage-1 correctness is not inside `other_n_correct`.
- Difficulty uses the other three historical V2-B models only.
- Study-1 manipulated actions are not used.
- q1/q2/hidden/difficulty preprocessing for CV is fit on training folds only.
- Hidden fraction is rebuilt from historical V2 primary hidden cells and checked against Task 005 `q2_results.csv`.

## Errors

- none

