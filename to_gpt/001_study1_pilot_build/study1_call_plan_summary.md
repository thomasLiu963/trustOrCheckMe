# Study 1 call-plan summary

Offline plan only. No model API calls were made.

- Primary calls: **2800** (required 2800)
- Repeat-extra calls: **1120** (required 1120)
- Total possible later pilot: **3920** (required 3920)
- Models: `openai_gpt56_sol, anthropic_sonnet5`
- Authority: AI-system only
- L: 10 and 20; C = 1
- L=10 grid: [0.8, 0.88, 0.89, 0.91, 0.99]
- L=20 grid: [0.9, 0.93, 0.94, 0.96, 0.99]
- L=10 local contrast: 0.89 vs 0.91
- L=20 local contrast: 0.94 vs 0.96
- Primary ID list SHA-256: `badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94`
- Repeat ID list SHA-256: `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`
- Study 1 checkpoint path: `/Users/thomas/Desktop/trustOrCheckMe/results/study1_causal_pilot/study1.sqlite3`
- Historical checkpoint path: `/Users/thomas/Desktop/trustOrCheckMe/results/v2/raw/v2.sqlite3`
- Historical write target: no

## Prompt-audit presentation IDs

These two IDs were chosen only to display high vs lower GPT reported confidence.
They do not change the frozen 100-question experimental sample.
They were not chosen using correctness or historical Stage-3 actions.

- High GPT reported confidence: `mmlu_pro:test:10539` (0.999)
- Lower GPT reported confidence: `mmlu_pro:test:774` (0.03)

Historical wording source: `src/v2_prompts.py:build_verification_prompt AI-system primary (v2_owner_match_v1, paraphrase=False)`

Confidence grids JSON: `{"10": [0.8, 0.88, 0.89, 0.91, 0.99], "20": [0.9, 0.93, 0.94, 0.96, 0.99]}`
