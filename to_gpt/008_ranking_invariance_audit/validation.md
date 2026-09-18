# Task 008 validation

Zero API calls. Read-only analysis. `paperDirection.txt` was not modified.

## Grid

- Merged qualitative rows: 1600 (expected 1600)
- Questions: 100 (expected 100)
- Models: ['anthropic_sonnet5', 'openai_gpt56_sol']
- Families: ['moderate', 'stronger']
- Conditions: ['displayed_0.70', 'displayed_0.90', 'displayed_0.99', 'hidden']
- Complete: True
- Difficulty leakage rule: GPT difficulty uses Claude/Gemini/Grok Stage-1 correctness; Claude difficulty uses GPT/Gemini/Grok. Target own correctness is never in difficulty.

## Protected fingerprints (before = after)

- `v2`: 8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6 unchanged=True
- `study1`: f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd unchanged=True
- `q2`: 57be401c4f069d5541f3b785e3c51448e49e915727eef178062196ea95272e6a unchanged=True
- `task006_sqlite`: d26ad5d2d40b6f1d98648d05d4940d2fcfd2dc7d1cf0751a38a5b0a19ef8f013 unchanged=True
- `task007_full80_sqlite`: 0006743225492bc175b1e8925200c9523b0935b4c7f342906a26e67f060ba58e unchanged=True
- `task007_gemini_grok_sqlite`: 2a5761184247816bc9b77954f99864fe54b586a042b0e3c3488d4228c3377c6c unchanged=True
- `paperDirection`: d47eb938a831e728200bf32fb7f47ed16664639a2b1c508d3152892772c2b9da unchanged=True
- `merged100`: 018b844a14d384d13e4ce9e48ed00d2c4a05a63955d0b44699eb1c9254b89aff unchanged=True
- `difficulty_features`: 47e10aebcd05ad01cdac9d2c1e9e0d9307cbd2e29a217a551ee0d269ba1dd4ff unchanged=True
- `task003_report`: 4fdfceed3ea9d8b43d73fb16c55bc1db3da04634ad47adf4987afb6f44d71d62 unchanged=True
- `task004_report`: c264830dd5b89c00ce19a6659fe5be31e15aa91ce9b9313e31558306de1c08b0 unchanged=True
- `task005b_report`: 356fd4fc35aaac473fcc5551eb3d3cd6dc517273cf62851ca7de68d14b41daf2 unchanged=True
- `task005c_report`: 2973a2db3fde9773edf211af537825dbd023dd4aac134c97195cbc3361318253 unchanged=True
- `task007_report`: f67bde958f9bb089e8c73098887edd480b284f3358fcc74a9f51863d78af4100 unchanged=True
- `task007_packet`: ae78e7ead2032add2bfea27ace6c8795e9ad777500a396a2264eea3d1761e0f9 unchanged=True
- `checkpoint_a_report`: 1ca6513c8c9796607e12bd11c3d12ecf8c5f1e81b1b817fa67a590333e0d0f1e unchanged=True

## Analysis choices

- Statistical unit: question ID. Bootstrap 5,000, seed 20260917, percentile 95% CIs.
- Unpenalized logistic for low-dimensional models; Rasch item intercepts use L2 C=10 because unpenalized item FE separates.
- Difficulty is leave-one-target-out other-model Stage-1 correctness (0–3).
- Gemini/Grok n=20 are descriptive only.
- No historical sqlite writes. No confirmatory sample frozen.
