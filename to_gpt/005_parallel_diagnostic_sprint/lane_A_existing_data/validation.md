# Lane A data validation

- Primary cells: **2800**
- Unique questions: **100** (target 100)
- Per-model: `{'anthropic_sonnet5': 1400, 'openai_gpt56_sol': 1400}`
- Per-L: `{'10.0': 1400, '20.0': 1400}`
- Per-condition: `{'hidden': 400, 'true_confidence_visible': 400, 'manipulated_1': 400, 'manipulated_2': 400, 'manipulated_3': 400, 'manipulated_4': 400, 'manipulated_5': 400}`
- Repeat questions: **20** (target 20)
- Repeat cells with three observations: **560** / 560
- Frozen ID hash: `badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94`
- Historical V2 sha256: `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6`
- Historical V2 unchanged during Lane A: **True**
- Study 1 sqlite sha256: `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd`
- paperDirection.txt sha256: `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36`

No material validation issues. Correctness and question-ID joins are reliable.
