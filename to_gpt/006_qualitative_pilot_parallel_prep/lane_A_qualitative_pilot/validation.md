# Lane A validation / preflight

- Exact 20 IDs: see `call_manifest.csv`; hash `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`
- GPT endpoint: `gpt-5.6-sol`
- Claude endpoint: `claude-sonnet-5`
- Moderate prompt hash: `ec90530c644038dcee86accfc578c501533fdb69d5ea23caeae23dcc1d06dbad`
- Stronger prompt hash: `b758935fecdd49ad7a6a0a491d9c71abee35f90981a7030d90ac59a18dca9f3b`
- Exact 8 conditions per question/model: hidden, 0.70, 0.90, 0.99 × moderate/stronger
- Scientific call cap: **320**
- paperDirection.txt will not be changed
- Numerical L, C, expected-value formula, outweighs, justified, and 'not as a default' are absent from paid prompt bodies
- New sqlite: `/Users/thomas/Desktop/trustOrCheckMe/results/study1_qualitative_pilot/qualitative_pilot.sqlite3`
- Preflight ok: **True**
