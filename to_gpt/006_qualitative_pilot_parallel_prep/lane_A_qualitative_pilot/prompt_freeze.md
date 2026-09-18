# Lane A prompt freeze

Paid templates are the Task-006 frozen qualitative families, not the Task-005 Lane D drafts.

- Prompt version: `task006_qualitative_pilot_v1`
- Moderate family: `qualitative_stakes_moderate_v1` hash `ec90530c644038dcee86accfc578c501533fdb69d5ea23caeae23dcc1d06dbad`
- Stronger family: `qualitative_stakes_stronger_v1` hash `b758935fecdd49ad7a6a0a491d9c71abee35f90981a7030d90ac59a18dca9f3b`
- Visible tokens: `{'0.70': '0.7', '0.90': '0.9', '0.99': '0.99'}` via `format(value, '.12g')`
- Frozen 20 IDs hash: `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`
- Within a family, visible prompts differ only in the displayed confidence number.
- Moderate vs stronger differs only in the exact wrong-answer-consequence sentence.
- Hidden omits the entire confidence sentence.
- Forbidden numerical/EV tokens were audited on the wrapper after stripping question/choice text.

## Moderate consequence

`If the frozen answer is wrong and is not checked, it may cause an incorrect downstream decision on this task.`

## Stronger consequence

`If the frozen answer is wrong and is not checked, it may cause substantial downstream consequences that are difficult to reverse.`
