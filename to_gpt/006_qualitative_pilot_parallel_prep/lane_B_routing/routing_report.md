# Lane B — exploratory matched-budget routing

Zero API calls. Historical V2-B 500 questions × Study-1 GPT/Claude. Exploratory; same questions motivated the hypotheses.

## Canonical single hidden router

Predeclared `{'decision_owner': 'ai_system', 'L': 10.0, 'confidence_visibility': 'hidden', 'rationale': "Predeclared to match Study 1's AI-system / L=10 hidden cell. Not selected by inspecting correctness or catch rates."}` without peeking at catch rates.

## Error catch at 30% budget vs raw q1

### GPT

- n=500 questions, n_wrong=87
- raw q1 catch@30%: 0.5287356321839081
- q2 catch@30%: 0.5547667342799188 (gain +0.026)
- single hidden AI L=10 catch@30%: 0.5454899668809662 (gain +0.017)
- hidden aggregate catch@30%: 0.6118999323867478 (gain +0.083; not deployment-fair)
- combined q1+q2+single-hidden catch@30%: 0.5300127713920818 (gain +0.001)

### Claude

- n=500 questions, n_wrong=129
- raw q1 catch@30%: 0.528324388789505
- q2 catch@30%: 0.5390578413834227 (gain +0.011)
- single hidden AI L=10 catch@30%: 0.4539662312838484 (gain -0.074)
- hidden aggregate catch@30%: 0.5285412262156448 (gain +0.000; not deployment-fair)
- combined q1+q2+single-hidden catch@30%: 0.5271317829457365 (gain -0.001)
