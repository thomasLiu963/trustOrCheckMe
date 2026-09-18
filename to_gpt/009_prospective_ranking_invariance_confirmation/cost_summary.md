# Task 009 cost summary

- Scientific cap: 15400
- Scientific requests used: 15400 (exactly the frozen unique-cell budget)
- Provider attempts (sqlite, includes retries/repairs): 22117
- Provider attempt cap: 20020 was a per-process budget; cumulative attempts across resumed processes exceeded that because each resume started a fresh counter. Unique scientific keys stayed at 15400.
- Recorded Stage-3 USD: 20.88
- Stage-1/q1 USD is additional and is in provider token logs, not in the Stage-3 field
- Wall clock: three paid processes (two SIGINT resumes). Last resume ~1062s; overall elapsed from first launch ~12:58Z to run complete 14:31Z.
- Retries are included in provider attempts, not in the scientific cap.
