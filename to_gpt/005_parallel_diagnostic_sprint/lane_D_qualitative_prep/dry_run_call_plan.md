# Lane D dry-run call plan (not executed)

**API calls made:** 0  
**Authorization:** none — Task 005 forbids running this experiment.

## Roster (if GPT later freezes a prompt)

| factor | value |
|---|---|
| questions | frozen Study-1 100 IDs |
| models | `openai_gpt56_sol`, `anthropic_sonnet5` |
| stakes families | moderate (A), stronger (B) |
| conditions | hidden, 0.70, 0.90, 0.99 |
| L / C | none (qualitative only) |
| authority | AI-system wording |

## Arithmetic

```
100 questions × 2 models × 2 stakes × 4 conditions = 1,600 scientific calls
```

No repeats in this dry-run plan. If GPT later wants stability, that is a separate numbered task.

## Intended comparisons (future, not now)

1. Within family, 0.70 vs 0.90 vs 0.99: does GPT still show a near-step without numerical `L`/`C`?
2. Hidden vs high displayed score: does visibility still suppress verification without an explicit threshold formula?
3. Moderate vs stronger: does stakes intensity move the curve, or only GPT's arithmetic cliff disappear?
4. Claude as contrast, not a required replication.

## Infrastructure that would be required later

- New sqlite path, e.g. `results/study1_qualitative_stakes/` — never `v2.sqlite3`, never Study-1 sqlite.
- Rendered-prompt audit with byte diffs before paid calls.
- Same Stage-3 JSON schema/parser as Study 1.
- Explicit scientific cap 1600 and a provider-attempt cap.

## Explicitly not done in Task 005

- No prompt freeze.
- No smoke test.
- No 1,600 calls.
- No database.

GPT must review `prompt_candidate_A.md`, `prompt_candidate_B.md`, and `prompt_audit.md` before any later numbered task may collect these data.
