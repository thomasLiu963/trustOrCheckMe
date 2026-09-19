# Task 011 deviations

## Sampler bug and second excluded pilot

The first excluded pilot (seed `20260921`) used a round-robin over `difficulty|platform|month` keys sorted lexicographically, so `hard` filled the entire N=40 before any `medium` item was drawn. That sample is permanently excluded. Stratification was corrected to allocate proportionally by difficulty and then by platform. A second excluded pilot (seed `20260924`) was drawn from the remaining pool. This is an implementation fix, not prompt tuning.

## Predeclared hard-only confirmatory N

GPT frozen-code accuracy on the mixed second pilot was 87.5% (>80%), so the predeclared ladder is hard-only. After excluding both pilots, 286 hard problems remain (<300). The confirmatory sample uses the entire remaining hard pool rather than mixing medium back in.

## Engineering

Concurrency is code=24 / route=24 / 12 per provider; hidden tests use 16 worker processes. Code generation uses unstructured 4096-token completions because models.yaml's 64-token JSON cap cannot emit programs. These choices do not change prompts, endpoints, or analyses.
