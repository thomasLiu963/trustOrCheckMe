# Task 011 benchmark validation

Label: **Phase A**, zero primary-model scientific calls.

## Source

- Official repo: `https://github.com/LiveCodeBench/LiveCodeBench`
- Pinned commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` (expected pin `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`)
- HuggingFace dataset: `livecodebench/code_generation_lite`
- Release: `release_v6` (newest official lite snapshot in this environment; files test.jsonl–test6.jsonl)
- Evaluator: LiveCodeBench `lcb_runner.evaluation.compute_code_generation_metrics.check_correctness` wrapping `testing_util.run_test`

Lite is the official default harness (`code_generation_lite`). Hidden private tests remain hidden from the model. We do **not** claim contamination-free beyond LCB's contest-date methodology (problems from 2023-05-07 through 2025-04-06).

## Pool

- Release_v6 problems: 1055
- Difficulty (all): {'easy': 322, 'medium': 383, 'hard': 350}
- Eligible medium+hard after official ERRATA exclusions: **723**
- Eligible platforms: {'leetcode': 309, 'atcoder': 409, 'codeforces': 5}
- Eligible ID list sha256: `c73475ca426c2b30026185a62c5b9b813ebb1bfbd2647946726523bfe8c74e4c`
- Status: **POOL_OK_SAMPLE_500**

Eligibility: code-generation tasks, Python-compatible official tests, difficulty in {medium, hard}, not listed in official `ERRATA.md`. No problem was dropped because a model would fail it.

## Harness check

Problem `1873_A` (easy stdin, public tests only):

- public-test echo program passed: True
- wrong program passed: False
- syntax-error program passed: False
- infinite-loop / timeout passed: False
- harness_ok: **True**

Isolation: {'process_wrapper': 'LiveCodeBench check_correctness multiprocessing.Process + SIGALRM', 'env_secrets_stripped': True, 'timeout_seconds': 6, 'memory_guard': 'LCB reliability_guard ~4GB', 'network': 'not granted; child inherits stripped env without API keys'}

Generated code is untrusted. Evaluation runs in a child process with API keys stripped from the environment, LCB's 4GB reliability guard, and wall-clock timeout. Host secrets are not passed through.

## Decision

Proceed to the excluded 40-problem pilot.
