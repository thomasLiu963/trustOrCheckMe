# Task 010 detection floor

The Task-009 analysis could reliably detect reranking of approximately the magnitudes below under each simulated alternative family.

Power uses the **full Task-009 rule** (held-out improvement ≥ 0.01 or question-bootstrap 95% UB ≥ 0.01). Point-estimate-only power is lower and is in `positive_control_power_curve.csv`. False-positive rate at zero injection: GPT 0.000, Claude 0.000.

## GPT

- `score_x_difficulty`: 80% power first reached at magnitude=7.735 (label=1.5× Claude-008 γ), power=0.97, mean latent adjacent Spearman=0.500, mean pairwise reversal=0.250. 90% power is also reached at this cell. At 1.0× (γ=5.157) power is only 0.214 and Spearman is still 1.00, because binary wrongness does not reorder items until the interaction flips the two groups.
- `generic_item_noise`: 80% power **not reached** on the simulated grid. Realized adjacent Spearman was already ~0.55 at the smallest noise and power remained 0.000.

Plain sentence: the Task-009 analysis could reliably detect GPT score×difficulty reranking of approximately **1.5× the Claude-008 development γ (γ≈7.73), a large Spearman-0.50 rewrite**, or larger. It could not reliably detect smaller score×difficulty changes, and it did not detect generic item–condition noise at all on this grid.

## Claude

- `score_x_difficulty`: 80% power **not reached** on the simulated grid. At 2.0× (γ=10.31), power=0.518, Spearman=0.50, reversal=0.25.
- `generic_item_noise`: 80% power **not reached**. Realized Spearman ~0.41–0.42, power=0.000.

Plain sentence: the Task-009 analysis could **not** reliably detect Claude reranking of the magnitudes simulated here, including a large Spearman-0.50 score×difficulty rewrite at 2× the development anchor.

## Shared template

Predeclared wording category justified: **WEAK**.
