# Study 1 stability analysis method

Exploratory analysis of the frozen 20-question repeat subset.
The unit of resampling is the question, not the generation.

- Bootstrap seed: 20260917
- Bootstrap resamples: 5000
- Each repeated cell contributes the original primary observation plus two independent extra generations.
- Identical-prompt rerun noise is the probability that a later generation disagrees with the primary generation of the same prompt, averaged within question then across questions/cells.
- Manipulation change is the probability that the low vs high displayed-confidence actions disagree, matched on question and generation index.
- Aggregated response curves average the three binary actions per question, then bootstrap questions.
- Do not treat 60 observations as 60 independent questions.

A PASS does not establish self-provenance, practical agent generalization, a hidden-state mechanism, suppression of internal uncertainty, or novelty relative to all prior literature.
