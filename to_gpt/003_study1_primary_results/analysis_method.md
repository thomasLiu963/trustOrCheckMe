# Analysis method

Exploratory analysis of the Study 1 primary 2,800-cell dataset.

- Sampling unit: question ID. All within-question conditions are kept together.
- Bootstrap: 5000 question-level resamples, seed 20260917, nominal percentile 95% intervals.
- Paired contrasts resample questions, not isolated cells.
- Verification indicator: 1 if parsed action is VERIFY_FIRST, else 0.
- Mechanical rule: VERIFY_FIRST if (1-q)*L > C with C=1; otherwise USE_UNVERIFIED. Equal-threshold cells therefore USE.
- L=10 mathematical threshold q=0.90; L=20 threshold q=0.95.
- True-visible is a question-specific historical-confidence reference, not a point on the constant-score dose-response curve.
- Hidden/true-visible historical comparison uses V2 AI-system primary-family cells on the same questions, models, and L.
- Repeated-generation stability is not estimated here. Intervals are not confirmatory significance tests.
- Correctness-conditioned contrasts are secondary and likely noisy (n=100 already split).
