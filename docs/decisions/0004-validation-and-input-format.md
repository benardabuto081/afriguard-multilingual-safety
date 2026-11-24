# 0004 - Validation protocol and input format

- Date: 2026-11-25
- Evidence: CPU reference baseline (TF-IDF + linear SVM, diagnostic only; results/tables/reference_baseline.csv).
  - Task A dev weighted F1 (mean of 10 languages): prompt-only 0.834, response-only 0.914, prompt+response 0.922.
  - Task B dev: prompt-only 0.458, response-only 0.510, prompt+response 0.540.
  - 5-fold CV on train minus dev score (prompt+response): Task A 0.017, Task B 0.088.
- Decision 1: model input is prompt + response (test.csv contains both). Prompt-only is reported as an ablation.
- Decision 2: official dev (1,200 items) is the primary validation set for all comparisons. Train only on train during experiments; refit on train+dev for the final model once configurations are frozen. Random CV on train is not used for selection (Task B optimism of about 0.09; scenario-sibling leakage is a hypothesis, not proven).
- Decision 3: output the plain label string so the model is usable under an unknown evaluation harness. Organizer question about the harness is open.
