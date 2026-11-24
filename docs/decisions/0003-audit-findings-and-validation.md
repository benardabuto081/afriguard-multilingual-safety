# 0003 - Audit findings and implications

- Date: 2026-11-25
- Both task labels exist for every item: no task-label masking needed for the multi-task model.
- Safe items exist in every category: in-data hard negatives are available; synthetic data is not needed to start E4.
- Per-language category priors differ: report per-language results; watch for language-to-category shortcuts.
- Validation: official dev (120 per language) is small. Whether a random held-out split from train is leak-free is tested empirically (CV on train vs train-to-dev score).
- Input format (prompt only vs prompt+response) is an implementation choice and is tested with a cheap reference baseline before fine-tuning.
