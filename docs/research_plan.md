# Research Plan

**Title:** Language-Aware Multilingual Adaptation for Robust Safety Classification in African Languages: A Study on AfriGuard

**Aim:** Investigate computationally efficient language-aware multilingual adaptation strategies for robust safety classification across African languages using small generative language models.

## Objectives
1. Establish baseline safety classification performance of selected small generative models across the ten AfriGuard languages.
2. Investigate effects of multilingual sampling and cross-lingual transfer across languages with different representation levels.
3. Evaluate hard-negative training for discriminating harmful from benign prompts with similar features.
4. Compare joint multi-task supervision with independent task-specific training (binary safety, ten-category).
5. Evaluate performance, cross-lingual generalisation, robustness and efficiency under the official framework.

## Research questions
- RQ1: How well do existing small generative models classify safety prompts in African languages?
- RQ2: How do multilingual sampling strategies affect performance across languages with different representation?
- RQ3: Does hard-negative training improve harmful/benign discrimination?
- RQ4: Does joint multi-task supervision improve both tasks?
- RQ5: What trade-offs exist between performance, cross-lingual generalisation and computational efficiency?

## Competition constraints
Generative models only (decoder-only or encoder-decoder), at most 5B parameters; no encoder-only models; no external safety classifiers or APIs at inference; official test split never used for training or selection; reproducible submission with Hugging Face link and system description paper. Official documentation takes precedence.

## Schedule
Dev phase Nov 18-Dec 23, 2026; test phase Dec 24, 2026-Jan 5, 2027; paper due Jan 10, 2027; notification Jan 15; camera-ready Jan 17. Reconfirm against official announcements.

## Administrative status
- OpenReview account: setup to be initiated by Bernard (institutional email verification can take time). Status: PENDING.
