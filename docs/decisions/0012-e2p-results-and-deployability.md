# 0012 - E2p results, organizer clarifications, what can be deployed

- Date: 2026-11-26
- Evidence: shared-task page lists the input of both tasks as "User prompt" only; forum answers archived in docs/organizer_forum_2026-11-26.md (prompt-only final evaluation; organizers' own log-likelihood scoring under a variety of templates; separate models allowed; test items 3-of-3 agreement; responses out of scope). Merged-vs-adapter question unanswered; "usable like tiny aya" implies a standard Transformers checkpoint.
- E2p full (Tiny Aya Earth, prompt only, 8-template bank, 2 epochs, train split only; dev; 4 held-out templates): Task A 0.8819, Task B 0.5738; spread over all 6 templates 0.0176 (A) and 0.0195 (B). Prompt-only TF-IDF reference: A 0.834, B 0.458. With-response E2 (0.9291 / 0.6252) is not the test condition.
- Template augmentation (pilot scale, 150 per language, 1 epoch): single fixed template collapses on held-out raw templates (B held-out mean 0.263, min 0.074) vs template bank (0.462, min 0.432).
- Joint vs single-task (pilot): unresolved. The multi-task pilot had 187 updates vs 93 for single-task (confounded) and per-template ranks flip.
- Risk: length-normalised scoring (mean log-prob per token) collapses Task A on raw title-case template h4 (0.4238 vs 0.8714 with summed log-probs). Probable cause: multi-token "Unsafe" vs single-token "Safe" (to be confirmed by tokenisation). Organizers' normalisation unknown.
- Calibration: the organizers compute log-likelihoods, so post-hoc calibration cannot be applied by us at evaluation time; usable only for Codabench label files. Gains must be put into the weights (for example class weighting).
- Label noise: other participants report about 136 Task A items labelled safe with refusal responses (113 in train; about 74 look mislabelled). About 1% of train; no manual relabeling; model-based filtering is optional and low priority.
- Next: merge and verify an fp16 checkpoint; AfriqueQwen3.5-4B smoke test (licence, parameter count, T4 feasibility); CPU analysis of per-language/category errors and calibration upper bound; dev submission from E2p log-probs.
