# 0009 - E2 interruption, template erratum, compact-template pilot

- Date: 2026-11-25
- E2 (Tiny Aya Earth, unified output, 2 epochs, 8,607 train examples, default template): reached about update 790 of 1,076 when the 12-hour Kaggle session ended; checkpoint at update 750 backed up to a private Hugging Face repo and resumed in a new session. Observed: 407-420 tokens/s, loss about 0.07-0.10 at updates 400-790, no non-finite steps.
- First resume attempt used wrong run_ft.sh arguments (output directory not under experiments/multitask) and would have restarted from scratch; caught and relaunched correctly.
- Erratum to decision 0008: the one-line system message did NOT shorten Tiny Aya prompts. Cohere's chat template always inserts its default system preamble (about 400 tokens); the check printed 429 tokens (default) vs 470 (with our system message), and training sequences averaged 735 tokens. The user text alone is about 316 tokens on average.
- Plan: add a compact template (Cohere special tokens, no preamble) and run a pilot-scale comparison (abl_joint vs abl_joint_compact, same data and seed). If accuracy holds, later runs use the compact template at roughly half the cost; the current E2 remains the full-data reference.
