# 0005 - Verified-leaderboard interface (open item)

- Date: 2026-11-25
- Known: teams submit test predictions on Codabench; organizers re-run the submitted Hugging Face model on the held-out test set for the verified leaderboard and check the generative, at most 5B, no-external-safety-component constraints.
- Unknown (public documentation silent): whether teams supply an inference script, whether a fixed harness or prompt template is used, whether the response column is passed, whether LoRA adapters are accepted or a merged checkpoint is required.
- Status: question sent to organizers; awaiting reply.
- Provisional strategy:
  - Keep the prompt templates, output parsing and deterministic generation config (do_sample=False) versioned in the repository.
  - Final checkpoint must load via AutoModelForCausalLM.from_pretrained.
  - Keep both the LoRA adapter and a merged standalone checkpoint; verify prediction equivalence between them.
  - Keep prompt+response as primary input and --prompt-only as an ablation until organizers confirm the harness input.
