# 0007 - Backbone pilot, leaderboard reading, open licence question

- Date: 2026-11-25
- Smoke test (Qwen3-4B QLoRA, T4): finite loss, peak 8.66 GB, about 345 tokens/s; projected about 2.8 h per full epoch. Gemma 3 4B 0-shot produced 100% unparsed output in fp16 (diagnosis pending); its tokenizer is about 25-47% more token-efficient on these languages.
- Leaderboard (dev phase, 2026-11-24): top Task A 0.9366, top Task B 0.6787; our TF-IDF diagnostic (A 0.922, B 0.540) matches the board's TF-IDF entries, so it is scored on dev with the same metric. Task A is saturated; Task B separates teams. Dev labels are public, so board scores may include dev-trained models.
- Decision: run a backbone pilot (150 per language, 1 epoch, full dev eval) on Qwen3-4B, Qwen2.5-3B-Instruct and CohereLabs/tiny-aya-earth (3.35B, CC-BY-NC-4.0, gated); add Gemma 3 4B only if it runs stably. Select the long-run backbone from the pilot. Sampling experiment E3 stays on hold (alpha 0.5/0.7 differ from uniform by at most 1.6% at these language sizes).
- Open: whether a CC-BY-NC base model is acceptable for the submission/prizes and the released weights; how rankings combine Tasks A and B.
