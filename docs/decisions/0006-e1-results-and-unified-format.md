# 0006 - E1 results, unified output format, E3 sampling issue

- Date: 2026-11-25
- E1 (full dev, 10 languages, prompt+response; results/tables/e1_baselines.csv): best Task A 0.703 (Qwen2.5-3B-Instruct, 0-shot generate); best Task B 0.331 (Qwen3-4B, 3-shot choice). Reference TF-IDF diagnostic: A 0.922, B 0.540. Zero/few-shot is far below it, so fine-tuning is required.
- Few-shot improves Task B by about 0.05-0.06 and does not help Task A. Qwen3-4B and Qwen2.5-3B are within noise at n=120 per language; backbone is chosen by a fine-tuning pilot, not E1.
- Unified model emits both labels in one structured output (one pass per example) to halve training and inference cost. Prompt omits the category list (learned in fine-tuning). Training uses train only; dev is the validation set.
- Issue flagged: train language sizes are 848-880 (max/min ratio about 1.04), so temperature sampling by size (alpha 0.5, 0.7) is almost identical to uniform and cannot show an effect. E3 is on hold until E2 gives per-language dev errors; a difficulty-aware weighting will then be proposed.
