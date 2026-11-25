# 0008 - Backbone selection from the pilot

- Date: 2026-11-25
- Pilot: 150 train examples per language (1,500), 1 epoch (93 updates), unified output, full dev (10 x 120), seed 42, prompt+response.
  - Qwen2.5-3B-Instruct: A 0.8915, B 0.3787. Qwen3-4B: A 0.8949, B 0.3799. Tiny Aya Earth (3.35B): A 0.9048, B 0.5053.
- Task B gap of about 0.125 for Tiny Aya is far outside the roughly +/-0.015 noise of a 1,200-item average; Task A differences are within noise; Qwen2.5 vs Qwen3 is a tie.
- Gains concentrate in languages listed on the Tiny Aya model card; no gain in Oromo or Twi (not seen on the card). Hypothesis only.
- Decision: Tiny Aya Earth is the E2 backbone. Licence is CC-BY-NC-4.0 (gated; acceptable-use addendum). Qwen3-4B (Apache-2.0) remains the licence-safe fallback. Organizer question on NC bases is open.
- Tiny Aya's default chat template inflates training sequences (mean 696 vs about 300 tokens of content); a one-line system message replaces it from E2 onwards. Earlier pilot used the default template.
- Pipeline checks: structured output parses (unparsed about 0); official validator and scorer reproduce our metrics on the pilot submission (see run log).
