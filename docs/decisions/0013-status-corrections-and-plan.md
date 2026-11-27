# 0013 - Corrections to the 2026-11-27 status report, coverage facts, plan

- Date: 2026-11-27
- Dev submission #1070920 (prompt-only, E2p, template t1): official A 0.8865 / B 0.5864, identical to local scoring; rank 12 of 12 on the dev board. The board is not comparable: entries 9-11 (TF-IDF and organizer baseline) are consistent with response use (our TF-IDF reference: 0.922 with response, 0.834 without); for LLM entries it is unknown.
- Not established: that ranks 1-8 used the response; that Afq_best/afq_lora use AfriqueQwen (inferred from names). The one explicit AfriqueQwen entry (1 epoch) scored 0.9308/0.6154 vs TinyAya_QLoRA 0.9358/0.6203.
- Language coverage (AfriqueLLM paper, arXiv 2601.06395): continued pretraining on 20 African languages including Amharic, Hausa, Igbo, Oromo (93M tokens, upsampled 5x), Shona, Swahili, Yoruba, Zulu; Wolof (18M) and Twi (10.6M) excluded (<90M tokens). Tiny Aya's preamble lists Amharic, Hausa, Igbo, Malagasy, Shona, Swahili, Wolof, Xhosa, Yoruba, Zulu (no Oromo, no Twi). Paper's Qwen results are for 8B/14B; the Qwen3.5-4B models are not described there.
- AfriqueQwen3.5-4B-Instruct v1, 50Langs-Instruct v1 and v2: licence cc-by-4.0, 4,205,751,296 parameters (checked via model_info).
- E2p weaknesses (dev, held-out-template mean): Oromo (A 0.816, B 0.390) and Twi (A 0.823, B 0.490); Task B categories Ethical & Moral Risk (precision 0.287) and Systemic Risks (F1 0.383); Task A false positives on Hate/Advice/IP, misses on Ethical & Moral Risk.
- Calibration: oracle 5-fold class-bias gain +0.006 (A), +0.011 (B); not deployable under the organizers' harness. Reported as analysis only.
- Confident errors (p>=0.9): 41 of 1,200 Task A dev items, spread across languages; not concentrated where the forum reports mislabelled train items. No relabeling.
- Plan: (1) CPU merge, numeric spot check and private HF push of the e2p adapter; (2) AfriqueQwen smoke test and pilot (t1, h2, h4 only) against the matched Tiny Aya pilot; (3) choose backbone by mean and per-language results; (4) optional language/cell upsampling flags (E3/E4) at full scale; (5) final refit on train+dev with frozen configuration; (6) verified-harness-style check of the merged checkpoint.

## Addendum (2026-11-27, after the CPU session)
- Merge of e2p_full (CPU, fp32 merge, fp16 save): 3.349B parameters, single 6.7 GB safetensors; pushed to private repo Abuto/afriguard-tinyaya-e2p. Spot check on 6 Swahili dev items x 2 tasks: argmax agreement 12/12, median |log-prob diff| 0.15-0.18, max 1.08 (adapter run used a 4-bit base). Full dev evaluation of the merged model on GPU is still pending.
- Tokenisation: "safe" is 1 token and "unsafe" 2 tokens under t1, h1, h2 and h4 (Tiny Aya tokenizer). With per-token mean scoring and a near-certain second token, "unsafe" wins whenever P(safe) < 0.62, which explains the collapse on h4 (0.4238 vs 0.8714 summed) where the model is least confident, and not on confident chat templates.
- Stress templates s1 (letter answers), s2 (Yes/No), s3 (no BOS) added for evaluation only (never trained).
