# System paper outline (working notes; not a submission draft)

Working title: Language-Aware Multilingual Adaptation for Robust Safety Classification in African Languages: A Study on AfriGuard

Claims and their evidence (fill numbers from results/ only; mark anything unverified as TODO):
1. Prompt-only, template-robust fine-tuning for label log-likelihood evaluation.
   Evidence: results/tables/lbl_eval.csv (p_fixed vs p_multi vs e2p_full); decisions 0011, 0012, 0014.
2. Sensitivity of the verified metric to unseen prompt formats and scoring rules (sum vs per-token vs per-character; BOS; letters; yes/no).
   Evidence: results/reports/score_variants_e2p_full.csv; merged-model stress probes (decision 0014).
3. Responses are a shortcut: classifiers given the response score higher on dev (E2: 0.929/0.625; prompt-only E2p: 0.882/0.574) but the task is defined on prompt intent.
   Evidence: results/tables/ft_runs.csv; organizer statement (docs/organizer_forum_2026-11-26.md).
4. Language coverage and performance: Oromo and Twi (outside Tiny Aya's listed languages) are weakest on both tasks.
   Evidence: results/reports/analysis_lbl_e2p_full.txt; backbone comparison (TODO: AfriqueQwen pilot).
5. Category structure: Ethical & Moral Risk is a precision sink in Task B; Task A errors concentrate in Hate/Advice/IP safe prompts and Ethical & Moral Risk unsafe prompts.
   Evidence: results/reports/analysis_lbl_e2p_full.txt.
6. Calibration: oracle class-bias correction adds at most 0.006 (A) / 0.011 (B), and cannot be applied under the organizers' harness.
7. Efficiency: QLoRA on one T4, hours per run (results/tables/ft_runs.csv; decision 0009 for the template-length finding).
Limitations to state: dev set is small (120 per language, about +/-0.04 per-language noise); single seeds except where noted; test conditions inferred, not disclosed; labels in train include disagreement items.
Open items: backbone choice, v2 bank result, train+dev refit, final release checks.
