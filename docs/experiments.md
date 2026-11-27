# Experiment Log

Programme: about 6-8 substantive training runs, subject to data and compute.

| ID | Description | Status |
|----|-------------|--------|
| E1 | Baseline evaluation (zero-shot, three-shot, deterministic generation, optional token scoring) | planned |
| E2 | Unified multi-task QLoRA baseline, uniform sampling | planned |
| E3 | Sampling: uniform vs alpha 0.5 vs alpha 0.7 | planned |
| E4 | Hard-negative training | planned |
| E5 | Ablations (single vs multi-task, standard vs hard-negative, sampling) and calibration | planned |

## Discipline
Fixed seeds recorded; three seeds for shortlisted configs where compute allows; strict train/validation separation; failed runs recorded; no result described as successful before its output is inspected.

## Run records
(none yet)

## Run records
- E1 (2026-11-25, Kaggle T4): Qwen3-4B and Qwen2.5-3B-Instruct, 0/3-shot, generate/choice, full dev. See results/tables/e1_baselines.csv and decision 0006. Gemma 3 4B pending a loading/dtype check.
- Backbone pilot (2026-11-25, Kaggle T4, 150/lang x 1 epoch): see decision 0008 and results/tables/ft_runs.csv.
- Planned: E2 full (Tiny Aya Earth, 2 epochs, joint); pilot-scale ablations: joint vs Task A-only vs Task B-only vs prompt-only (RQ4 and input ablation).
- E2 (2026-11-25/26, Kaggle T4): Tiny Aya Earth, joint, 8,607 ex, 2 epochs: dev A 0.9291, B 0.6252. Pilot ablations and compact-template pilot: results/tables/ft_runs.csv, decision 0010.
- E2p (2026-11-26, Kaggle T4): prompt-only template-bank training, label log-likelihood evaluation under 4 held-out templates: full run A 0.8819 / B 0.5738; pilots p_multi, p_fixed, p_A, p_B. Decision 0012; table results/tables/lbl_eval.csv.
