# Methodology (approved blueprint)

Primary contribution: language-aware multilingual adaptation combined with hard-negative training, on a unified multi-task framework.

## Components
- **A. Unified multi-task learning:** one model for binary safe/unsafe and ten-category classification. Value to be established against independent task-specific training. Missing task labels are handled by masking, never by discarding or inventing labels.
- **B. Language-aware sampling:** uniform vs temperature-based (alpha = 0.5, 0.7). Implementation determined from the actual language distribution.
- **C. Hard-negative training:** benign prompts sharing sensitive vocabulary or context with harmful ones. Terminology: hard-negative training (not contrastive learning unless a contrastive objective is implemented). Any synthetic example needs documented provenance and quality control.

## Backbones (candidates)
Qwen3-4B (primary), Gemma 3 4B, Qwen2.5-3B-Instruct. Selection by eligibility, tokenizer/template compatibility, licence, multilingual ability, compute and reproducibility. Record model ID, revision, tokenizer, template, configuration and licence for every run.

## Fine-tuning
QLoRA or another parameter-efficient method, subject to a feasibility check on the chosen GPU host.

## Evaluation
Official weighted F1 per task and language plus overall average; supplementary macro F1, precision/recall, confusion matrices, category-level results, error and false-refusal analysis, time and resource cost. Thresholds tuned on validation data only.
