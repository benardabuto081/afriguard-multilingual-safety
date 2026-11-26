# 0011 - Prompt-only, label log-likelihood pipeline (E2p)

- Date: 2026-11-26
- Trigger: organizer forum answers (relayed; verbatim text to be archived): final evaluation input is prompt only; scoring uses the organizers' own log-likelihood over class labels with several prompt templates; submission is a standalone merged Hugging Face model; test items have 3-of-3 annotator agreement.
- Consequence: E2/E2c (prompt+response input, fixed joint output format) are reference runs with response, not the test condition. Prompt-only pilot-scale result (response removed): A 0.866 / B 0.469 vs 0.899 / 0.497 with response.
- Corrections to the 2026-11-26 handover report: no evidence of joint-training synergy (joint 0.8993/0.4971 vs A-only 0.9011, B-only 0.5156; compact seed SD about 0.005 (A) and 0.009 (B)); the "promptonly" ablation is input without response, not instruction framing; Earth vs Global within seed noise; epoch-1 checkpoint was taken mid-schedule; Qwen3-4B 0.3307 on B was 3-shot choice.
- Method (E2p): one sequence per (example, task), answer = label string only; template bank of 8 (chat and raw styles, two casings of safe/unsafe) for training; 4 held-out templates for evaluation; 80% compact / 20% native chat rendering for Tiny Aya; raw templates start with the tokenizer BOS token. Evaluation: sum of token log-probs per candidate label (cached prefix, verified against a naive reference), argmax; length-normalised variant also reported.
- Open: whether the harness uses the chat template, raw text, a BOS token, option lists, letter/number answers, or capitalised labels; each is a possible train/test mismatch. Question to organizers sent.
- Licence: organizers cited tiny-aya as an example of an acceptable submission; still CC-BY-NC-4.0.
