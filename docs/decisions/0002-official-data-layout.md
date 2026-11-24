# 0002 - Official data layout and task inputs (from Codabench page)

- Date: 2026-11-25
- Layout: public_data/{task_a_safety,task_b_category}/<language>/{train,dev,test,sample_submission}.csv
- Columns: id, language, prompt, response, label (no label in test). Ids unique per language only; never merge languages.
- Both tasks share items and ids; only the label differs.
- Input available to the model: prompt and response. Task definition: intent of the prompt.
- Official dev size is stated inconsistently (text: 20 per language; table: 120). Verify from data.
- Scenario counts (about 250 train) are far below prompt counts (about 860): scenario grouping likely. Train/validation splits must be grouped by scenario, not random. Verify from data.
- Ranking: weighted F1 per language, averaged over ten languages; missing language counts as 0.
- Train+dev may be merged for the final submission. Dev is too small for model selection; hold out part of train.
- English is not part of the data.
