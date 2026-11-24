# Data card: AfriGuard shared-task data (development phase release)

Source: AfriGuard shared task on Codabench (competition 18627); dataset paper arXiv 2610.26474.
Terms: safety dataset containing prompts describing harmful intent; research use for improving model safety only.
Redistribution: not redistributed in this repository; `data/raw/` is git-ignored.

## Structure
- `<task>/<language>/{train,dev,sample_submission}.csv`; test is released 24 Dec 2026.
- Tasks: `task_a_safety` (safe/unsafe), `task_b_category` (10 categories). Same items and ids in both tasks.
- Columns: id, language (ISO code), prompt, response, label (absent in test).
- Ids are unique within a language folder. A `multilingual/` folder is the union of the ten languages (verified by the audit), not an eleventh language.
- English is not present in the data.

## Verified statistics (see `docs/dataset_audit.md`, `results/tables/`)
- Ten languages; train 848-880 and dev 120 prompts per language; 9,807 labelled items in total.
- Task A over train+dev: 4,599 safe, 5,208 unsafe.
- Task B labels are present for safe and unsafe items alike; categories correlate with safety but are not conditional on it.
- Category distributions differ strongly between languages.
- No duplicate ids; no blank prompts or labels; a few blank responses in train.

## Open questions
- Scenario membership is not recoverable from ids or text similarity.
- Whether response text carries label information beyond the prompt (to be tested).
- Test-split distribution (stated by organisers to be near balanced for Task A).
