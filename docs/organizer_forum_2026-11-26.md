# Organizer answers (Codabench forum, 2026-11-26), archived verbatim

## Thread: verified-leaderboard re-run questions (reply by israela)

Q: Inference harness and script: do we supply our own predict.py / prompt template, or is a fixed harness used?
A: There is no need to provide an inference script, and we will reuse our own evaluation with a variety of prompt templates. The final labeling, of course, will be a log-likelihood calculation of the classes. Participants are only required to provide a Hugging Face-deployed model usable like (AfriQwen or tiny aya)

Q: Input fields: prompt only, or prompt + response?
A: In the final evaluation, only the prompt will be input; responses will not be part of the input. Refer to Figure 1 in the AfriGuard paper (please read the details of the input/output part of shared task page)

Q: Multi-task vs separate checkpoints?
A: We allow submitting separate models, but the test evaluation will run across all languages. Meaning, if model A only works in Amharic and model B works across all languages, we will use 10-language averages for comparison, which would favor model B.

(The question about merged weights vs PEFT adapters was not answered in this reply.)

## Thread: Task A labels that look wrong (reply by israela)

Q: What does label refer to? ... Is this a known issue ... Can participants train on relabeled data?
A: The responses in this dataset are usable for other tasks mentioned in the paper and are LLM-generated; they don't indicate whether the prompt is safe or unsafe. So, using the responses to deduce whether the label is safe is out of scope for this task.
Our final evaluation set will focus on whether models can classify a given prompt as safe or unsafe; the models will not see responses for this task. In the paper, there is a third task that aims to observe model output and asks whether the model answers are valid, and we plan to organize other tasks that focus on model outputs, so responses should be ignored.
The dataset construction and final labels come from three people agreeing on whether the prompt's intent is safe or unsafe; of course, there might be cases where people don't fully agree, and most of the items with disagreement are in the training set. The final test set includes data that is agreed upon by all three language annotators. Participants are free to relabel the datasets, but for the final paper, three-person agreement would make the results solid, and having only one person relabel the data would indicate a research gap.
Please check annotator agreements in the appendix of the paper if that can help you the disagreement observed.

## Shared-task page (read 2026-11-26)
Tasks 1 and 2 list the input as "User prompt" only. Ranking: verified leaderboard (organizers re-run the model). Prizes: judged on the system paper, verified-leaderboard results and novelty of the approach. Models: generative, at most 5B parameters, no external guardrails or closed APIs at inference.
