# Evaluation

## Events

Score the event CSVs from your Lab runs against the corrected ground truth:

```bash
python3 scripts/evaluate_events.py \
  --gt evaluation/ground-truth/event_gt_corrected.csv \
  --pred results/lab_s/pred_event_ours.csv results/lab_l/pred_event_ours_lab_l.csv \
  --output results/event_metrics.json
```

Predictions must cover every GT object/session exactly once; HOLD counts as an error.

## QA-Event and QA-Freq

[Questions](qa/questions.jsonl) contains 50 event questions and 30 frequency questions, split equally between Lab-S and Lab-L. These development questions were selected for unique, positive answers with prior experimental results available. They are not a held-out benchmark and do not establish reproduction of the paper’s QA results.

Open the container with `python3 scripts/launch.py shell`, then run:

```bash
python scripts/evaluate_qa.py \
  --memory Lab-S=results/lab_s/qa_memory.json \
  --memory Lab-L=results/lab_l/qa_memory.json \
  --output results/qa
```

Local Qwen interprets each question; Python retrieves or counts saved Delta events. Gold answers are used only for scoring. Every question counts, including invalid or unknown answers. The default is FP16; add `--precision 4bit` to reduce memory use. Use a new output directory.

Scoring writes `predictions.jsonl`, `scores.jsonl` and `metrics.json` to the output directory.
