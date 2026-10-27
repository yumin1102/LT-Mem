import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index_rows(rows):
    indexed = {}
    for row in rows:
        qid = row["qid"]
        if not qid or qid in indexed:
            raise ValueError(f"Empty or duplicate QA id: {qid!r}")
        indexed[qid] = row
    return indexed


def summarize(rows):
    def score(items):
        correct = sum(row["correct"] for row in items)
        return {
            "correct": correct,
            "total": len(items),
            "accuracy": correct / len(items) if items else None,
        }

    report = {"all": score(rows)}
    for key in ("type", "env"):
        report[key] = {
            value: score([row for row in rows if row[key] == value])
            for value in sorted({row[key] for row in rows})
        }
    report["statuses"] = dict(Counter(row["status"] for row in rows))
    return report


def main():
    parser = argparse.ArgumentParser(description="Run or score structured event/frequency QA")
    parser.add_argument("--qa", type=Path, default=Path("evaluation/qa/questions.jsonl"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pred", type=Path, help="JSONL containing qid and structured answer string"
    )
    source.add_argument(
        "--memory",
        action="append",
        metavar="ENV=PATH",
        help="Saved qa_memory.json per environment; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("fp16", "4bit"),
        default="fp16",
        help="Local QA precision; does not change pipeline configuration",
    )
    args = parser.parse_args()
    questions = read_rows(args.qa)
    gt = index_rows(questions)
    if not gt:
        raise ValueError("QA set must not be empty")
    contexts = {}
    if args.memory:
        from ltmem.event_queries import answer_event_question
        from ltmem.model_runtime import ModelRuntime
        from ltmem.runtime.config import seed_everything

        for spec in args.memory:
            env, path = spec.split("=", 1)
            if env in contexts:
                raise ValueError(f"Duplicate environment: {env}")
            contexts[env] = json.loads(Path(path).read_text())
        if {row["env"] for row in questions} - contexts.keys():
            raise ValueError("Missing saved memory for a QA environment")
        os.environ["USE_4BIT"] = "true" if args.precision == "4bit" else "false"
        seed_everything(args.seed)
        runtime = ModelRuntime(provider_kind="qwen")
    else:
        predictions = index_rows(read_rows(args.pred))
        if gt.keys() != predictions.keys():
            raise ValueError("Predictions must cover every QA id exactly once")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with (
        (args.output / "predictions.jsonl").open("w") as pred_file,
        (args.output / "scores.jsonl").open("w") as score_file,
    ):
        for item in questions:
            if args.memory:
                prediction = answer_event_question(
                    item["question"], contexts[item["env"]], model_runtime=runtime
                )
            else:
                prediction = predictions[item["qid"]]
            answer = prediction.get("answer")
            record = {key: item[key] for key in ("qid", "env", "type", "question")}
            record.update(
                answer=answer,
                expected_answer=item["answer"],
                status=prediction.get("status", "provided"),
                correct=isinstance(answer, str) and answer == item["answer"],
            )
            records.append(record)
            pred_file.write(
                json.dumps({**prediction, "qid": item["qid"]}, ensure_ascii=False) + "\n"
            )
            score_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_file.flush()
            score_file.flush()
            print(f"{len(records)}/{len(questions)} {item['qid']}: {answer}", flush=True)
    report = summarize(records)
    if args.memory:
        report["inference"] = {"provider": "qwen", "precision": args.precision, "seed": args.seed}
    report["protocol"] = (
        "Natural-language question -> local Qwen query -> deterministic Delta lookup/count -> exact structured-answer match. No gold query/answer/evidence is passed to Qwen."
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
