import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


LABELS = ("APPEAR", "DISAPPEAR", "REAPPEAR", "MOVE", "NONE")


def load(paths, *, prediction=False):
    result = {}
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"env", "session", "object", "event"} <= set(reader.fieldnames or []):
                raise ValueError(f"{path}: required columns are env, session, object, event")
            for line, row in enumerate(reader, 2):
                key = tuple((row[k] or "").strip() for k in ("env", "session", "object"))
                if not all(key):
                    raise ValueError(f"{path}:{line}: empty merge key")
                label = (row["event"] or "").strip().upper()
                if label not in LABELS and not (prediction and label == "HOLD"):
                    raise ValueError(f"{path}:{line}: unsupported event {label!r}")
                if key in result:
                    raise ValueError(f"{path}:{line}: duplicate key {key}")
                result[key] = label
    if not result:
        raise ValueError("No event rows found")
    return result


def evaluate(gt, pred):
    if gt.keys() != pred.keys():
        raise ValueError(
            f"Coverage mismatch: missing={sorted(gt.keys() - pred.keys())}, extra={sorted(pred.keys() - gt.keys())}"
        )
    correct = sum(gt[k] == pred[k] for k in gt)
    truth_count, pred_count = Counter(gt.values()), Counter(pred.values())
    per_class = {}
    for label in LABELS:
        tp = sum(gt[k] == pred[k] == label for k in gt)
        denominator = truth_count[label] + pred_count[label]
        per_class[label] = {
            "gt_count": truth_count[label],
            "pred_count": pred_count[label],
            "f1": 2 * tp / denominator if denominator else 0.0,
        }
    return {
        "n": len(gt),
        "correct": correct,
        "exact_accuracy": correct / len(gt),
        "event_micro_f1": correct / len(gt),
        "held": pred_count["HOLD"],
        "per_class": per_class,
        "errors": [
            {"env": k[0], "session": k[1], "object": k[2], "gt": gt[k], "prediction": pred[k]}
            for k in sorted(gt)
            if gt[k] != pred[k]
        ],
    }


def identity(path):
    path = Path(path)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(
        description="Score complete LT-Mem event CSVs without models or API calls"
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gt, pred = load([args.gt]), load(args.pred, prediction=True)
    report = {
        "ground_truth": identity(args.gt),
        "predictions": [identity(p) for p in args.pred],
        "all": evaluate(gt, pred),
        "by_environment": {
            env: evaluate(
                {k: v for k, v in gt.items() if k[0] == env},
                {k: v for k, v in pred.items() if k[0] == env},
            )
            for env in sorted({k[0] for k in gt})
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for name, metrics in [("All", report["all"]), *report["by_environment"].items()]:
        print(
            f"{name}: {metrics['correct']}/{metrics['n']}, Event micro-F1={metrics['event_micro_f1']:.3f}, held={metrics['held']}"
        )


if __name__ == "__main__":
    main()
