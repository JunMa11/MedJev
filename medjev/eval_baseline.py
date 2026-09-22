"""Score the rule-based baseline against a MedJev split.

    uv run python -m medjev.eval_baseline --split development
    uv run python -m medjev.eval_baseline --split test --allow-test

Reports, per question: accuracy, macro-F1, the majority-class rate (the trivial
floor), and for `score` questions the mean absolute level error. Brier is
computed from a one-hot-with-confidence distribution; the confidence is fitted
on development and reused elsewhere, so the number is the best a hard-decision
rule can do rather than an arbitrary choice.
"""
import argparse
import collections
import json
import os

import numpy as np
from sklearn.metrics import f1_score

from medjev.baseline import predict
from medjev.labels import QUESTIONS

CONF_GRID = np.arange(0.35, 1.0, 0.025)


def n_options(qid, q):
    if q["type"] == "noul":
        return 2
    return len(QUESTIONS[qid]["criteria"])


def brier(correct, k, conf):
    """Mean multi-class Brier for a hard prediction held with probability
    `conf`, the rest spread evenly over the other k-1 options."""
    rest = (1.0 - conf) / (k - 1)
    hit = (1 - conf) ** 2 + (k - 1) * rest ** 2
    miss = conf ** 2 + (1 - rest) ** 2 + (k - 2) * rest ** 2
    return np.where(correct, hit, miss).mean()


def load(split, root):
    path = os.path.join(root, f"{split}.jsonl")
    return [json.loads(line) for line in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="development")
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--conf", default="runs/baseline-conf.json",
                    help="where fitted confidences are stored (written on development, read otherwise)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test split is locked: pass --allow-test to read it")

    # majority-class reference is fitted on train, never on the split being scored
    train_major = {}
    tally = {}
    for r in load("train", args.data):
        for qid, q in r["questions"].items():
            tally.setdefault(qid, collections.Counter())[str(q["label"])] += 1
    for qid, c in tally.items():
        train_major[qid] = c.most_common(1)[0][0]

    records = load(args.split, args.data)
    gold, pred = {}, {}
    for r in records:
        p = predict(r["state"])
        for qid, q in r["questions"].items():
            gold.setdefault(qid, []).append(q["label"])
            pred.setdefault(qid, []).append(p[qid])

    fitting = args.split == "development"
    confs = {} if fitting else json.load(open(args.conf))

    rows, totals = {}, {"n": 0, "correct": 0}
    for qid in QUESTIONS:
        if qid not in gold:
            continue
        y = np.array([str(v) for v in gold[qid]])
        yh = np.array([str(v) for v in pred[qid]])
        correct = y == yh
        k = n_options(qid, {"type": QUESTIONS[qid]["type"]})
        labels = sorted(set(y) | set(yh))
        majority = float(np.mean(y == train_major[qid]))
        if fitting:
            confs[qid] = float(CONF_GRID[int(np.argmin([brier(correct, k, c) for c in CONF_GRID]))])
        row = {
            "type": QUESTIONS[qid]["type"], "n": int(len(y)),
            "accuracy": round(float(correct.mean()), 4),
            "majority_class_train_fitted": round(float(majority), 4),
            "majority_label": train_major[qid],
            "lift_over_majority": round(float(correct.mean() - majority), 4),
            "macro_f1": round(float(f1_score(y, yh, labels=labels, average="macro", zero_division=0)), 4),
            "brier": round(float(brier(correct, k, confs[qid])), 4),
            "confidence": round(confs[qid], 3),
        }
        if QUESTIONS[qid]["type"] == "score":
            row["mae_levels"] = round(float(np.abs(y.astype(int) - yh.astype(int)).mean()), 4)
        rows[qid] = row
        totals["n"] += len(y)
        totals["correct"] += int(correct.sum())

    by_type = {}
    for t in ("noul", "choice", "score"):
        sel = [r for r in rows.values() if r["type"] == t]
        if sel:
            n = sum(r["n"] for r in sel)
            by_type[t] = {
                "questions": len(sel), "n": n,
                "micro_accuracy": round(sum(r["accuracy"] * r["n"] for r in sel) / n, 4),
                "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in sel])), 4),
                "majority_micro": round(sum(r["majority_class_train_fitted"] * r["n"] for r in sel) / n, 4),
            }

    report = {
        "split": args.split, "records": len(records), "questions": totals["n"],
        "micro_accuracy": round(totals["correct"] / totals["n"], 4),
        "macro_accuracy_over_questions": round(float(np.mean([r["accuracy"] for r in rows.values()])), 4),
        "by_type": by_type, "per_question": rows,
    }
    if fitting:
        os.makedirs(os.path.dirname(args.conf), exist_ok=True)
        json.dump(confs, open(args.conf, "w"), indent=2)
    out = args.out or f"runs/baseline-{args.split}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
