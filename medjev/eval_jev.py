"""Score a recorded Jev run against the gold labels.

    uv run python -m medjev.eval_jev --split test

Uses kev's own conventions so Jev, the rule baseline and a future Kev
checkpoint are directly comparable: accuracy is argmax of the returned
probability vector, Brier is the squared error of the full distribution, and
`score` MAE uses the distribution's expected level.
"""
import argparse
import collections
import json
import os

import numpy as np
from sklearn.metrics import f1_score

from medjev.labels import QUESTIONS

# Inference output of the reference systems (hosted Jev, the zero-shot Qwen probes).
# Kept under data/ rather than runs/: these are fixed evaluation inputs, not artifacts of
# a training run, and the Jev answers were paid for once and are replayed thereafter.
RESULTS = "data/results"


def keys_for(qid):
    q = QUESTIONS[qid]
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def dist(answer, keys):
    """Jev answer -> probability vector over `keys`."""
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return np.array([1 - p, p])
    probs = answer.get("probabilities") or {}
    return np.array([float(probs.get(k, 0.0)) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_path = args.run or f"{RESULTS}/jev-{args.split}.jsonl"
    gold_recs = {json.loads(l)["idx"]: json.loads(l)
                 for l in open(os.path.join(args.data, f"{args.split}.jsonl"))}

    answers, latency, in_tok, out_tok, errors, model = {}, [], 0, 0, 0, None
    for line in open(run_path):
        row = json.loads(line)
        if "error" in row:
            errors += 1
            continue
        answers[row["idx"]] = row["answers"]
        latency.append(row["latency_ms"])
        in_tok += (row.get("usage") or {}).get("input_tokens") or 0
        out_tok += (row.get("usage") or {}).get("output_tokens") or 0
        model = row.get("model") or model

    per_q = collections.defaultdict(lambda: {"y": [], "yh": [], "p": []})
    missing = 0
    for idx, rec in gold_recs.items():
        got = answers.get(idx)
        if got is None:
            missing += 1
            continue
        for qid, q in rec["questions"].items():
            if qid not in got:
                missing += 1
                continue
            keys = keys_for(qid)
            p = dist(got[qid], keys)
            if p.sum() <= 0:
                p = np.ones(len(keys)) / len(keys)
            p = p / p.sum()
            y = keys.index("true" if q["label"] is True else "false") if q["type"] == "noul" \
                else (keys.index(q["label"]) if q["type"] == "choice" else int(q["label"]))
            per_q[qid]["y"].append(y)
            per_q[qid]["yh"].append(int(p.argmax()))
            per_q[qid]["p"].append(p)

    rows, tot_n, tot_c = {}, 0, 0
    for qid in QUESTIONS:
        if qid not in per_q:
            continue
        d = per_q[qid]
        y, yh, P = np.array(d["y"]), np.array(d["yh"]), np.vstack(d["p"])
        onehot = np.eye(P.shape[1])[y]
        row = {
            "type": QUESTIONS[qid]["type"], "n": int(len(y)),
            "accuracy": round(float((y == yh).mean()), 4),
            "macro_f1": round(float(f1_score(y, yh, average="macro", zero_division=0)), 4),
            "brier": round(float(((P - onehot) ** 2).sum(1).mean()), 4),
            "mean_confidence": round(float(P.max(1).mean()), 4),
        }
        if QUESTIONS[qid]["type"] == "score":
            levels = np.arange(P.shape[1])
            row["mae_levels"] = round(float(np.abs(P @ levels - y).mean()), 4)
        rows[qid] = row
        tot_n += len(y); tot_c += int((y == yh).sum())

    by_type = {}
    for t in ("noul", "choice", "score"):
        sel = [r for r in rows.values() if r["type"] == t]
        if sel:
            n = sum(r["n"] for r in sel)
            by_type[t] = {"questions": len(sel), "n": n,
                          "micro_accuracy": round(sum(r["accuracy"] * r["n"] for r in sel) / n, 4),
                          "brier": round(sum(r["brier"] * r["n"] for r in sel) / n, 4)}

    lat = sorted(latency)
    report = {
        "system": "jev (TypeSafe System One)", "model": model, "split": args.split,
        "records_scored": len(answers), "questions_scored": tot_n,
        "records_missing": missing, "request_errors": errors,
        "micro_accuracy": round(tot_c / tot_n, 4),
        "macro_accuracy_over_questions": round(float(np.mean([r["accuracy"] for r in rows.values()])), 4),
        "by_type": by_type,
        "runtime": {
            "latency_ms_mean": round(float(np.mean(lat)), 1),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
            "latency_ms_p90": round(float(np.percentile(lat, 90)), 1),
            "latency_ms_p99": round(float(np.percentile(lat, 99)), 1),
            "input_tokens": in_tok, "output_tokens": out_tok,
            "tokens_per_record": round((in_tok + out_tok) / max(len(answers), 1), 1),
        },
        "per_question": rows,
    }
    out = args.out or f"{RESULTS}/jev-{args.split}-report.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
