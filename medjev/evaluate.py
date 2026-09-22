"""Score a MedJev checkpoint on a split, in the same shape as the rule baseline
report so the three systems are directly comparable.

    python -m medjev.evaluate --run runs/medjev-0.8b --split development
    python -m medjev.evaluate --run runs/medjev-0.8b --split test --allow-test

Inference uses the serving path, not the training path: the state is encoded
once per record and every question branch reads it from the KV/recurrent cache
(`DecisionModel.probs_and_prefix`). On this hybrid backbone the training path
re-runs the state for each of the ~9 questions, so the served cost is roughly a
ninth of it — that is where "one model call answers all 11 variables" comes
from, and `latency_ms` below measures exactly that per-record call.

The report carries `majority_class` per question, which is the floor any claim
of improvement has to clear, alongside accuracy, macro-F1, Brier and ECE.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score

from medjev.model import MAX_BRANCH, MAX_STATE
from medjev.records import load_records, materialize


def keys_of(q):
    """Option keys in the order the model scores them, matching `medjev.api`."""
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def gold_key(q):
    if q["type"] == "noul":
        return str(bool(q["label"])).lower()
    return str(q["label"])


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint directory (or Hub id)")
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--split", default="development")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 is the exact path and what the reported accuracy uses; bf16 is the serving path")
    ap.add_argument("--max_state", type=int, default=MAX_STATE)
    ap.add_argument("--max_branch", type=int, default=MAX_BRANCH)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        raise SystemExit("test split is locked: pass --allow-test to read it")

    from medjev.checkpoint import load
    if a.device == "cuda" and a.dtype == "fp32":
        # fp32-exact scoring: TF32's 10-bit mantissa moves probabilities by ~1e-3
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tok, model = load(a.run, a.device, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    model.lm.config.use_cache = True

    records = load_records(os.path.join(a.data, f"{a.split}.jsonl"), source="medjev")
    if a.limit:
        records = records[: a.limit]

    rows = []
    latencies = []
    truncated = 0
    t0 = time.time()
    for i, req in enumerate(records):
        rec = materialize(req)
        enc = model.encode(tok, rec, max_state=a.max_state, max_branch=a.max_branch)
        truncated += int(enc["state_truncated"])
        if a.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            probs, _ = model.probs_and_prefix(enc)      # state once, branches from its cache
        if a.device == "cuda":
            torch.cuda.synchronize()
        latencies.append(1000 * (time.perf_counter() - start))
        for p, (qid, q) in zip(probs, req["questions"].items()):
            keys = keys_of(q)
            p = np.asarray(p, dtype=float)
            j = int(p.argmax())
            g = keys.index(gold_key(q))
            rows.append({"qid": qid, "type": q["type"], "gold": keys[g], "pred": keys[j],
                         "correct": j == g, "p_gold": float(p[g]), "conf": float(p[j]),
                         "brier": float(((p - np.eye(len(keys))[g]) ** 2).sum()),
                         "level_err": abs(j - g) if q["type"] == "score" else None})
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(records)} records, {np.median(latencies):.0f} ms/record median", flush=True)

    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)

    per_question, qtype = {}, {}
    for qid, rs in sorted(by_q.items(), key=lambda kv: (kv[1][0]["type"], kv[0])):
        gold = [r["gold"] for r in rs]
        pred = [r["pred"] for r in rs]
        correct = np.array([r["correct"] for r in rs])
        majority = max(Counter(gold).values()) / len(rs)
        qtype[qid] = rs[0]["type"]
        entry = {"type": rs[0]["type"], "n": len(rs),
                 "accuracy": round(float(correct.mean()), 4),
                 "majority_class": round(majority, 4),
                 "lift_over_majority": round(float(correct.mean()) - majority, 4),
                 "macro_f1": round(float(f1_score(gold, pred, average="macro", zero_division=0)), 4),
                 "brier": round(float(np.mean([r["brier"] for r in rs])), 4),
                 "ece": round(ece([r["conf"] for r in rs], correct), 4),
                 "mean_confidence": round(float(np.mean([r["conf"] for r in rs])), 4)}
        if rs[0]["type"] == "score":
            entry["mean_abs_level_error"] = round(float(np.mean([r["level_err"] for r in rs])), 4)
        per_question[qid] = entry

    by_type = {}
    for t in ("noul", "choice", "score"):
        qs = [q for q in per_question if qtype[q] == t]
        if not qs:
            continue
        n = sum(per_question[q]["n"] for q in qs)
        by_type[t] = {"questions": len(qs), "n": n,
                      "micro_accuracy": round(sum(per_question[q]["accuracy"] * per_question[q]["n"] for q in qs) / n, 4),
                      "macro_f1_mean": round(float(np.mean([per_question[q]["macro_f1"] for q in qs])), 4),
                      "brier": round(sum(per_question[q]["brier"] * per_question[q]["n"] for q in qs) / n, 4),
                      "majority_micro": round(sum(per_question[q]["majority_class"] * per_question[q]["n"] for q in qs) / n, 4)}

    report = {
        "run": a.run, "split": a.split, "dtype": a.dtype, "device": a.device,
        "records": len(records), "questions": len(rows),
        "max_state": a.max_state, "states_truncated": truncated,
        "micro_accuracy": round(float(np.mean([r["correct"] for r in rows])), 4),
        "macro_accuracy_over_questions": round(float(np.mean([per_question[q]["accuracy"] for q in per_question])), 4),
        "brier": round(float(np.mean([r["brier"] for r in rows])), 4),
        "majority_micro": round(sum(per_question[q]["majority_class"] * per_question[q]["n"] for q in per_question) / len(rows), 4),
        "latency_ms_per_record": {"median": round(float(np.median(latencies)), 2),
                                  "p95": round(float(np.percentile(latencies, 95)), 2),
                                  "mean": round(float(np.mean(latencies)), 2)},
        "questions_per_second": round(len(rows) / (sum(latencies) / 1000), 1),
        "wall_seconds": round(time.time() - t0, 1),
        "by_type": by_type, "per_question": per_question,
    }
    out = a.out or os.path.join(a.run, f"eval-{a.split}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
