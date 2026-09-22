"""Prior-corrected Jev: fit one temperature + class-prior vector per question on
development, apply it to test.

Jev is zero-shot, so it cannot know the annotation thresholds behind an ordered
question (our `diagnostic_workup_intensity` bins count entries in the structured
summary; Jev just reads a dense case report and calls it extensive). Rescaling
its probability vector by a fitted class prior corrects that without touching
the model, and makes the comparison against a development-tuned rule baseline
fair.

    uv run python -m medjev.calibrate_jev
"""
import json

import numpy as np
from sklearn.metrics import f1_score

from medjev.eval_jev import dist, keys_for
from medjev.labels import QUESTIONS

# Inference output of the reference systems (hosted Jev, the zero-shot Qwen probes).
# Kept under data/ rather than runs/: these are fixed evaluation inputs, not artifacts of
# a training run, and the Jev answers were paid for once and are replayed thereafter.
RESULTS = "data/results"

GRID_T = np.arange(0.5, 3.05, 0.1)


def collect(split):
    gold = {json.loads(l)["idx"]: json.loads(l) for l in open(f"data/medjev-v1/{split}.jsonl")}
    ans = {}
    for l in open(f"{RESULTS}/jev-{split}.jsonl"):
        r = json.loads(l)
        if "error" not in r:
            ans[r["idx"]] = r["answers"]
    out = {}
    for idx, rec in gold.items():
        if idx not in ans:
            continue
        for qid, q in rec["questions"].items():
            keys = keys_for(qid)
            p = dist(ans[idx][qid], keys)
            p = p / p.sum() if p.sum() > 0 else np.ones(len(keys)) / len(keys)
            y = keys.index("true" if q["label"] is True else "false") if q["type"] == "noul" \
                else (keys.index(q["label"]) if q["type"] == "choice" else int(q["label"]))
            d = out.setdefault(qid, {"P": [], "y": []})
            d["P"].append(p); d["y"].append(y)
    return {k: {"P": np.vstack(v["P"]), "y": np.array(v["y"])} for k, v in out.items()}


def apply(P, T, w):
    Q = np.clip(P, 1e-6, 1) ** (1.0 / T) * w
    return Q / Q.sum(1, keepdims=True)


def fit(P, y, k):
    """Coordinate search over temperature and a per-class weight vector."""
    best = (None, None, -1)
    prior = np.bincount(y, minlength=k) / len(y)
    pred_prior = P.mean(0)
    for T in GRID_T:
        for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = (prior / np.maximum(pred_prior, 1e-6)) ** mix
            acc = (apply(P, T, w).argmax(1) == y).mean()
            if acc > best[2]:
                best = (T, w, acc)
    return best


def main():
    dev, test = collect("development"), collect("test")
    rows, tot_n, tot_c = {}, 0, 0
    for qid in QUESTIONS:
        if qid not in dev or qid not in test:
            continue
        k = len(keys_for(qid))
        T, w, dev_acc = fit(dev[qid]["P"], dev[qid]["y"], k)
        P, y = test[qid]["P"], test[qid]["y"]
        Q = apply(P, T, w)
        yh = Q.argmax(1)
        onehot = np.eye(k)[y]
        rows[qid] = {
            "type": QUESTIONS[qid]["type"], "n": int(len(y)),
            "raw_accuracy": round(float((P.argmax(1) == y).mean()), 4),
            "calibrated_accuracy": round(float((yh == y).mean()), 4),
            "calibrated_macro_f1": round(float(f1_score(y, yh, average="macro", zero_division=0)), 4),
            "calibrated_brier": round(float(((Q - onehot) ** 2).sum(1).mean()), 4),
            "temperature": round(float(T), 2),
            "dev_accuracy": round(float(dev_acc), 4),
        }
        tot_n += len(y); tot_c += int((yh == y).sum())
    report = {"system": "jev + development-fitted prior correction", "split": "test",
              "micro_accuracy": round(tot_c / tot_n, 4), "questions_scored": tot_n,
              "per_question": rows}
    json.dump(report, open(f"{RESULTS}/jev-test-calibrated.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
