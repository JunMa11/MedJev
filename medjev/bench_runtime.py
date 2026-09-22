"""Measure serving latency per question category, for each system.

    uv run python -m medjev.bench_runtime --records 300 --jev-records 200

For every category (noul / choice / score) a request is built holding ONLY that
category's questions, and the same records are timed for the full 11-question
request. This separates the fixed cost of reading the note from the marginal
cost of each additional question, which is the number that decides how many
variables you can afford to extract.

Writes runs/runtime-by-category.json.
"""
import argparse
import json
import os
import statistics
import time

CATS = ("noul", "choice", "score", "all")


def subset(record, cat):
    qs = {qid: q for qid, q in record["questions"].items()
          if cat == "all" or q["type"] == cat}
    return {**record, "questions": qs} if qs else None


def bench_medjev(records, run, device, dtype, max_state, max_branch):
    import numpy as np
    import torch
    from medjev.checkpoint import load
    from medjev.records import materialize

    if device == "cuda" and dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tok, model = load(run, device, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
    model.lm.config.use_cache = True

    out = {}
    for cat in CATS:
        lat, nq = [], []
        for i, rec in enumerate(records):
            sub = subset(rec, cat)
            if sub is None:
                continue
            enc = model.encode(tok, materialize(sub), max_state=max_state, max_branch=max_branch)
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                model.probs_and_prefix(enc)
            if device == "cuda":
                torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - t)
            if i >= 5:                      # drop warm-up records
                lat.append(ms); nq.append(len(sub["questions"]))
        out[cat] = summarize(lat, nq)
        print(f"  medjev {cat:6s} {out[cat]['ms_per_record_median']:7.1f} ms/record "
              f"({out[cat]['questions_per_record']:.1f} q)", flush=True)
    return out


def bench_rules(records):
    from medjev.baseline import predict
    from medjev.labels import QUESTIONS
    # the rule set computes every variable in one pass over the note; time the
    # subset of rules each category needs by calling the shared predictor and
    # attributing its cost to the questions asked
    out = {}
    for cat in CATS:
        lat, nq = [], []
        for i, rec in enumerate(records):
            sub = subset(rec, cat)
            if sub is None:
                continue
            wanted = set(sub["questions"])
            t = time.perf_counter()
            p = predict(rec["state"])
            p = {k: v for k, v in p.items() if k in wanted}
            ms = 1000 * (time.perf_counter() - t)
            if i >= 5:
                lat.append(ms); nq.append(len(wanted))
        out[cat] = summarize(lat, nq)
        print(f"  rules  {cat:6s} {out[cat]['ms_per_record_median']:7.2f} ms/record", flush=True)
    return out


def bench_jev(records, workers, model_name):
    import threading
    from medjev.run_jev import api_request, call, load_key
    key = load_key()
    out = {}
    for cat in CATS:
        subs = [s for s in (subset(r, cat) for r in records) if s]
        lat, nq, lock = [], [], threading.Lock()
        queue = list(subs)

        def loop():
            while True:
                with lock:
                    if not queue:
                        return
                    rec = queue.pop()
                try:
                    _, ms, _ = call(api_request(rec, model_name), key, 120)
                except Exception as e:
                    print("   jev request failed:", e); continue
                with lock:
                    lat.append(ms); nq.append(len(rec["questions"]))

        threads = [threading.Thread(target=loop, daemon=True) for _ in range(workers)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        out[cat] = summarize(lat, nq)
        print(f"  jev    {cat:6s} {out[cat]['ms_per_record_median']:7.1f} ms/record", flush=True)
    return out


def summarize(lat, nq):
    lat_sorted = sorted(lat)
    med = statistics.median(lat) if lat else None
    qpr = sum(nq) / len(nq) if nq else 0
    return {
        "n_records": len(lat),
        "questions_per_record": round(qpr, 2),
        "ms_per_record_median": round(med, 3) if med else None,
        "ms_per_record_mean": round(statistics.mean(lat), 3) if lat else None,
        "ms_per_record_p95": round(lat_sorted[int(len(lat_sorted) * 0.95)], 3) if lat else None,
        "ms_per_question_median": round(med / qpr, 3) if med and qpr else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--records", type=int, default=300)
    ap.add_argument("--jev-records", type=int, default=200)
    ap.add_argument("--run", default="runs/medjev-0.8b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--workers", type=int, default=1, help="Jev concurrency; 1 measures true latency")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--skip", default="", help="comma-separated: medjev,rules,jev")
    ap.add_argument("--out", default="runs/runtime-by-category.json")
    a = ap.parse_args()

    from medjev.model import MAX_BRANCH, MAX_STATE
    recs = [json.loads(l) for l in open(os.path.join(a.data, f"{a.split}.jsonl"))]
    skip = set(x for x in a.skip.split(",") if x)

    report = {"split": a.split, "records_local": a.records, "records_jev": a.jev_records,
              "jev_workers": a.workers, "systems": {}}
    if "rules" not in skip:
        print("regex rules (CPU)")
        report["systems"]["Regex rules (CPU)"] = bench_rules(recs[: a.records])
    if "medjev" not in skip:
        print("MedJev-0.8B (local GPU)")
        report["systems"]["MedJev-0.8B (local GPU)"] = bench_medjev(
            recs[: a.records], a.run, a.device, a.dtype, MAX_STATE, MAX_BRANCH)
    if "jev" not in skip:
        print(f"Jev (hosted API, {a.workers} worker)")
        report["systems"]["Jev (hosted API)"] = bench_jev(recs[: a.jev_records], a.workers, a.model)

    json.dump(report, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
