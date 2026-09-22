"""Run the official Jev (TypeSafe System One) over a MedJev split.

Records the raw answer, per-request latency and token usage for every record;
scoring happens separately in `medjev.eval_jev`.

    uv run python -m medjev.run_jev --split test --allow-test --limit 50
    uv run python -m medjev.run_jev --split test --allow-test --workers 8

Output is appended to data/results/jev-<split>.jsonl and the run is resumable: records
already present in that file are skipped.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

# Inference output of the reference systems (hosted Jev, the zero-shot Qwen probes).
# Kept under data/ rather than runs/: these are fixed evaluation inputs, not artifacts of
# a training run, and the Jev answers were paid for once and are replayed thereafter.
RESULTS = "data/results"

API_URL = "https://api.typesafe.ai/v1/systemone"
FIELDS = ("type", "instructions", "criteria")


def load_key(env_file=".env"):
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and os.path.exists(env_file):
        for line in open(env_file):
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("set TYPESAFE_API_KEY (env var or .env)")
    return key


def api_request(record, model):
    """Same shape kev sends: the label and src are stripped, everything the
    model is allowed to see is kept."""
    return {"state": record["state"], "model": model, "questions": {
        qid: {k: v for k, v in q.items() if k in FIELDS}
        for qid, q in record["questions"].items()}}


def call(payload, key, timeout, retries=5):
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                latency = (time.perf_counter() - t0) * 1000
                return json.loads(r.read()), latency, attempt
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code < 500 and e.code != 429:
                raise RuntimeError(last)  # a real client error must surface
        except Exception as e:  # timeouts, connection resets
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(30, 2 ** attempt) + random.random())
    raise RuntimeError(f"giving up after {retries} attempts: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--limit", type=int, default=None, help="first N unfinished records (pilot runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test split is locked: pass --allow-test")

    key = load_key()
    out_path = args.out or f"{RESULTS}/jev-{args.split}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass

    records = [json.loads(l) for l in open(os.path.join(args.data, f"{args.split}.jsonl"))]
    todo = [r for r in records if r["idx"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(records)} records, {len(done)} already done, running {len(todo)} with {args.workers} workers")

    lock = threading.Lock()
    fh = open(out_path, "a")
    stats = {"ok": 0, "failed": 0, "in_tok": 0, "out_tok": 0, "retries": 0, "latency": []}
    t_start = time.perf_counter()

    def work(rec):
        try:
            resp, latency, retried = call(api_request(rec, args.model), key, args.timeout)
        except Exception as e:
            row = {"idx": rec["idx"], "error": str(e)}
            with lock:
                stats["failed"] += 1
                fh.write(json.dumps(row) + "\n"); fh.flush()
            return
        usage = resp.get("usage") or {}
        row = {"idx": rec["idx"], "model": resp.get("model"), "answers": resp.get("answers", {}),
               "latency_ms": round(latency, 1), "usage": usage, "retries": retried,
               "n_questions": len(rec["questions"])}
        with lock:
            stats["ok"] += 1
            stats["in_tok"] += usage.get("input_tokens") or 0
            stats["out_tok"] += usage.get("output_tokens") or 0
            stats["retries"] += retried
            stats["latency"].append(latency)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            n = stats["ok"] + stats["failed"]
            if n % 25 == 0 or n == len(todo):
                el = time.perf_counter() - t_start
                print(f"  {n}/{len(todo)}  ok={stats['ok']} failed={stats['failed']}  "
                      f"{el:.0f}s elapsed  {n / el:.2f} rec/s  {stats['in_tok'] + stats['out_tok']} tokens",
                      flush=True)

    threads, queue = [], list(todo)
    qlock = threading.Lock()

    def loop():
        while True:
            with qlock:
                if not queue:
                    return
                rec = queue.pop()
            work(rec)

    for _ in range(args.workers):
        t = threading.Thread(target=loop, daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join()
    fh.close()

    wall = time.perf_counter() - t_start
    lat = sorted(stats["latency"])
    summary = {
        "split": args.split, "model": args.model, "workers": args.workers,
        "requested": len(todo), "ok": stats["ok"], "failed": stats["failed"], "retries": stats["retries"],
        "wall_seconds": round(wall, 1),
        "records_per_second": round(stats["ok"] / wall, 3) if wall else None,
        "latency_ms": {
            "mean": round(sum(lat) / len(lat), 1) if lat else None,
            "p50": round(lat[len(lat) // 2], 1) if lat else None,
            "p90": round(lat[int(len(lat) * 0.9)], 1) if lat else None,
            "max": round(lat[-1], 1) if lat else None,
        },
        "input_tokens": stats["in_tok"], "output_tokens": stats["out_tok"],
    }
    path = out_path.replace(".jsonl", "-runinfo.json")
    prev = json.load(open(path)) if os.path.exists(path) else []
    prev.append(summary)
    json.dump(prev, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
