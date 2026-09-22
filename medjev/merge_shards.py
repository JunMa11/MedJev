"""Merge sharded base_probe runs into one report.

    uv run python -m medjev.merge_shards runs/x-s0 runs/x-s1 runs/x-s2 --out runs/x
"""
import argparse
import json
import os

from medjev.base_probe import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, reports = [], []
    for d in a.shards:
        rows += json.load(open(os.path.join(d, "rows.json")))
        reports.append(json.load(open(os.path.join(d, "report.json"))))

    qtype = {r["question"]: r["type"] for r in rows}
    overall, by_type, per_q = score(rows, qtype)
    head = reports[0]
    rt = [r["runtime"] for r in reports]
    report = {
        "model": head["model"], "split": head["split"], "prompt": head["prompt"],
        "readout": head["readout"], "max_note_tokens": head["max_note_tokens"],
        "records": len({r["idx"] for r in rows}),
        "shards": [r["shard"] for r in reports],
        "overall": overall, "by_type": by_type, "by_question": per_q,
        "runtime": {
            "gpus": len(rt), "gpu": rt[0]["gpu"], "parameters": rt[0]["parameters"],
            "batch_size": rt[0]["batch_size"],
            # shards run concurrently: wall clock is the slowest, GPU-seconds is the sum
            "wall_s": round(max(r["total_wall_s"] for r in rt), 1),
            "gpu_seconds": round(sum(r["inference_s"] for r in rt), 1),
            "per_shard_inference_s": [r["inference_s"] for r in rt],
            "items_per_s_aggregate": round(sum(r["items_per_s"] for r in rt), 2),
            "ms_per_item_per_gpu": round(sum(r["ms_per_item"] for r in rt) / len(rt), 1),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rt),
            "peak_gpu_gb": max(r["peak_gpu_gb"] for r in rt),
            "single_gpu_equivalent_s": round(sum(r["inference_s"] for r in rt), 1),
        },
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "rows.json"), "w"))
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
