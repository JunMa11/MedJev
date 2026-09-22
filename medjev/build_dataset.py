"""Convert augmented-clinical-notes into Kev-format labelled requests.

    python -m medjev.build_dataset --out data/medjev-v1

Writes train/development/test JSONL (80/10/10, split by record idx) plus
stats.json. State is `full_note`; `note` and `conversation` are ignored.
"""
import argparse
import collections
import hashlib
import json
import os

from medjev.labels import QUESTIONS, build_questions, parse_summary

SRC = "medjev-acn"

# json.dumps escapes \n and \r but leaves these raw, and Python's str.splitlines()
# (which many JSONL readers use) breaks a line on every one of them. A handful of
# PMC notes contain U+2028, so an unescaped line is silently truncated mid-record.
LINE_BREAKS = str.maketrans({c: "\\u%04x" % ord(c) for c in "\x0b\x0c\x1c\x1d\x1e\x85  "})


def jsonl_line(obj):
    """One JSONL line that survives str.splitlines() as well as split('\\n')."""
    return json.dumps(obj, ensure_ascii=False).translate(LINE_BREAKS) + "\n"


def split_of(idx, ratios=(0.8, 0.1, 0.1)):
    """Deterministic, record-stable split: hash the source idx."""
    h = int(hashlib.sha1(str(idx).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    return "development" if h < ratios[0] + ratios[1] else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="augmented-clinical-notes/augmented_notes_30K.jsonl")
    ap.add_argument("--out", default="data/medjev-v1")
    ap.add_argument("--min_questions", type=int, default=3,
                    help="drop records with fewer labelled questions than this")
    ap.add_argument("--undocumented_smoking_rate", type=float, default=0.15,
                    help="share of records that keep smoking_status='not_documented'")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w") for s in ("train", "development", "test")}

    stats = {
        "records_read": 0, "summaries_unparsable": 0, "records_dropped": 0,
        "records_written": collections.Counter(),
        "questions_written": collections.Counter(),
        "label_counts": collections.defaultdict(collections.Counter),
        "state_words": collections.Counter(),
    }

    for line in open(args.src):
        rec = json.loads(line)
        stats["records_read"] += 1
        summary = parse_summary(rec["summary"])
        if summary is None or not isinstance(summary, dict):
            stats["summaries_unparsable"] += 1
            continue

        idx = rec["idx"]
        # keep a fixed minority of "not_documented" smoking records so the option
        # is learnable without swamping the other three
        keep_und = (int(hashlib.sha1(("smk" + str(idx)).encode()).hexdigest()[:8], 16)
                    / 0xFFFFFFFF) < args.undocumented_smoking_rate
        questions = build_questions(summary, keep_undocumented_smoking=keep_und)
        if len(questions) < args.min_questions:
            stats["records_dropped"] += 1
            continue

        state = rec["full_note"].strip()
        for qid, q in questions.items():
            q["src"] = f"{SRC}:{qid}"
            stats["questions_written"][qid] += 1
            stats["label_counts"][qid][str(q["label"])] += 1

        sp = split_of(idx)
        files[sp].write(jsonl_line({"idx": idx, "state": state, "questions": questions}))
        stats["records_written"][sp] += 1
        w = len(state.split())
        stats["state_words"]["<=295" if w <= 295 else "296-780" if w <= 780 else ">780"] += 1

    for f in files.values():
        f.close()

    out = {
        "records_read": stats["records_read"],
        "summaries_unparsable": stats["summaries_unparsable"],
        "records_dropped": stats["records_dropped"],
        "records_written": dict(stats["records_written"]),
        "questions_written": dict(stats["questions_written"]),
        "total_questions": sum(stats["questions_written"].values()),
        "state_words_vs_384_token_budget": dict(stats["state_words"]),
        "label_distribution": {k: dict(v.most_common()) for k, v in stats["label_counts"].items()},
        "question_types": {k: v["type"] for k, v in QUESTIONS.items()},
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
