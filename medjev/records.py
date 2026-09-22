"""Labelled requests on disk -> the internal records `encode()` consumes.

Derived from `kev/kev/data.py` (Jared Palmer, Apache-2.0) — see NOTICE. MedJev
vendors the three functions it uses and drops the rest: upstream's dataset builders
for Banking77/BoolQ/AG News/etc. pulled in `datasets` and a frozen-suite provenance
system that a single-corpus, single-schema project has no use for.

A **labelled request** is one JSON object per line:

    {"state": "<full_note>",
     "questions": {"hospital_admission": {"type": "noul", "instructions": "...",
                                          "criteria": {...}, "label": true, "src": "..."}}}

Labels are the option name for `choice`, `true`/`false` for `noul`, and the 0-based
level index for `score`. A question whose label cannot be determined is **omitted**,
not defaulted, so records carry different subsets of the schema.

`materialize()` puts a request through the same `api.to_record` path that serving
uses, so training text is byte-identical to inference text.
"""
import hashlib
import json
from pathlib import Path

from medjev.api import Request, to_record


def source_seed(seed, source):
    """A stable 64-bit seed for (run seed, arbitrary string), so per-record
    augmentation is reproducible and independent of iteration order."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def load_records(path, source="medjev"):
    """Labelled requests from a JSONL file, one per line.

    Lines are split on "\\n" only. `str.splitlines()` also breaks on U+2028, U+2029
    and other separators that `json.dumps` leaves raw inside a string, which
    silently cuts a record in half — real notes in this corpus contain them, so this
    is not hypothetical (`build_dataset.jsonl_line` escapes them on write, and
    `tests/test_medjev.py` guards both ends)."""
    records = []
    for n, line in enumerate(Path(path).read_text().split("\n")):
        if not line.strip():
            continue
        r = json.loads(line)
        if "state" not in r or not isinstance(r.get("questions"), dict) or not r["questions"]:
            raise ValueError(f"{path}:{n + 1}: a record needs a state and a non-empty questions object")
        for qid, q in r["questions"].items():
            if "label" not in q:
                raise ValueError(f"{path}:{n + 1}: question {qid!r} has no label")
            q.setdefault("src", f"{source}_{q['type']}")
        text = (json.dumps(r["state"], sort_keys=True, ensure_ascii=False)
                if not isinstance(r["state"], str) else r["state"])
        r["_meta"] = {**{"source": source, "variant": "clean", "id": f"{source}/{n}",
                         "group_id": f"{source}/{n}", "row": n, "split": "custom",
                         "text_sha256": hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()},
                      **r.get("_meta", {})}
        records.append(r)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def materialize(req):
    """Labelled request -> internal record, via the serving path, with integer labels
    and the per-question metadata the trainer and scorer need (`qid`, `qtype`,
    `keys`)."""
    clean = {"state": req["state"],
             "questions": {qid: {k: v for k, v in q.items() if k not in ("label", "src")}
                           for qid, q in req["questions"].items()}}
    rec, meta = to_record(Request.model_validate(clean))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        y = src_q["label"]
        q["label"] = (int(y) if m["type"] == "noul"
                      else m["keys"].index(y) if m["type"] == "choice" else int(y))
        q["src"] = src_q["src"]
        q["qtype"] = m["type"]
        q["qid"] = qid
        q["keys"] = (m["keys"] if m["type"] == "choice"
                     else ["false", "true"] if m["type"] == "noul"
                     else [str(i) for i in range(len(q["options"]))])
    return rec


def permute_choice_options(req, rng):
    """Shuffle the option order of every `choice` question. This is MedJev's only
    augmentation.

    Upstream also injects "none of the above" options and irrelevant distractors to
    teach open-world Choice. MedJev's `choice` vocabularies are closed and identical
    at serving time, so an injected option could only teach an answer that is never
    correct. Shuffling is what keeps the answers order-robust.

    The unused `rng.random()` draw is deliberate: it keeps the random stream aligned
    with upstream's `augment(..., p_none=0, p_none_distract=0, p_distract=0)`, so a
    run trained before this refactor reproduces bit-for-bit from the same seed.
    """
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        crit = dict(q["criteria"])
        rng.random()                       # stream alignment; see the docstring
        keys = list(crit)
        rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}
    return out
