"""Put MedJev next to the baselines on the same split, as a markdown table.

    python -m medjev.compare --split test

The three systems write different report shapes — `medjev.eval_baseline` (rule
baseline), `medjev.base_probe` (zero-shot letter logits) and `medjev.evaluate`
(the fine-tuned model) — so this normalises them onto the fields they share:
per-question accuracy, and the majority-class floor both of them must clear.
"""
import argparse
import glob
import json
import os

# Inference output of the reference systems (hosted Jev, the zero-shot Qwen probes).
# Kept under data/ rather than runs/: these are fixed evaluation inputs, not artifacts of
# a training run, and the Jev answers were paid for once and are replayed thereafter.
RESULTS = "data/results"


def read_rule(path):
    d = json.load(open(path))
    # eval_baseline reports the majority floor as `majority_class` or, once fitted on
    # train rather than the scored split, `majority_class_train_fitted`
    return {q: {"acc": v["accuracy"], "type": v["type"], "n": v["n"],
                "majority": v.get("majority_class", v.get("majority_class_train_fitted")),
                "f1": v.get("macro_f1")}
            for q, v in d["per_question"].items()}, d["micro_accuracy"]


def read_probe(path):
    d = json.load(open(path))
    return {q: {"acc": v["acc"], "type": v["type"], "n": v["n"],
                "majority": v["majority_acc"], "f1": None}
            for q, v in d["by_question"].items()}, d["overall"]["micro_acc"]


def read_medjev(path):
    d = json.load(open(path))
    return {q: {"acc": v["accuracy"], "type": v["type"], "n": v["n"],
                "majority": v["majority_class"], "f1": v.get("macro_f1")}
            for q, v in d["per_question"].items()}, d["micro_accuracy"]


def read_jev(path):
    """`medjev.eval_jev`'s hosted System One report: no majority floor of its own, so it
    borrows the one the other systems computed for the same question."""
    d = json.load(open(path))
    return {q: {"acc": v["accuracy"], "type": v["type"], "n": v["n"],
                "majority": None, "f1": v.get("macro_f1")}
            for q, v in d["per_question"].items()}, d["micro_accuracy"]


def first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="development")
    ap.add_argument("--rule", default=None, help="runs/baseline-<split>.json")
    ap.add_argument("--probe", default=None, help="a medjev.base_probe output directory")
    ap.add_argument("--instruct", default=None, help="a second base_probe directory (chat-prompted)")
    ap.add_argument("--jev", default=None, help="a hosted-Jev report json")
    ap.add_argument("--medjev", default=None, help="a medjev.evaluate report json")
    a = ap.parse_args()

    rule = a.rule or f"runs/baseline-{a.split}.json"
    probe = a.probe or first(f"{RESULTS}/base-*-{a.split}/report.json") or first(f"{RESULTS}/base-*-{a.split}")
    instruct = a.instruct or first(f"{RESULTS}/instruct-*{a.split}*/report.json")
    jev = a.jev or first(f"{RESULTS}/jev-{a.split}-report.json")
    medjev = a.medjev or first(f"runs/medjev-*/eval-{a.split}.json")

    systems = []
    if medjev and os.path.exists(medjev):
        systems.append(("MedJev-0.8B", *read_medjev(medjev), medjev))
    if rule and os.path.exists(rule):
        systems.append(("rule baseline", *read_rule(rule), rule))
    for label, path in (("base 0.8B zero-shot", probe), ("instruct 0.8B", instruct)):
        p = path if path and path.endswith(".json") else (os.path.join(path, "report.json") if path else None)
        if p and os.path.exists(p):
            systems.append((label, *read_probe(p), p))
    if jev and os.path.exists(jev):
        systems.append(("hosted Jev", *read_jev(jev), jev))
    if not systems:
        raise SystemExit(f"no reports found for split {a.split}")

    qids = sorted({q for _, per, _, _ in systems for q in per},
                  key=lambda q: (next(per[q]["type"] for _, per, _, _ in systems if q in per), q))
    names = [s[0] for s in systems]
    print(f"\n## {a.split} split\n")
    print("| question | type | n | majority | " + " | ".join(names) + " |")
    print("|---|---|---|---|" + "---|" * len(names))
    for q in qids:
        ref = next(per[q] for _, per, _, _ in systems if q in per and per[q]["majority"] is not None)
        cells = [f"{per[q]['acc']:.3f}" if q in per else "—" for _, per, _, _ in systems]
        print(f"| `{q}` | {ref['type']} | {ref['n']} | {ref['majority']:.3f} | " + " | ".join(cells) + " |")
    print("| **micro accuracy** | | | | " + " | ".join(f"**{m:.4f}**" for _, _, m, _ in systems) + " |")
    print()
    for name, _, micro, path in systems:
        print(f"- {name}: {micro:.4f}  ({path})")


if __name__ == "__main__":
    main()
