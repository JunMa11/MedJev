"""Three test-set figures:

  1. accuracy by question category, all systems     -> test-accuracy-by-category.png
  2. serving latency by question category           -> test-runtime-by-category.png
  3. accuracy per clinical variable, all systems    -> test-accuracy-by-variable.png

    .venv/bin/python -m medjev.figures

All figures are set in Times New Roman (see `use_times`). Numbers are read from the
reports in runs/ so the figures cannot drift from the evaluations. Latency comes from `medjev.bench_runtime`, which times a request
holding only one category's questions.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medjev.labels import QUESTIONS

# Inference output of the reference systems (hosted Jev, the zero-shot Qwen probes).
# Kept under data/ rather than runs/: these are fixed evaluation inputs, not artifacts of
# a training run, and the Jev answers were paid for once and are replayed thereafter.
RESULTS = "data/results"

# Every figure is set in Times New Roman. The fallbacks are metric-compatible or
# near-compatible Times clones, in descending order of fidelity, so a machine
# without the Microsoft font still renders a Times-like serif rather than silently
# dropping to DejaVu Sans; `use_times()` says which one it actually got.
# mathtext.fontset="stix" keeps any math in a Times-matching face.
SERIF_STACK = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Tinos",
               "Liberation Serif", "FreeSerif", "DejaVu Serif"]


def use_times(verbose=True):
    """Apply the serif stack to every subsequent figure. Returns the family that
    matplotlib will actually resolve to."""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    resolved = next((f for f in SERIF_STACK if f in available), None)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,   # STIX/Times lack U+2212; use ASCII hyphen
    })
    if verbose and resolved != SERIF_STACK[0]:
        print(f"note: 'Times New Roman' not installed; figures will use {resolved!r}")
    return resolved


use_times(verbose=False)

# The model was retrained on 2026-09-21 (runs/medjev-0.8b/, final adapter at
# step 5272), so this points at that run's own test report again rather than at
# runs/medjev-0.8b-eval-test-SNAPSHOT.json, which holds the numbers of the
# earlier checkpoint that was deleted from disk.
MEDJEV_TEST = "runs/medjev-0.8b/eval-test.json"

CATS = [("noul", "Yes/no\n(noul)"), ("choice", "Multiple choice\n(choice)"),
        ("score", "Ordered score\n(score)"), ("overall", "All questions")]

# categorical slots 1-5 in the palette's documented order (validated for the
# adjacent pairlist in both modes)
THEME = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "grid": "#e4e3df", "ref": "#9a9992",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]},
}


def by_type(per_question, acc_key, type_key="type"):
    agg = {}
    for qid, row in per_question.items():
        t = QUESTIONS[qid]["type"] if type_key is None else row[type_key]
        a = agg.setdefault(t, [0.0, 0])
        a[0] += row[acc_key] * row["n"]; a[1] += row["n"]
    out = {t: v[0] / v[1] for t, v in agg.items()}
    tot = sum(v[0] for v in agg.values()), sum(v[1] for v in agg.values())
    out["overall"] = tot[0] / tot[1]
    return out


def collect():
    medjev = json.load(open(MEDJEV_TEST))
    rules = json.load(open("runs/baseline-test.json"))
    jev = json.load(open(f"{RESULTS}/jev-test-report.json"))
    cal = json.load(open(f"{RESULTS}/jev-test-calibrated.json"))
    instr = json.load(open(f"{RESULTS}/instruct-qwen3.5-0.8b-test-chat/report.json"))

    def from_by_type(d, key="micro_accuracy", overall=None):
        out = {t: d["by_type"][t][key] for t in ("noul", "choice", "score")}
        out["overall"] = overall
        return out

    return [
        # (label, colour slot, values)  — drawn best-first, hue order preserved
        ("MedJev-0.8B (fine-tuned)", 0, from_by_type(medjev, "micro_accuracy", medjev["micro_accuracy"])),
        ("Jev + prior correction", 1, by_type(cal["per_question"], "calibrated_accuracy")),
        ("Regex rules", 2, from_by_type(rules, "micro_accuracy", rules["micro_accuracy"])),
        ("Jev (zero-shot)", 3, from_by_type(jev, "micro_accuracy", jev["micro_accuracy"])),
        ("Qwen3.5-0.8B-Instruct (zero-shot)", 4,
         {**{t: instr["by_type"][t]["micro_acc"] for t in ("noul", "choice", "score")},
          "overall": instr["overall"]["micro_acc"]}),
    ]


def draw(mode, systems):
    c = THEME[mode]
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
    fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])

    n = len(systems)
    x = np.arange(len(CATS))
    width = 0.78 / n

    for i, (label, slot, vals) in enumerate(systems):
        ys = [vals[k] for k, _ in CATS]
        pos = x - 0.39 + width * (i + 0.5)
        ax.bar(pos, ys, width * 0.9, label=label, color=c["series"][slot],
               edgecolor=c["surface"], linewidth=1.2, zorder=3)
        for px, y in zip(pos, ys):      # relief rule: every bar carries its value
            ax.text(px, y + 0.012, f"{y:.3f}", ha="center", va="bottom", rotation=90,
                    fontsize=6.6, color=c["secondary"], zorder=4)

    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in CATS], color=c["primary"], fontsize=10)
    ax.set_ylabel("Accuracy on the test split", color=c["secondary"], fontsize=10)
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(colors=c["secondary"], labelsize=9, length=0)
    ax.yaxis.grid(True, color=c["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])

    ax.set_title("Clinical variable extraction from PMC case notes — 2,895 held-out notes, 26,286 questions",
                 color=c["primary"], fontsize=12, pad=14, loc="left")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False,
                    fontsize=9, handlelength=1.1, handleheight=1.1, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(c["secondary"])

    fig.tight_layout()
    out = "docs/figures/test-accuracy-by-category.png"
    fig.savefig(out, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


RUNTIME_CATS = [("noul", "Yes/no\n(5 questions)"), ("choice", "Multiple choice\n(2.3 questions)"),
                ("score", "Ordered score\n(1.8 questions)"), ("all", "All 11 variables\n(9.1 questions)")]


def draw_runtime():
    """Latency per record, by the category of question asked. Log axis: the
    systems span ~80x, and a bar from a zero baseline cannot be drawn on a log
    scale, so this is a dot plot."""
    c = THEME["light"]
    data = json.load(open("runs/runtime-by-category.json"))
    order = ["MedJev-0.8B (local GPU)", "Jev (hosted API)", "Regex rules (CPU)"]
    systems = [(name, i, data["systems"][name]) for i, name in enumerate(order) if name in data["systems"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
    fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])
    x = np.arange(len(RUNTIME_CATS))
    span = 0.46

    for i, (label, slot, vals) in enumerate(systems):
        off = -span / 2 + span * (i / max(len(systems) - 1, 1))
        ys = [vals[k]["ms_per_record_median"] for k, _ in RUNTIME_CATS]
        ax.scatter(x + off, ys, s=110, color=c["series"][slot], label=label,
                   edgecolor=c["surface"], linewidth=1.5, zorder=3)
        for px, y in zip(x + off, ys):
            ax.annotate(f"{y:,.0f} ms" if y >= 10 else f"{y:.1f} ms", (px, y),
                        textcoords="offset points", xytext=(0, 11), ha="center",
                        fontsize=8, color=c["secondary"], zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(1.6, 900)
    ax.set_yticks([2, 5, 10, 25, 50, 100, 250, 500])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in RUNTIME_CATS], color=c["primary"], fontsize=10)
    ax.set_xlim(-0.55, len(RUNTIME_CATS) - 0.45)
    ax.set_ylabel("Median latency per note (ms, log scale)", color=c["secondary"], fontsize=10)
    ax.tick_params(which="both", colors=c["secondary"], labelsize=9, length=0)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.grid(True, color=c["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.set_title("Serving cost by question category — one request per note, "
                 f"{data['records_local']} test notes",
                 color=c["primary"], fontsize=12, pad=14, loc="left")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
                    fontsize=9, handlelength=1.0, columnspacing=2.0)
    for t_ in leg.get_texts():
        t_.set_color(c["secondary"])

    fig.tight_layout()
    out = "docs/figures/test-runtime-by-category.png"
    fig.savefig(out, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)

    with open("docs/figures/test-runtime-by-category.csv", "w") as f:
        f.write("system," + ",".join(k for k, _ in RUNTIME_CATS) + ",ms_per_question_all\n")
        for label, _, vals in systems:
            f.write(label + "," + ",".join(f"{vals[k]['ms_per_record_median']:.2f}" for k, _ in RUNTIME_CATS)
                    + f",{vals['all']['ms_per_question_median']:.2f}\n")
    return out, systems


def collect_per_variable():
    """(variable, type, n, {system: accuracy}) for all 11 variables."""
    medjev = json.load(open(MEDJEV_TEST))["per_question"]
    rules = json.load(open("runs/baseline-test.json"))["per_question"]
    jev = json.load(open(f"{RESULTS}/jev-test-report.json"))["per_question"]
    cal = json.load(open(f"{RESULTS}/jev-test-calibrated.json"))["per_question"]
    instr = json.load(open(f"{RESULTS}/instruct-qwen3.5-0.8b-test-chat/report.json"))["by_question"]

    rows = []
    for qid in medjev:
        rows.append({
            "qid": qid, "type": QUESTIONS[qid]["type"], "n": rules[qid]["n"],
            "MedJev-0.8B (fine-tuned)": medjev[qid]["accuracy"],
            "Jev + prior correction": cal[qid]["calibrated_accuracy"],
            "Regex rules": rules[qid]["accuracy"],
            "Jev (zero-shot)": jev[qid]["accuracy"],
            "Qwen3.5-0.8B-Instruct (zero-shot)": instr[qid]["acc"],
        })
    order = {"noul": 0, "choice": 1, "score": 2}
    rows.sort(key=lambda r: (order[r["type"]], -r["MedJev-0.8B (fine-tuned)"]))
    return rows


VAR_SERIES = [("MedJev-0.8B (fine-tuned)", 0), ("Jev + prior correction", 1), ("Regex rules", 2),
              ("Jev (zero-shot)", 3), ("Qwen3.5-0.8B-Instruct (zero-shot)", 4)]
TYPE_LABEL = {"noul": "Yes/no  ·  noul", "choice": "Multiple choice  ·  choice",
              "score": "Ordered score  ·  score"}


def draw_by_variable():
    """Horizontal grouped bars: 11 variables x 6 systems. Bars rather than dots
    because five systems sit within 0.01 of each other on several variables -
    dots would overlap, bars stay separated by position."""
    c = THEME["light"]
    rows = collect_per_variable()
    n_series = len(VAR_SERIES)
    bar_h = 0.8 / n_series

    # one slot per variable, plus a blank slot between question types
    ypos, ticks, labels, seps = {}, [], [], []
    y = 0.0
    prev = None
    for r in rows:
        if prev is not None and r["type"] != prev:
            seps.append(y - 0.5)
            y += 0.9
        ypos[r["qid"]] = y
        ticks.append(y)
        labels.append(f"{r['qid']}   (n={r['n']:,})")
        prev = r["type"]
        y += 1.0

    fig, ax = plt.subplots(figsize=(11.5, 8.6), dpi=200)
    fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])

    for i, (label, slot) in enumerate(VAR_SERIES):
        color = c["series"][slot]
        ys = [ypos[r["qid"]] + 0.4 - bar_h * (i + 0.5) for r in rows]
        xs = [r[label] for r in rows]
        ax.barh(ys, xs, bar_h * 0.88, color=color, label=label,
                edgecolor=c["surface"], linewidth=0.9, zorder=3)
        if slot == 0:                      # label the fine-tuned model only
            for yy, xx in zip(ys, xs):
                ax.text(xx + 0.008, yy, f"{xx:.3f}", va="center", ha="left",
                        fontsize=7.5, color=c["secondary"], zorder=4)

    for s in seps:                          # hairline between question types
        ax.axhline(s, color=c["grid"], linewidth=1, zorder=1)
    for t_, lab in TYPE_LABEL.items():
        ys = [ypos[r["qid"]] for r in rows if r["type"] == t_]
        ax.text(1.085, (min(ys) + max(ys)) / 2, lab, rotation=270, va="center", ha="center",
                fontsize=9, color=c["secondary"])

    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9.5, color=c["primary"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Accuracy on the test split", color=c["secondary"], fontsize=10)
    ax.tick_params(which="both", colors=c["secondary"], labelsize=9, length=0)
    ax.xaxis.grid(True, color=c["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.set_title("Accuracy per clinical variable — 2,895 held-out notes",
                 color=c["primary"], fontsize=12, pad=14, loc="left")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=False,
                    fontsize=9, handlelength=1.1, handleheight=1.1, columnspacing=1.6)
    for t_ in leg.get_texts():
        t_.set_color(c["secondary"])

    fig.tight_layout()
    out = "docs/figures/test-accuracy-by-variable.png"
    fig.savefig(out, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)

    with open("docs/figures/test-accuracy-by-variable.csv", "w") as f:
        f.write("variable,type,n," + ",".join(s for s, _ in VAR_SERIES) + "\n")
        for r in rows:
            f.write(f"{r['qid']},{r['type']},{r['n']}," +
                    ",".join(f"{r[s]:.4f}" for s, _ in VAR_SERIES) + "\n")
    return out, rows


def main():
    systems = collect()
    rows = [("system", *[k for k, _ in CATS])]
    for label, _, vals in systems:
        rows.append((label, *[f"{vals[k]:.4f}" for k, _ in CATS]))
    with open("docs/figures/test-accuracy-by-category.csv", "w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print("wrote", draw("light", systems))
    for r in rows:
        print(f"{r[0]:36s} " + "  ".join(f"{v:>8s}" for v in r[1:]))

    out, rows = draw_by_variable()
    print("\nwrote", out)
    for r in rows:
        print(f"  {r['qid']:30s} {r['type']:6s} "
              + "  ".join(f"{r[s]:.3f}" for s, _ in VAR_SERIES))

    out, rt = draw_runtime()
    print("\nwrote", out)
    for label, _, vals in rt:
        print(f"{label:28s} " + "  ".join(f"{vals[k]['ms_per_record_median']:>9.2f}" for k, _ in RUNTIME_CATS)
              + f"   | {vals['all']['ms_per_question_median']:.2f} ms/question")


if __name__ == "__main__":
    main()
