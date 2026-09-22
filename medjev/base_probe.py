"""Zero-shot baseline: how well does the *base* Qwen3.5 model answer the MedJev
questions with no fine-tuning, read out through next-token letter logits?

Same readout as kev/scripts/base_mmlu_probe.py, over data/medjev-v1/*.jsonl:
build `<note>\n<instructions>\nA. ...\nB. ...\nAnswer:` and softmax the logits
of " A".." P" at the last position. This is the floor the fine-tuned MedJev
pointer head has to beat.

    uv run python -m medjev.base_probe --split development
    uv run python -m medjev.base_probe --split test --allow-test --out runs/base-0.8b
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDEFGHIJKLMNOP"


def options(q):
    """(keys, descriptions, gold_key) with the same normalisation kev uses:
    noul -> false/true, score -> stringified level index."""
    if q["type"] == "noul":
        keys = ["false", "true"]
        desc = [q["criteria"][k] for k in keys]
        gold = str(bool(q["label"])).lower()
    elif q["type"] == "score":
        keys = [str(i) for i in range(len(q["criteria"]))]
        desc = list(q["criteria"])
        gold = str(q["label"])
    else:
        keys = list(q["criteria"])
        desc = [q["criteria"][k] or k for k in keys]
        gold = q["label"]
    return keys, desc, gold


SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")


def build_prompt(state, q, mode, tok):
    """plain: raw completion prompt, letters read with a leading space (base models).
    chat: SemIf's readout — chat template, system instruction, JSON payload (instruct models)."""
    keys, desc, gold = options(q)
    if mode == "chat":
        payload = {"evidence": state, "criterion": q["instructions"],
                   "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(desc)]}
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        body = "\n".join(f"{LETTERS[i]}. {d}" for i, d in enumerate(desc))
        prompt = f"{state}\n{q['instructions']}\n{body}\nAnswer:"
    return prompt, keys, gold


def score(items, qtype):
    """Metrics from scored items (or rows loaded back from rows.json)."""
    hits, n, brier = defaultdict(int), defaultdict(int), defaultdict(float)
    majority = defaultdict(lambda: defaultdict(int))
    for c in items:
        q = c["question"]
        hits[q] += int(c["pred"] == c["label"])
        n[q] += 1
        brier[q] += sum((p - (i == c["label"])) ** 2 for i, p in enumerate(c["p"]))
        majority[q][c["label"]] += 1

    per_q = {q: {"type": qtype[q], "n": n[q], "acc": round(hits[q] / n[q], 4),
                 "brier": round(brier[q] / n[q], 4),
                 "majority_acc": round(max(majority[q].values()) / n[q], 4)}
             for q in sorted(n, key=lambda q: (qtype[q], q))}
    by_type = {}
    for t in ("noul", "choice", "score"):
        qs = [q for q in n if qtype[q] == t]
        if not qs:
            continue
        tot = sum(n[q] for q in qs)
        by_type[t] = {"questions": len(qs), "n": tot,
                      "micro_acc": round(sum(hits[q] for q in qs) / tot, 4),
                      "macro_acc": round(sum(hits[q] / n[q] for q in qs) / len(qs), 4),
                      "majority_micro_acc": round(sum(max(majority[q].values()) for q in qs) / tot, 4)}
    overall = {"n": len(items),
               "micro_acc": round(sum(hits.values()) / len(items), 4),
               "macro_acc": round(sum(hits[q] / n[q] for q in n) / len(n), 4),
               "brier": round(sum(brier.values()) / len(items), 4),
               "majority_micro_acc": round(sum(max(majority[q].values()) for q in n) / len(items), 4)}
    return overall, by_type, per_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--split", default="development")
    ap.add_argument("--allow-test", action="store_true",
                    help="required to touch the test split, so it is never read by accident")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-note-tokens", type=int, default=3072,
                    help="truncate the note itself (not the question) to this many tokens")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", choices=["plain", "chat"], default="plain",
                    help="chat: apply the chat template (use for the post-trained Qwen3.5-0.8B)")
    ap.add_argument("--shard", default="0/1", help="i/N: evaluate every Nth item (split one split over N GPUs)")
    ap.add_argument("--out", help="directory for rows.json / report.json")
    a = ap.parse_args()

    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing to read the test split without --allow-test")

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"          # exact: padding after the last real token cannot reach it
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = [tok.encode((" " if a.prompt == "plain" else "") + L, add_special_tokens=False)[0] for L in LETTERS]
    if len(set(letter_ids)) != len(letter_ids):
        raise ValueError("answer-slot tokens collide")

    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    t_loaded = time.time()
    n_params = sum(p.numel() for p in model.parameters())

    # ---- build every (record, question) item ------------------------------ #
    items, qtype = [], {}
    path = os.path.join(a.data, f"{a.split}.jsonl")
    with open(path) as f:
        for n, line in enumerate(f):
            if a.limit and n >= a.limit:
                break
            rec = json.loads(line)
            state = rec["state"]
            enc = tok(state, add_special_tokens=False)["input_ids"]
            if len(enc) > a.max_note_tokens:
                state = tok.decode(enc[: a.max_note_tokens])
            for qid, q in rec["questions"].items():
                qtype[qid] = q["type"]
                prompt, keys, gold = build_prompt(state, q, a.prompt, tok)
                items.append({"idx": rec["idx"], "question": qid, "type": q["type"], "src": q["src"],
                              "keys": keys, "label": keys.index(gold), "prompt": prompt})

    si, sn = (int(x) for x in a.shard.split("/"))
    if sn > 1:
        items = items[si::sn]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))   # length-sorted: less padding
    n_prompt_tokens = sum(len(tok(c["prompt"], add_special_tokens=False)["input_ids"]) for c in items)
    t_prepped = time.time()
    t0 = t_prepped
    with torch.no_grad():
        for b in range(0, len(order), a.batch_size):
            chunk = [items[i] for i in order[b: b + a.batch_size]]
            enc = tok([c["prompt"] for c in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=a.prompt == "plain").to(a.device)
            last = enc["attention_mask"].sum(1) - 1
            # lm_head only at the answer position: the full-sequence projection over a 248k vocab
            # is what OOMs on a shared GPU, and every other position is discarded anyway.
            hidden = model.model(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"]).last_hidden_state
            logits = model.lm_head(hidden[torch.arange(len(chunk), device=a.device), last]).float()
            for c, row in zip(chunk, logits):
                k = len(c["keys"])
                c["p"] = torch.softmax(row[letter_ids[:k]], -1).tolist()
                c["pred"] = int(max(range(k), key=c["p"].__getitem__))
            if b % (a.batch_size * 50) == 0:
                done = b + len(chunk)
                print(f"{done}/{len(order)}  {done / max(time.time() - t0, 1e-9):.1f} items/s", flush=True)

    # ---- scoring ---------------------------------------------------------- #
    overall, by_type, per_q = score(items, qtype)
    report = {
        "model": a.model, "split": a.split, "records": len({c["idx"] for c in items}),
        "readout": "zero-shot next-token letter logits" + (" (chat template)" if a.prompt == "chat" else ""),
        "prompt": a.prompt,
        "shard": a.shard,
        "max_note_tokens": a.max_note_tokens,
        "overall": overall, "by_type": by_type, "by_question": per_q,
    }
    t_end = time.time()
    report["runtime"] = {
        "device": a.device,
        "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
        "parameters": n_params,
        "batch_size": a.batch_size,
        "model_load_s": round(t_loaded - t_start, 1),
        "data_prep_s": round(t_prepped - t_loaded, 1),
        "inference_s": round(t_end - t0, 1),
        "total_wall_s": round(t_end - t_start, 1),
        "items_per_s": round(len(items) / (t_end - t0), 2),
        "ms_per_item": round(1000 * (t_end - t0) / len(items), 1),
        "prompt_tokens": n_prompt_tokens,
        "prompt_tokens_per_s": round(n_prompt_tokens / (t_end - t0), 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if a.device.startswith("cuda") else None,
    }
    report["seconds"] = report["runtime"]["inference_s"]   # kept: earlier runs compare on this field
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "rows.json"), "w") as f:
            json.dump([{k: v for k, v in c.items() if k != "prompt"} for c in items], f)
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
