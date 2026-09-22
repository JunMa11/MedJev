"""Fine-tune Qwen3.5-0.8B-Base into MedJev: a LoRA adapter and a pointer head over
a frozen backbone, no text generation, on the MedJev clinical-variable dataset.

    # single GPU
    python -m medjev.train --out runs/medjev-0.8b

    # all three GPUs
    torchrun --nproc_per_node=3 -m medjev.train --out runs/medjev-0.8b --accum 3

The architecture and training format come from kev (see NOTICE), vendored into
`medjev.model` / `medjev.records`. What MedJev does differently, and why:

* **State budget.** `medjev.model.MAX_STATE` is 2048, not upstream's 384. Notes have a
  median of ~670 tokens and 384 costs 2.7 micro points; see `docs/state_budget.md`
  for the measured sweep behind the current value.
* **No suite machinery.** One source, one fixed 11-question schema, so the
  frozen-suite provenance, holdout sources, eval-only guards and source mixing
  are all dropped.
* **Augmentation is permutation only.** Upstream injects "none of the above" options
  and distractors to teach open-world Choice. MedJev's Choice questions are a
  closed, fixed vocabulary that is identical at serving time, so an injected
  option would only teach an answer the model must never give. Option order is
  still shuffled every epoch — that is what keeps Choice order-robust.
* **Question subsampling.** The backbone is hybrid (12 Gated DeltaNet layers),
  so each question runs as its own row with the state repeated
  (`forward_rows_batch`). Cost is therefore `questions x state tokens`, and with
  ~9 questions per record the state dominates everything. `--questions_per_record`
  trains on a random subset per record per epoch, so an epoch costs
  proportionally less while every question is still seen across epochs.
* **Class weighting.** Several MedJev variables are skewed (`follow_up_planned`
  is 10.6% positive). `--class_weight balanced` reweights each question's
  classes by inverse frequency, which buys macro-F1 at a small cost in accuracy.
* **Data parallelism.** Ranks hold identical replicas and each takes a disjoint
  shard of the records; gradients are averaged by hand at every optimizer step
  (`all_reduce_grads`). Only ~11M parameters are trainable, so the all-reduce is
  negligible, and doing it manually avoids wrapping the custom `forward_batch` /
  `forward_rows_batch` entry points in `DistributedDataParallel`, which only
  installs its reducer when you call `DDP.forward`.
* **In-training validation.** `--val_every` steps, a fixed development subset is
  scored through the serving path (state encoded once per record). The shard is
  split across ranks and the counts all-reduced, so validation costs a few
  seconds and is not rank-0 serial.

Evaluate a finished checkpoint with `python -m medjev.evaluate --run <dir>`.
"""
import argparse
import contextlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from medjev.labels import QUESTIONS
from medjev.model import MAX_BRANCH, MAX_STATE, DecisionModel, load_tokenizer
from medjev.records import load_records, materialize, permute_choice_options, source_seed

SOURCE = "medjev"


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #

def setup_dist():
    """(rank, world_size, device). Falls back to single-process when torchrun did
    not set the rendezvous variables."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, dev
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            device_id=torch.device(f"cuda:{local}"))
    return rank, world, f"cuda:{local}"


def broadcast_parameters(model, world):
    """Rank 0's LoRA and pointer head win. Identical seeds should already produce
    identical initialisation, but replicas that silently diverge train happily and
    report a plausible loss, so this is not left to chance."""
    if world > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)


def all_reduce_grads(params, world):
    """Average gradients across ranks. Called once per optimizer step, after the
    last micro-batch of the accumulation group."""
    if world == 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


def assert_replicas_agree(params, device, world, where):
    """Gradients are all-reduced by hand here, so a bug would leave the replicas
    drifting apart while every rank still reports a falling loss. Compare a cheap
    checksum of the trainable weights: max and min across ranks must be identical."""
    if world == 1:
        return
    checksum = torch.stack([p.data.double().sum() for p in params]).sum().to(device)
    lo, hi = checksum.clone(), checksum.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.allclose(lo, hi, rtol=0, atol=1e-6):
        raise RuntimeError(f"replicas diverged at {where}: checksum spread {float(hi - lo):.3e}")


def reduce_counts(values, device, world):
    if world == 1:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #

def question_loss(z, q, dev, ord_w):
    """Cross-entropy over a question's options, plus — for `score` questions — the
    ranked probability score, which is a proper scoring rule for ordered levels.

    Adapted from kev's trainer (see NOTICE). The RPS term penalises being wrong by
    two levels more than being wrong by one, which plain cross-entropy does not:
    `symptom_severity`, `treatment_response` and `diagnostic_workup_intensity` are
    genuinely ordered, so it is on by default (`--ord_w 0.3`). Upstream's soft-target
    branch is dropped — MedJev labels are always hard."""
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y)
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


# --------------------------------------------------------------------------- #
# label / weighting helpers
# --------------------------------------------------------------------------- #

def class_weights(reqs):
    """Inverse-frequency weight per (question id, label), normalised so the mean
    weight of a question's observed labels is 1 — the loss scale stays comparable
    to the unweighted run, only the balance between classes changes."""
    counts = defaultdict(Counter)
    for r in reqs:
        for qid, q in r["questions"].items():
            counts[qid][str(q["label"])] += 1
    out = {}
    for qid, c in counts.items():
        raw = {k: len(c) * sum(c.values()) / (len(c) * v) for k, v in c.items()}
        mean = sum(raw[k] * c[k] for k in c) / sum(c.values())
        out[qid] = {k: v / mean for k, v in raw.items()}
    return out


def label_key(q):
    """The materialized question's label as `class_weights` keyed it (the raw
    request label stringified): option name, "True"/"False", or the level index."""
    return str(bool(q["label"])) if q["qtype"] == "noul" else q["keys"][q["label"]]


def subsample_questions(req, rng, k):
    """Keep k randomly chosen questions (all of them when k <= 0 or k >= len)."""
    qs = req["questions"]
    if k <= 0 or k >= len(qs):
        return req
    keep = rng.sample(sorted(qs), k)
    return {**req, "questions": {qid: qs[qid] for qid in keep}}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, tok, reqs, args, rank, world, device):
    """Accuracy and loss on a fixed development subset, scored the way the model
    is served: one state pass per record, question branches off its cache. No
    augmentation, so the number is comparable across steps."""
    was_training = model.training
    model.eval()
    model.lm.config.use_cache = True
    hits = defaultdict(float)
    n = defaultdict(float)
    loss_sum = n_q = 0.0
    try:
        for req in reqs[rank::world]:
            rec = materialize(req)
            enc = model.encode(tok, rec, max_state=args.max_state, max_branch=args.max_branch)
            probs = model.probs_and_prefix(enc)[0]
            for p, q in zip(probs, rec["questions"]):
                p = p.float()
                y = q["label"]
                hits[q["qid"]] += float(int(p.argmax()) == y)
                n[q["qid"]] += 1
                loss_sum += float(-torch.log(p[y].clamp_min(1e-9)))
                n_q += 1
    finally:
        model.lm.config.use_cache = False
        if was_training:
            model.train()

    # the 11-question schema is fixed and identical on every rank, so the reduction
    # order is fixed too and no cross-rank gather is needed
    qids = sorted(QUESTIONS)
    qtypes = {q: QUESTIONS[q]["type"] for q in qids}
    flat = reduce_counts([hits[q] for q in qids] + [n[q] for q in qids] + [loss_sum, n_q], device, world)
    h, c = flat[:len(qids)], flat[len(qids):2 * len(qids)]
    loss_sum, n_q = flat[-2], flat[-1]

    per_q = {q: h[i] / c[i] for i, q in enumerate(qids) if c[i]}
    out = {"val/loss": loss_sum / max(n_q, 1),
           "val/accuracy": sum(h) / max(sum(c), 1),
           "val/questions": sum(c)}
    for t in ("noul", "choice", "score"):
        idx = [i for i, q in enumerate(qids) if qtypes.get(q) == t]
        if idx and sum(c[i] for i in idx):
            out[f"val/acc_{t}"] = sum(h[i] for i in idx) / sum(c[i] for i in idx)
    for q, acc in per_q.items():
        out[f"val_q/{q}"] = acc
    return out


# --------------------------------------------------------------------------- #

def save(model, tok, out, args, revision, extra=None):
    os.makedirs(out, exist_ok=True)
    model.lm.save_pretrained(out)
    # absolute for a local base directory, so a checkpoint loads from any working directory
    base = os.path.abspath(args.base) if os.path.isdir(args.base) else args.base
    torch.save({"head": model.head.state_dict(), "base": base, "base_revision": revision,
                "lora": args.lora, "head_dim": args.head_dim, "option_isolation": False,
                "special_embeddings": False, "weights_dtype": "fp32",
                "max_state": args.max_state, "max_branch": args.max_branch,
                "holdout": [], "args": vars(args), **(extra or {})}, os.path.join(out, "head.pt"))
    tok.save_pretrained(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen3.5-0.8B-Base", help="local dir or Hub id of the base model")
    ap.add_argument("--base_revision", default="")
    ap.add_argument("--data", default="data/medjev-v1/train.jsonl")
    ap.add_argument("--val_data", default="data/medjev-v1/development.jsonl")
    ap.add_argument("--out", default="runs/medjev-0.8b")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="upstream's 0.8B recipe; the pointer head trains from scratch")
    ap.add_argument("--head_lr", type=float, default=0.0, help="0 = same as --lr")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=256)
    ap.add_argument("--lora_targets", choices=["all", "dense", "attn", "qv"], default="all")
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass per rank; every question of a record is its own row")
    ap.add_argument("--accum", type=int, default=8,
                    help="micro-batches per optimizer step. Effective batch = world_size x batch x accum; "
                         "with 3 ranks use 3 to keep the single-GPU recipe's effective batch of ~8")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="bf16 autocast with fp32 master weights (CUDA only)")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    ap.add_argument("--max_state", type=int, default=MAX_STATE)
    ap.add_argument("--max_branch", type=int, default=MAX_BRANCH)
    ap.add_argument("--questions_per_record", type=int, default=0,
                    help="train on this many randomly chosen questions per record per epoch (0 = all ~9); "
                         "the state is re-run per question on this hybrid backbone, so this scales cost directly")
    ap.add_argument("--max_records", type=int, default=0, help="first N training records only (0 = all)")
    ap.add_argument("--ord_w", type=float, default=0.3,
                    help="weight of the ranked probability score on score questions; MedJev's three score "
                         "variables are genuinely ordered, so ordinal-aware loss is on by default")
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--save_every", type=int, default=0, help="also write an intermediate checkpoint every N optimizer steps (0 = off)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--val_every", type=int, default=250, help="validate every N optimizer steps (0 = off)")
    ap.add_argument("--val_records", type=int, default=400, help="development records used for in-training validation")
    ap.add_argument("--wandb", type=int, choices=[0, 1], default=1)
    ap.add_argument("--wandb_project", default="medjev")
    ap.add_argument("--wandb_name", default="", help="defaults to the basename of --out")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rank, world, dev = setup_dist()
    if a.device == "cpu":
        dev = "cpu"
    is_main = rank == 0
    if a.dtype == "bf16" and not str(dev).startswith("cuda"):
        ap.error("--dtype bf16 requires CUDA")

    out_dir = Path(a.out)
    if is_main:
        if out_dir.exists():
            ap.error(f"refusing to overwrite an existing run: {a.out}")
        out_dir.mkdir(parents=True)
    if world > 1:
        dist.barrier()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if str(dev).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()
    revision = a.base_revision or None

    tok = load_tokenizer(a.base, revision=revision)
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision, head_dim=a.head_dim,
                          lora_targets=a.lora_targets, dtype=torch.float32)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False

    broadcast_parameters(model, world)

    reqs = load_records(a.data, source=SOURCE)
    if a.max_records:
        reqs = reqs[: a.max_records]
    weights_by_q = class_weights(reqs) if a.class_weight == "balanced" else None

    val_reqs = []
    if a.val_every and a.val_records and os.path.exists(a.val_data):
        val_reqs = load_records(a.val_data, source=SOURCE)[: a.val_records]

    # every rank shuffles identically, then takes a disjoint stride; the tail is
    # dropped so all ranks run the same number of micro-batches (a rank that ran
    # one fewer would hang the next all_reduce)
    rng.shuffle(reqs)
    per_rank = len(reqs) // world
    shard = reqs[rank * per_rank: (rank + 1) * per_rank] if world > 1 else reqs

    n_q = sum(len(r["questions"]) for r in reqs)
    if is_main:
        print(f"world={world} device={dev} trainable={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M "
              f"records={len(reqs)} ({per_rank}/rank) questions={n_q} state<={a.max_state} val={len(val_reqs)}", flush=True)

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    trainable = model.trainable_parameters()
    micro_per_epoch = math.ceil(len(shard) / a.batch)
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.05)

    run = None
    if is_main:
        with open(out_dir / "training_config.json", "w") as f:
            json.dump({"args": vars(a), "device": str(dev), "world_size": world, "records": len(reqs),
                       "records_per_rank": per_rank, "questions": n_q, "optimizer_steps": steps,
                       "effective_batch": world * a.batch * a.accum}, f, indent=2)
        if a.wandb:
            import wandb
            run = wandb.init(project=a.wandb_project, name=a.wandb_name or out_dir.name,
                             config={**vars(a), "world_size": world, "records": len(reqs),
                                     "questions": n_q, "optimizer_steps": steps,
                                     "effective_batch": world * a.batch * a.accum})

    log = open(out_dir / f"train_log_rank{rank}.jsonl", "w")
    model.train()
    t0 = time.time()
    acc = Counter()
    step = seen = rows = truncated = tokens = 0
    peak = 0
    for ep in range(a.epochs):
        if ep:
            # reshuffle identically on every rank, keeping the shards disjoint
            ep_rng = random.Random(f"{a.seed}:epoch{ep}")
            order = list(range(len(reqs)))
            ep_rng.shuffle(order)
            shard = [reqs[i] for i in order[rank * per_rank: (rank + 1) * per_rank]] if world > 1 else [reqs[i] for i in order]
        for mb in range(micro_per_epoch):
            chunk = shard[mb * a.batch: (mb + 1) * a.batch]
            recs, encs = [], []
            for req in chunk:
                item_rng = random.Random(source_seed(a.seed, f"{ep}:{req['_meta']['id']}"))
                # option order is reshuffled every epoch; nothing else is augmented
                v = permute_choice_options(subsample_questions(req, item_rng, a.questions_per_record), item_rng)
                rec = materialize(v)
                enc = model.encode(tok, rec, max_state=a.max_state, max_branch=a.max_branch)
                truncated += int(enc["state_truncated"])
                recs.append(rec)
                encs.append(enc)
                rows += len(rec["questions"])
                tokens += len(enc["seg"]) - enc["seg"].count(0) + len(rec["questions"]) * enc["seg"].count(0)
            with autocast:
                logits_b = model.forward_batch(encs)
            loss = 0.0
            for logits, rec in zip(logits_b, recs):
                terms = []
                for z, q in zip(logits, rec["questions"]):
                    term = question_loss(z.float(), q, dev, a.ord_w)
                    if weights_by_q is not None:
                        term = term * weights_by_q[q["qid"]][label_key(q)]
                    terms.append(term)
                ce = sum(terms) / len(terms)
                acc["ce"] += ce.item()
                acc["n"] += 1
                loss = loss + ce
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            (loss / (len(chunk) * a.accum)).backward()
            seen += len(chunk)
            if str(dev).startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated())
            if (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch:
                all_reduce_grads(trainable, world)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % a.log_every == 0:
                    loc = reduce_counts([acc["ce"], acc["n"]], dev, world)
                    entry = {"epoch": ep, "step": step, "of": steps, "train/loss": loc[0] / max(loc[1], 1),
                             "lr": sched.get_last_lr()[0], "records": seen * world, "rows": rows * world,
                             "s_per_record": (time.time() - t0) / max(seen, 1),
                             "eta_hours": (time.time() - t0) / max(seen, 1) * (a.epochs * per_rank - seen) / 3600,
                             "peak_gb": peak / 2**30}
                    if is_main:
                        print(f"ep{ep} step {step}/{steps} loss {entry['train/loss']:.4f} "
                              f"{entry['s_per_record']:.2f}s/rec eta {entry['eta_hours']:.1f}h peak {entry['peak_gb']:.1f}GB", flush=True)
                        log.write(json.dumps(entry) + "\n")
                        log.flush()
                        if run:
                            run.log({k: v for k, v in entry.items() if k not in ("epoch", "of")}, step=step)
                    acc = Counter()

                if a.val_every and val_reqs and (step % a.val_every == 0 or step == steps):
                    vt = time.time()
                    metrics = validate(model, tok, val_reqs, a, rank, world, dev)
                    if is_main:
                        print(f"  val step {step}: loss {metrics['val/loss']:.4f} acc {metrics['val/accuracy']:.4f} "
                              f"(noul {metrics.get('val/acc_noul', float('nan')):.3f} "
                              f"choice {metrics.get('val/acc_choice', float('nan')):.3f} "
                              f"score {metrics.get('val/acc_score', float('nan')):.3f}) "
                              f"in {time.time()-vt:.0f}s", flush=True)
                        log.write(json.dumps({"step": step, **metrics}) + "\n")
                        log.flush()
                        if run:
                            run.log(metrics, step=step)

                if a.save_every and step % a.save_every == 0:
                    assert_replicas_agree(trainable, dev, world, f"step {step}")
                    if is_main:
                        save(model, tok, os.path.join(a.out, f"step-{step}"), a, revision,
                             extra={"step": step, "epoch": ep})
                        print(f"checkpoint step-{step}", flush=True)

    assert_replicas_agree(trainable, dev, world, "end of training")
    if is_main:
        save(model, tok, a.out, a, revision)
        with open(out_dir / "training_metrics.json", "w") as f:
            json.dump({"wall_seconds": time.time() - t0, "records_seen": seen * world,
                       "question_rows": rows * world, "forward_tokens": tokens * world,
                       "truncated_states": truncated * world, "optimizer_steps": step,
                       "peak_device_bytes": peak, "device": str(dev), "world_size": world,
                       "dtype": a.dtype}, f, indent=2)
        print(f"saved {a.out} in {(time.time()-t0)/3600:.2f}h", flush=True)
        if run:
            run.finish()
    log.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
