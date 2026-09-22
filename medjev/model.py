"""The decision model: a frozen causal-LM backbone, a LoRA adapter, and a pointer
head that scores each option against a `<decide>` position. No text generation.

Derived from `kev/kev/model.py` (Jared Palmer, Apache-2.0) — see NOTICE. MedJev
vendors it so the package stands alone. Two deliberate differences from upstream:

* `MAX_STATE` is 2048, not 384. Clinical notes are median ~670 tokens and 384
  costs 2.7 micro points; `docs/state_budget.md` has the measured sweep.
* The Apple-Silicon paths are gone. MedJev requires CUDA — flash-linear-attention's
  Gated DeltaNet kernels are Triton-only — so the MPS shape-bucketing and the
  `eager` attention fallback that upstream needs are dead weight here.

Two forward forms, picked by the backbone:

* **Packed** (attention-only bases, e.g. Qwen3): state and all question branches in
  one sequence under a block-causal mask, so a token reads the state and its own
  question but never a sibling question.
* **Rows** (hybrid bases, e.g. Qwen3.5): the Gated DeltaNet layers are recurrent
  and ignore attention masks, so each question becomes its own causal row with the
  state repeated. Isolation is then exact by construction. This is the path MedJev
  actually trains and serves on; the packed path is kept so an attention-only base
  still works.

At serving time the state is encoded once and the branches read its cache
(`probs_and_prefix`), which is why answering all of a note's questions costs about
as much as answering one.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Rarely-used Qwen special tokens reused as delimiters (state, question, option,
# end-of-option, decide) so no embedding rows have to be added or trained; the LoRA
# adapter is what gives them meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]

# MedJev's state budget. 2,048 keeps 99% of notes whole for ~6% more compute than
# 1,024 (which keeps 85%) — the cap only binds on the tail of the length
# distribution, so the cost is mean(min(note_len, cap)), not cap. Beyond 2,048 the
# measured gain is nil, and truncating from the tail rather than the head is far
# worse. See docs/state_budget.md.
#
# MAX_BRANCH is the state+branch TOTAL, not the branch alone (see the check in
# `encode`). MedJev's longest question branch is 144 tokens.
MAX_STATE, MAX_BRANCH = 2048, 2560

OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: state/instruction tokens, and <decide>

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce a delimiter token —
    option boundaries stay unforgeable. The fast tokenizer ignores
    `split_special_tokens`, so `<|name|>` is rewritten to `<¦name¦>` first."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False):
    """Pack one record: `[<state> ...]` then, per question,
    `[<q> instr <opt> o </opt> ... <decide>]`.

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart
    after the state), decide_idx [Q], opt_idx [Q][K] (the `</opt>` token of each
    option) and opt (per-token option index: OPT_NONE for state/instruction,
    0..K-1 inside an option span, OPT_DECIDE for `<decide>`).

    The state is truncated from the HEAD — `state_tokens[:max_state - 1]` — which is
    measurably better than keeping the tail (docs/state_budget.md); `state_truncated`
    reports whether it happened.

    option_isolation: every option span becomes its own sub-branch seeing only the
    state, the instruction and itself, all spans share position ids, and `<decide>`
    sits at a fixed position after the longest span, making the readout
    permutation-invariant by construction. Not available on hybrid backbones.
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    s = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(s), [0] * len(s), list(range(len(s))), [OPT_NONE] * len(s)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(s):
            raise ValueError(f"branch too long: {len(br)}")
        base = len(ids)
        p0 = len(s)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = (list(range(p0, p0 + len(instr)))
                      + [p0 + len(instr) + i for sp in spans for i in range(len(sp))]
                      + [p0 + len(instr) + longest])
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += br
        seg += [k] * len(br)
        pos += br_pos
        opt += br_opt
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation,
            "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]],
            "state_truncated": len(state_tokens) + 1 > max_state}


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None):
    """Batched block-causal mask, additive [B, 1, L, L], right-padded to the longest
    sequence: attend(i, j) iff j <= i and (seg[j] == 0 or seg[j] == seg[i]).

    Padded keys are masked for every query; padded query rows keep the diagonal so
    no row is fully masked (finfo.min rather than -inf, so softmax stays finite
    either way). Real tokens never see pads, because pads sit after them and belong
    to no segment.

    opts (option isolation): inside a question, an option-span token attends to the
    state, the instruction and its own span only; `<decide>` attends to everything
    in its question."""
    length_max = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), length_max), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(length_max, length_max, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), length_max), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(length_max, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), length_max, length_max, dtype=dtype,
                       device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide",
    "opts"} — the branch tokens of question k with their state-continuing positions
    and the readout offsets *within the branch*. Feeding state + rows[k] as one
    causal row is equivalent to the packed block-causal form for that question on
    any architecture: the row holds exactly the tokens question k may attend to, at
    the same positions."""
    seg = enc["seg"]
    n_state = seg.count(0)
    rows, start = [], n_state
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1                                # <decide> is the last token of its branch
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end],
                     "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:n_state], enc["pos"][:n_state], rows


class PointerHead(nn.Module):
    """Scores each option's `</opt>` hidden state against the question's `<decide>`
    hidden state. A softmax over those scores is the answer distribution, so the
    output is always a proper distribution over exactly the allowed options."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):        # [d], [K, d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256,
                 option_isolation=False, special_embeddings=False, lora_targets="all",
                 dtype=torch.float32):
        super().__init__()
        # backbone only (no vocab head): nothing is ever generated. SDPA accepts the
        # arbitrary additive mask the packed path builds.
        attn = attn or "sdpa"
        self.lm = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=dtype, attn_implementation=attn).model
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        cfg = self.lm.config
        # Hybrid backbones (Qwen3.5: Gated DeltaNet layers, recurrent) cannot honour
        # the block-causal mask, so each question runs as its own causal row.
        self.hybrid = "linear_attention" in set(getattr(cfg, "layer_types", None) or [])
        if self.hybrid and option_isolation:
            raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
        self.option_isolation = option_isolation
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = ({"trainable_token_indices": {"embed_tokens": [tok.convert_tokens_to_ids(t) for t in SPECIAL]}}
                     if special_embeddings else {})
            targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       # "all" minus the DeltaNet projections on hybrids (retention ablation)
                       "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                       "qv": ["q_proj", "v_proj"]}[lora_targets]
            if self.hybrid and lora_targets in ("all", "attn"):
                # Gated DeltaNet projections (transformers 5 names) plus the mixer's out_proj
                targets = targets + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            peft_cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora,
                                  lora_dropout=0.05, target_modules=targets, **extra)
            self.lm = get_peft_model(self.lm, peft_cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        """`encode()` with this model's option-isolation setting; use this rather
        than the bare function from serving and evaluation code."""
        return encode(tok, rec, option_isolation=self.option_isolation, **kw)

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch. Pads are masked keys
        and sit after every real token, so padding never changes a real token's
        hidden state."""
        length = max(len(e["ids"]) for e in encs)
        ids = torch.full((len(encs), length), self.pad_id, device=self.device)
        pos = torch.zeros((len(encs), length), dtype=torch.long, device=self.device)
        for b, e in enumerate(encs):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"], device=self.device)
            pos[b, : len(e["pos"])] = torch.tensor(e["pos"], device=self.device)
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        lm_dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=lm_dtype,
                                 opts=[e["opt"] for e in encs] if isolate else None, length=length)
        # the head stays fp32
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)])
                for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def forward_rows_batch(self, encs):
        """Row form: every question of every record is one causal row (state tokens +
        its branch tokens), right-padded into a single batch. Same nested logits as
        `forward_batch`. Isolation is exact because the rows are independent; the
        state is recomputed per row, which training accepts and serving avoids via
        the prefix cache."""
        rows, owners = [], []
        for b, e in enumerate(encs):
            s, s_pos, brs = rows_of(e)
            for r in brs:
                rows.append((s + r["ids"], s_pos + r["pos"], len(s) + r["decide"],
                             [len(s) + o for o in r["opts"]]))
                owners.append(b)
        length = max(len(ids) for ids, *_ in rows)
        ids = torch.full((len(rows), length), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
        for i, (rid, rpos, _, _) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device)
            pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device)
            att[i, : len(rid)] = 1
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att).last_hidden_state.float()
        out = [[] for _ in encs]
        for i, (b, (_, _, d, oi)) in enumerate(zip(owners, rows)):
            out[b].append(self.head(h[i, d], h[i, torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """List of logits tensors, one per question."""
        if self.hybrid:
            return self.forward_rows_batch([enc])[0]
        return self._readout(self.hidden(enc), enc)

    def forward_batch(self, encs):
        """Per record, per question logits, from one padded forward pass."""
        if self.hybrid:
            return self.forward_rows_batch(encs)
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    # --- state-prefix reuse (serving) ------------------------------------------
    # Exact by construction: branch tokens never attend across questions, and the
    # state never sees the branches, so the state's hidden states and KV are
    # identical with or without the branches present.

    def _branch_rows_from_prefix(self, enc, cache):
        """Hybrid serving: replicate the cached state once per question and run the
        branches as causal rows — the `forward_rows_batch` layout minus the
        recomputed state. The cache is consumed (replicated, then extended)."""
        s, _, rows = rows_of(enc)
        n_q = len(rows)
        cache.reorder_cache(torch.zeros(n_q, dtype=torch.long, device=self.device))
        width = max(len(r["ids"]) for r in rows)
        ids = torch.full((n_q, width), self.pad_id, device=self.device)
        pos = torch.zeros((n_q, width), dtype=torch.long, device=self.device)
        att = torch.zeros((n_q, len(s) + width), dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            ids[i, : len(r["ids"])] = torch.tensor(r["ids"], device=self.device)
            pos[i, : len(r["pos"])] = torch.tensor(r["pos"], device=self.device)
            att[i, : len(s) + len(r["ids"])] = 1
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att,
                    past_key_values=cache, use_cache=True).last_hidden_state.float()
        return [F.softmax(self.head(h[i, r["decide"]],
                                    h[i, torch.tensor(r["opts"], device=self.device)]), -1).cpu()
                for i, r in enumerate(rows)]

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only -> (n_state_tokens, kv cache, state hidden states)."""
        from transformers import DynamicCache
        n_state = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:n_state]], device=self.device)
        pos = torch.tensor([enc["pos"][:n_state]], device=self.device)
        # the cache must know the layer types (hybrid backbones keep recurrent and
        # conv state per DeltaNet layer)
        out = self.lm(input_ids=ids, position_ids=pos,
                      past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        return n_state, out.past_key_values, out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One pass that also returns the reusable state prefix, so a cache miss costs
        a single forward pass rather than two. This is MedJev's serving and
        evaluation path: the state is encoded once per record however many questions
        it carries."""
        from transformers import DynamicCache
        n_state = enc["seg"].count(0)
        if self.hybrid:
            # recurrent layers cannot be cropped back to the state, so copy the cache
            # before consuming it
            import copy
            n_state, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, copy.deepcopy(cache)), (n_state, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt,
                                 opts=[enc["opt"]] if enc.get("option_isolation") else None)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask,
                      past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        h = out.last_hidden_state[0].float()
        # keep the state only (negative = drop that many trailing tokens)
        out.past_key_values.crop(-(len(enc["ids"]) - n_state))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (n_state, out.past_key_values, h[:n_state].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """`probs()` for a record whose state tokens equal the cached prefix's; only
        the branches run. The cache is cropped back to the state afterwards so it can
        be reused."""
        n_state, cache, h_state = prefix
        if enc["seg"].count(0) != n_state:
            raise ValueError("prefix does not match this record's state")
        if self.hybrid:
            import copy
            return self._branch_rows_from_prefix(enc, copy.deepcopy(cache))   # stored prefix stays pristine
        ids = torch.tensor([enc["ids"][n_state:]], device=self.device)
        pos = torch.tensor([enc["pos"][n_state:]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt,
                                 opts=[enc["opt"]] if enc.get("option_isolation") else None)[:, :, n_state:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache,
                          attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - n_state))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
