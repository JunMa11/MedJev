"""Side-by-side comparison UI: base Qwen3.5-0.8B vs hosted Jev vs MedJev.

    .venv/bin/python -m medjev.serve_compare --run runs/medjev-0.8b/step-5000

Serves a single-page interface on http://127.0.0.1:8765 that runs a slice of a
split (100 cases by default, up to all of it) through all three systems at once,
with a live progress bar and a live accuracy readout per system, and then lets
you open any individual case to see where they disagreed.

The three systems answer the *same* typed questions through three different
readouts, and the UI is built around that contrast:

  * ``base``   - Qwen3.5-0.8B-Base, zero shot, read out of the next-token letter
                 logits exactly as ``medjev.base_probe --prompt plain`` does.
  * ``jev``    - the hosted TypeSafe System One API. ``data/results/jev-<split>.jsonl``
                 already holds a full recorded run for development and test, so
                 the default is to replay it; ``--jev-source live`` re-buys the
                 answers from the API instead.
  * ``medjev`` - the fine-tuned checkpoint through the *serving* path: the state
                 is encoded once and every question branch reads its cache
                 (``probs_and_prefix``), so the reported latency is the real
                 per-record cost of answering all of a record's variables.

Models load lazily on first use and every GPU call is serialised behind one
lock, so this stays polite on a shared box.
"""
import argparse
import atexit
import collections
import http.server
import json
import mimetypes
import os
import socketserver
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request

import numpy as np

# Inference output of the reference systems (hosted Jev, the zero-shot Qwen probes).
# Kept under data/ rather than runs/: these are fixed evaluation inputs, not artifacts of
# a training run, and the Jev answers were paid for once and are replayed thereafter.
RESULTS = "data/results"

HERE = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.join(HERE, "webui")
ROOT = os.path.dirname(HERE)
SYSTEMS = ("base", "jev", "medjev")
JEV_LIMIT = 100          # the only case count at which the live (paid) Jev API is called


# --------------------------------------------------------------------------- #
# question plumbing
# --------------------------------------------------------------------------- #
def keys_and_options(q):
    """(keys, human-readable option text) in the order every system scores them.

    Same normalisation kev uses: noul -> false/true, score -> level index."""
    if q["type"] == "noul":
        keys = ["false", "true"]
        return keys, [q["criteria"][k] for k in keys]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))], list(q["criteria"])
    keys = list(q["criteria"])
    return keys, [q["criteria"][k] or k for k in keys]


def gold_key(q):
    """The gold option key, or None for an unlabelled (pasted) note."""
    if q.get("label") is None:
        return None
    if q["type"] == "noul":
        return str(bool(q["label"])).lower()
    return str(q["label"])


def dummy_label(q):
    """materialize() insists on a label; probabilities never depend on it."""
    return False if q["type"] == "noul" else (0 if q["type"] == "score" else list(q["criteria"])[0])


def question_view(questions):
    """Question specs plus gold, in the exact option order the systems score."""
    view = {}
    for qid, q in questions.items():
        keys, desc = keys_and_options(q)
        g = gold_key(q)
        view[qid] = {"type": q["type"], "instructions": q["instructions"],
                     "keys": keys, "options": desc,
                     "gold": g, "gold_index": keys.index(g) if g is not None else None}
    return view


# --------------------------------------------------------------------------- #
# the three systems
# --------------------------------------------------------------------------- #
class Engine:
    """The three systems.

    `workers` maps a local system to the port of a worker process that owns it.
    A worker is masked to one physical GPU with CUDA_VISIBLE_DEVICES, which is a
    process-wide setting — that is the whole reason the two local models run in
    separate processes rather than separate threads. When a system is proxied it
    no longer takes `self.gpu`, so base and MedJev genuinely run at the same time
    on their own cards and neither one's latency includes the other's."""

    def __init__(self, run, device, dtype, max_state, max_branch, base_model, base_batch,
                 jev_model, jev_source, jev_cache_path, workers=None):
        self.run_path, self.device, self.dtype = run, device, dtype
        self.max_state, self.max_branch = max_state, max_branch
        self.base_model_path, self.base_batch = base_model, base_batch
        self.jev_model, self.jev_source = jev_model, jev_source
        self.gpu = threading.Lock()
        self.loading = threading.Lock()
        self._medjev = self._base = None
        self.workers = workers or {}
        self.jev_cache = self._load_jev_cache(jev_cache_path)
        self.status = {"medjev": "idle", "base": "idle",
                       "jev": "cache" if self.jev_source == "cache" else
                              ("ready" if self._jev_key() else "no-key")}

    # ---- loading --------------------------------------------------------- #
    def _jev_key(self):
        key = os.environ.get("TYPESAFE_API_KEY")
        if key:
            return key
        env = os.path.join(ROOT, ".env")
        if os.path.exists(env):
            for line in open(env):
                if line.startswith("TYPESAFE_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return None

    def _load_jev_cache(self, path):
        """A recorded `medjev.run_jev` run, keyed by record index."""
        if not path or not os.path.exists(path):
            return {}
        cache = {}
        for line in open(path):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "error" not in row and row.get("answers"):
                cache[row["idx"]] = row
        return cache

    def medjev(self):
        with self.loading:
            if self._medjev is None:
                import torch
                from medjev.checkpoint import load
                self.status["medjev"] = "loading"
                tok, model = load(self.run_path, self.device,
                                  dtype=torch.bfloat16 if self.dtype == "bf16" else torch.float32)
                model.lm.config.use_cache = True
                self._medjev = (tok, model)
                self._warm_medjev()
                self.status["medjev"] = "ready"
        return self._medjev

    def _warm_medjev(self):
        """One throwaway pass: the first CUDA call pays kernel autotuning, and the UI
        reports latency per record, so that cost must not land on the first case."""
        import torch
        from medjev.records import materialize
        tok, model = self._medjev
        rec = materialize({"state": "warm up", "questions": {"w": {
            "type": "noul", "instructions": "ignore", "label": False, "src": "warmup",
            "criteria": {"true": "yes", "false": "no"}}}})
        with torch.no_grad():
            model.probs_and_prefix(model.encode(tok, rec, max_state=self.max_state, max_branch=self.max_branch))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def base(self):
        with self.loading:
            if self._base is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.status["base"] = "loading"
                tok = AutoTokenizer.from_pretrained(self.base_model_path)
                tok.padding_side = "right"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
                model = AutoModelForCausalLM.from_pretrained(self.base_model_path, dtype=dtype)
                model = model.to(self.device).eval()
                self._base = (tok, model)
                with torch.no_grad():                      # same warm-up argument as MedJev
                    model(**tok(["warm up\nAnswer:"], return_tensors="pt").to(self.device))
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                self.status["base"] = "ready"
        return self._base

    # ---- inference ------------------------------------------------------- #
    def run_medjev(self, rec):
        import torch
        from medjev.records import materialize
        tok, model = self.medjev()
        questions = rec["questions"]
        req = {"state": rec["state"], "questions": {
            qid: {**{k: v for k, v in q.items() if k in ("type", "instructions", "criteria")},
                  "label": q.get("label") if q.get("label") is not None else dummy_label(q),
                  "src": "ui"}
            for qid, q in questions.items()}}
        with self.gpu:
            enc = model.encode(tok, materialize(req),
                               max_state=self.max_state, max_branch=self.max_branch)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                probs, _ = model.probs_and_prefix(enc)   # state once, branches from its cache
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency = 1000 * (time.perf_counter() - t0)
        return {"probs": {qid: np.asarray(p, dtype=float).tolist() for p, qid in zip(probs, questions)},
                "latency_ms": round(latency, 1),
                "state_truncated": bool(enc["state_truncated"]),
                "state_tokens": enc["seg"].count(0)}

    def run_base(self, rec):
        """Zero-shot letter logits, one prompt per question (mirrors medjev.base_probe)."""
        import torch
        from medjev.base_probe import LETTERS
        tok, model = self.base()
        state, questions = rec["state"], rec["questions"]
        ids = tok(state, add_special_tokens=False)["input_ids"]
        truncated = len(ids) > 3072
        if truncated:
            state = tok.decode(ids[:3072])
        prompts, meta = [], []
        for qid, q in questions.items():
            keys, desc = keys_and_options(q)
            body = "\n".join(f"{LETTERS[i]}. {d}" for i, d in enumerate(desc))
            prompts.append(f"{state}\n{q['instructions']}\n{body}\nAnswer:")
            meta.append((qid, len(keys)))
        letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
        out = {}
        with self.gpu:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for b in range(0, len(prompts), self.base_batch):
                    chunk = prompts[b: b + self.base_batch]
                    e = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
                    last = e["attention_mask"].sum(1) - 1
                    h = model.model(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).last_hidden_state
                    logits = model.lm_head(h[torch.arange(len(chunk), device=self.device), last]).float()
                    for (qid, k), row in zip(meta[b: b + self.base_batch], logits):
                        out[qid] = torch.softmax(row[letter_ids[:k]], -1).tolist()
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency = 1000 * (time.perf_counter() - t0)
        return {"probs": out, "latency_ms": round(latency, 1), "state_truncated": truncated}

    def _jev_vectors(self, answers, questions):
        out = {}
        for qid, q in questions.items():
            keys, _ = keys_and_options(q)
            a = answers.get(qid)
            if a is None:
                out[qid] = None
                continue
            if a.get("type") == "noul" or q["type"] == "noul":
                p = float(a.get("noul", 0.5))
                vec = [1 - p, p]
            else:
                probs = a.get("probabilities") or {}
                vec = [float(probs.get(k, 0.0)) for k in keys]
                if sum(vec) <= 0:
                    vec = [1 / len(keys)] * len(keys)
            s = sum(vec)
            out[qid] = [v / s for v in vec]
        return out

    def run_jev(self, rec):
        questions = rec["questions"]
        if self.jev_source == "cache":
            row = self.jev_cache.get(rec.get("idx"))
            if row is None:
                raise RuntimeError(f"record {rec.get('idx')} is not in the recorded Jev run")
            return {"probs": self._jev_vectors(row["answers"], questions),
                    "latency_ms": row.get("latency_ms"), "state_truncated": False,
                    "model": row.get("model"), "cached": True}
        from medjev.run_jev import api_request, call
        key = self._jev_key()
        if not key:
            raise RuntimeError("no TYPESAFE_API_KEY in the environment or .env")
        payload = {"state": rec["state"], "questions": {
            qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            for qid, q in questions.items()}}
        resp, latency, _ = call(api_request(payload, self.jev_model), key, timeout=120)
        return {"probs": self._jev_vectors(resp.get("answers") or {}, questions),
                "latency_ms": round(latency, 1), "state_truncated": False,
                "model": resp.get("model"), "cached": False}

    def run_remote(self, port, rec):
        """Hand one record to the worker process that owns this system's GPU."""
        body = json.dumps({"state": rec["state"], "questions": {
            qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            for qid, q in rec["questions"].items()}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/infer", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def run(self, system, rec):
        port = self.workers.get(system)
        if port:
            return self.run_remote(port, rec)
        return {"base": self.run_base, "jev": self.run_jev, "medjev": self.run_medjev}[system](rec)


# --------------------------------------------------------------------------- #
# batch job
# --------------------------------------------------------------------------- #
class Job:
    """One benchmark run: every selected system over the same N records.

    Each system gets its own thread. With one worker process per local model (the
    default) the two models sit on different GPUs and overlap fully. In
    --single-process mode they instead share one card and take turns on
    `engine.gpu`, so that neither one's reported latency measures contention with
    the other."""

    def __init__(self, engine, records, systems):
        self.engine, self.records = engine, records
        self.lock = threading.Lock()
        self.cancelled = False
        self.started = time.time()
        self.finished = None
        self.systems = {}
        for s in systems:
            self.systems[s] = {
                "state": "queued", "done": 0, "total": len(records), "hit": 0, "n": 0,
                "lat": [], "errors": 0, "last_error": None,
                "per_q": collections.defaultdict(lambda: {"hit": 0, "n": 0}),
            }
        self.preds = {s: {} for s in systems}
        self.threads = []

    def start(self):
        for s in self.systems:
            t = threading.Thread(target=self._work, args=(s,), daemon=True)
            t.start()
            self.threads.append(t)

    def _work(self, system):
        st = self.systems[system]
        with self.lock:
            st["state"] = "loading" if system not in self.engine.workers and system != "jev" else "running"
        try:
            # only load in this process for a system this process actually owns;
            # a proxied system is already loaded inside its own worker.
            if system not in self.engine.workers:
                if system == "medjev":
                    self.engine.medjev()
                elif system == "base":
                    self.engine.base()
        except Exception as e:
            with self.lock:
                st["state"], st["last_error"] = "failed", f"{type(e).__name__}: {e}"
            return
        with self.lock:
            st["state"] = "running"
        for rec in self.records:
            if self.cancelled:
                break
            try:
                res = self.engine.run(system, rec)
            except Exception as e:
                traceback.print_exc()
                with self.lock:
                    st["errors"] += 1
                    st["last_error"] = f"{type(e).__name__}: {e}"
                    st["done"] += 1
                continue
            with self.lock:
                self.preds[system][rec["idx"]] = res["probs"]
                if res.get("latency_ms") is not None:
                    st["lat"].append(res["latency_ms"])
                for qid, q in rec["questions"].items():
                    p = res["probs"].get(qid)
                    if not p:
                        continue
                    keys, _ = keys_and_options(q)
                    gi = keys.index(gold_key(q))
                    ok = int(int(np.argmax(p)) == gi)
                    st["hit"] += ok
                    st["n"] += 1
                    st["per_q"][qid]["hit"] += ok
                    st["per_q"][qid]["n"] += 1
                st["done"] += 1
        with self.lock:
            st["state"] = "cancelled" if self.cancelled else "done"
            if all(v["state"] in ("done", "failed", "cancelled") for v in self.systems.values()):
                self.finished = time.time()

    def snapshot(self):
        with self.lock:
            out = {"running": any(v["state"] in ("queued", "loading", "running") for v in self.systems.values()),
                   "cancelled": self.cancelled,
                   "elapsed_s": round((self.finished or time.time()) - self.started, 1),
                   "total": len(self.records), "systems": {}}
            for s, st in self.systems.items():
                lat = sorted(st["lat"])
                out["systems"][s] = {
                    "state": st["state"], "done": st["done"], "total": st["total"],
                    "hit": st["hit"], "n": st["n"],
                    "accuracy": round(st["hit"] / st["n"], 4) if st["n"] else None,
                    "median_ms": round(lat[len(lat) // 2], 1) if lat else None,
                    "errors": st["errors"], "last_error": st["last_error"],
                    "per_q": {q: {"hit": v["hit"], "n": v["n"],
                                  "accuracy": round(v["hit"] / v["n"], 4) if v["n"] else None}
                              for q, v in st["per_q"].items()},
                }
            return out

    def case_results(self, idx):
        with self.lock:
            return {s: self.preds[s].get(idx) for s in self.preds if idx in self.preds[s]}


# --------------------------------------------------------------------------- #
# case library
# --------------------------------------------------------------------------- #
class Cases:
    def __init__(self, data_dir, split, limit):
        self.split, self.records = split, []
        for line in open(os.path.join(data_dir, f"{split}.jsonl")):
            line = line.strip()
            if not line:
                continue
            self.records.append(json.loads(line))
            if limit and len(self.records) >= limit:
                break
        self.by_idx = {r["idx"]: r for r in self.records}

    def summaries(self):
        return [{"idx": r["idx"], "chars": len(r["state"]), "n_questions": len(r["questions"]),
                 "preview": " ".join(r["state"].split())[:180]} for r in self.records]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(http.server.BaseHTTPRequestHandler):
    engine: Engine = None
    cases: Cases = None
    config: dict = None
    job: Job = None
    job_lock = threading.Lock()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if "/api/progress" not in self.path and "/api/cases" not in self.path:
            print(f"  {self.address_string()} {fmt % args}", flush=True)

    # -- helpers ----------------------------------------------------------- #
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        path = os.path.join(WEBUI, name)
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))

    # -- routes ------------------------------------------------------------ #
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path == "/api/meta":
                from medjev.labels import QUESTIONS
                specs = {}
                for qid, q in QUESTIONS.items():
                    keys, desc = keys_and_options(q)
                    specs[qid] = {"type": q["type"], "instructions": q["instructions"],
                                  "keys": keys, "options": desc}
                return self._send(200, {"questions": specs, "config": self.config,
                                        "status": self.engine.status,
                                        "split": self.cases.split,
                                        "n_cases": len(self.cases.records)})
            if url.path == "/api/cases":
                return self._send(200, {"cases": self.cases.summaries()})
            if url.path == "/api/progress":
                job = Handler.job
                return self._send(200, job.snapshot() if job else {"running": False, "systems": {}})
            if url.path.startswith("/api/case/"):
                idx = int(url.path.rsplit("/", 1)[1])
                rec = self.cases.by_idx.get(idx)
                if rec is None:
                    return self._send(404, {"error": f"no case {idx}"})
                job = Handler.job
                return self._send(200, {"idx": idx, "state": rec["state"],
                                        "questions": question_view(rec["questions"]),
                                        "results": job.case_results(idx) if job else {}})
            return self._static(os.path.basename(url.path))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad JSON: {e}"})
        try:
            if url.path == "/api/run":
                return self._start_job(payload)
            if url.path == "/api/stop":
                job = Handler.job
                if job:
                    job.cancelled = True
                return self._send(200, {"ok": True})
            if url.path == "/api/predict":
                return self._predict_one(payload)
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(200, {"error": f"{type(e).__name__}: {e}"})

    def _start_job(self, payload):
        with Handler.job_lock:
            if Handler.job and Handler.job.snapshot()["running"]:
                return self._send(409, {"error": "a run is already in progress"})
            systems = [s for s in (payload.get("systems") or SYSTEMS) if s in SYSTEMS]
            limit = int(payload.get("limit") or len(self.cases.records))
            records = self.cases.records[:limit]
            # Jev is answered live, one paid API call per record, so it is only ever
            # run at the one case count we agreed to spend on. Enforced here and not
            # only in the page, so no client can start a 2,895-case billing run.
            dropped = None
            if "jev" in systems and limit != JEV_LIMIT:
                systems = [x for x in systems if x != "jev"]
                dropped = f"jev only runs at {JEV_LIMIT} cases (live API); skipped for {limit}"
            if not systems:
                return self._send(400, {"error": dropped or "no systems selected"})
            job = Job(self.engine, records, systems)
            Handler.job = job
            job.start()
        return self._send(200, {"ok": True, "total": len(records), "systems": systems,
                                "dropped": dropped})

    def _predict_one(self, payload):
        """One ad-hoc note (the 'own note' pane), outside any batch run."""
        from medjev.labels import QUESTIONS
        state = (payload.get("note") or "").strip()
        if not state:
            return self._send(400, {"error": "empty note"})
        questions = {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                     for qid, q in QUESTIONS.items()}
        rec = {"idx": None, "state": state, "questions": questions}
        system = payload.get("system", "medjev")
        if system == "jev" and self.engine.jev_source == "cache":
            return self._send(200, {"system": system, "error": "Jev is in cached-replay mode; "
                                    "restart with --jev-source live to answer new notes"})
        res = self.engine.run(system, rec)
        res["system"] = system
        return self._send(200, res)


class WorkerHandler(http.server.BaseHTTPRequestHandler):
    """One model, one GPU, one endpoint. Started by the main server as a subprocess
    with CUDA_VISIBLE_DEVICES already set, so it can only see its own card."""
    engine: Engine = None
    system: str = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "system": self.system,
                                    "status": self.engine.status.get(self.system)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/infer":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            rec = json.loads(self.rfile.read(n) or b"{}")
            rec.setdefault("idx", None)
            return self._send(200, self.engine.run(self.system, rec))
        except Exception as e:
            traceback.print_exc()
            return self._send(200, {"error": f"{type(e).__name__}: {e}"})


def run_worker(a):
    """--worker <system>: serve exactly one model on exactly one GPU."""
    device, gpu = select_gpu(f"cuda:{a.worker_gpu}" if a.worker_gpu is not None else "auto")
    from medjev.model import MAX_BRANCH, MAX_STATE
    max_state = a.max_state or MAX_STATE
    max_branch = a.max_branch or (max_state + 512 if a.max_state else MAX_BRANCH)
    engine = Engine(a.run, device, a.dtype, max_state, max_branch, a.base_model, a.base_batch,
                    a.jev_model, "live", None)
    WorkerHandler.engine, WorkerHandler.system = engine, a.worker
    print(f"[worker {a.worker}] loading on {device} (physical GPU {gpu}) ...", flush=True)
    (engine.medjev if a.worker == "medjev" else engine.base)()
    print(f"[worker {a.worker}] ready on port {a.worker_port}", flush=True)
    Server(("127.0.0.1", a.worker_port), WorkerHandler).serve_forever()


def start_workers(a, gpus):
    """One subprocess per local system, each masked to its own GPU. Returns {system: port}."""
    procs, ports = [], {}
    for i, (system, gpu) in enumerate(gpus.items()):
        port = a.worker_port_base + i
        cmd = [sys.executable, "-u", "-m", "medjev.serve_compare",
               "--worker", system, "--worker-gpu", str(gpu), "--worker-port", str(port),
               "--run", a.run, "--base-model", a.base_model, "--base-batch", str(a.base_batch),
               "--dtype", a.dtype]
        if a.max_state:
            cmd += ["--max_state", str(a.max_state)]
        if a.max_branch:
            cmd += ["--max_branch", str(a.max_branch)]
        p = subprocess.Popen(cmd, cwd=ROOT)
        procs.append(p)
        ports[system] = port
    atexit.register(lambda: [p.terminate() for p in procs])

    deadline = time.time() + 600
    for system, port in ports.items():
        while True:
            if time.time() > deadline:
                raise SystemExit(f"worker {system} did not come up")
            dead = [p for p in procs if p.poll() is not None]
            if dead:
                raise SystemExit(f"worker process exited with code {dead[0].returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                    if json.loads(r.read()).get("ok"):
                        break
            except Exception:
                time.sleep(2)
    return ports


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def gpu_free_memory():
    """Free MiB per physical GPU, via nvidia-smi so nothing has to import torch yet."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return []


def select_gpu(requested):
    """Pick the least-loaded GPU, pin the process to it with CUDA_VISIBLE_DEVICES, and
    address it as cuda:0.

    This indirection is not cosmetic. Handing this model a non-zero device *string*
    silently returns wrong probabilities — measured on 30 development records,
    micro accuracy is 0.883 on cuda:0 but 0.359 on cuda:1 and cuda:2, with no error
    raised, and `torch.cuda.set_device` does not help. The DeltaNet Triton kernels
    do not follow a non-default device. Masking the other GPUs out gives the correct
    0.883 on every one of them, so that is the only way this server selects a GPU.

    Must run before torch is imported, hence nvidia-smi rather than torch.cuda."""
    if requested == "cpu":
        return "cpu", None
    free = gpu_free_memory()
    if not free:
        return "cpu", None
    if requested == "auto":
        gpu = max(range(len(free)), key=lambda i: free[i])
    else:
        gpu = int(requested.split(":")[1]) if ":" in requested else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return "cuda:0", gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/medjev-0.8b/step-5000", help="MedJev checkpoint directory")
    ap.add_argument("--base-model", default="Qwen3.5-0.8B-Base")
    ap.add_argument("--base-batch", type=int, default=16, help="prompts per forward pass for the zero-shot probe")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--jev-source", choices=["auto", "cache", "live"], default="auto",
                    help="auto: replay data/results/jev-<split>.jsonl if it exists, else call the API")
    ap.add_argument("--data", default="data/medjev-v1")
    ap.add_argument("--split", default="development")
    ap.add_argument("--cases", type=int, default=0,
                    help="expose only the first N records of the split (0 = all of it)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 is the serving default: no measured accuracy cost, 2.6x faster")
    ap.add_argument("--max_state", type=int, default=None)
    ap.add_argument("--max_branch", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--preload", action="store_true", help="(single-process mode) load both models up front")
    ap.add_argument("--base-gpu", type=int, default=None, help="physical GPU for the zero-shot probe")
    ap.add_argument("--medjev-gpu", type=int, default=None, help="physical GPU for the MedJev checkpoint")
    ap.add_argument("--single-process", action="store_true",
                    help="put both local models in this process on one GPU (they then take turns)")
    ap.add_argument("--worker", choices=["base", "medjev"], help=argparse.SUPPRESS)
    ap.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-port", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-port-base", type=int, default=8771, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return run_worker(a)
    if a.split == "test" and not a.allow_test:
        raise SystemExit("test split is locked: pass --allow-test to browse it")

    # Two GPUs, two processes. CUDA_VISIBLE_DEVICES is process-wide, and it is the
    # only placement that is numerically correct here (see select_gpu), so one model
    # per process is what "run them on separate GPUs" has to mean.
    workers, gpu_map = {}, {}
    if a.single_process:
        device, gpu_index = select_gpu(a.device)
    else:
        device, gpu_index = "cpu", None
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # the front end never touches a GPU
        free = gpu_free_memory()
        picks = sorted(range(len(free)), key=lambda i: -free[i])
        gpu_map = {"base": a.base_gpu if a.base_gpu is not None else (picks[0] if picks else 0),
                   "medjev": a.medjev_gpu if a.medjev_gpu is not None else (picks[1] if len(picks) > 1 else 0)}
        if gpu_map["base"] == gpu_map["medjev"]:
            print(f"warning: both models were assigned GPU {gpu_map['base']}; they will contend", flush=True)

    from medjev.model import MAX_BRANCH, MAX_STATE
    max_state = a.max_state or MAX_STATE
    max_branch = a.max_branch or (max_state + 512 if a.max_state else MAX_BRANCH)

    jev_cache_path = os.path.join(RESULTS, f"jev-{a.split}.jsonl")
    source = a.jev_source
    if source == "auto":
        source = "cache" if os.path.exists(jev_cache_path) else "live"

    if not a.single_process:
        print(f"starting workers: base on GPU {gpu_map['base']}, medjev on GPU {gpu_map['medjev']} ...", flush=True)
        workers = start_workers(a, gpu_map)

    engine = Engine(a.run, device, a.dtype, max_state, max_branch, a.base_model, a.base_batch,
                    a.jev_model, source, jev_cache_path if source == "cache" else None,
                    workers=workers)
    cases = Cases(a.data, a.split, a.cases)

    Handler.engine, Handler.cases = engine, cases
    Handler.config = {"run": a.run, "base_model": a.base_model, "jev_model": a.jev_model,
                      "device": device, "gpu": gpu_index, "dtype": a.dtype, "max_state": max_state,
                      "split": a.split, "jev_source": source,
                      "jev_available": source == "cache" or bool(engine._jev_key()),
                      "jev_cached": len(engine.jev_cache),
                      "jev_limit": JEV_LIMIT,
                      "gpus": gpu_map, "workers": workers}

    if a.single_process and a.preload:
        print("loading MedJev ...", flush=True); engine.medjev()
        print("loading base model ...", flush=True); engine.base()

    print(f"MedJev comparison UI  ->  http://{a.host}:{a.port}")
    if a.single_process:
        print(f"  both models on {device} (physical GPU {gpu_index}), {a.dtype}, max_state={max_state}")
    else:
        print(f"  base -> GPU {gpu_map['base']} (port {workers['base']}), "
              f"medjev -> GPU {gpu_map['medjev']} (port {workers['medjev']}), {a.dtype}, max_state={max_state}")
    print(f"  checkpoint {a.run}")
    print(f"  {len(cases.records)} cases from the {a.split} split")
    print(f"  hosted Jev: {source}"
          + (f", live at {JEV_LIMIT} cases only" if source == "live" else f" ({len(engine.jev_cache)} recorded answers)"))
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
