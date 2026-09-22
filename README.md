# MedJev: Ultra-fast Clinical Variables Extraction from Free-text Notes

 MedJev turns free-text clinical notes into table-ready data. It is small enough to run on one consumer GPU inside your own hospital: **thousands of notes an hour**, no subscription or per-use fees, and no patient text ever leaves your network.

https://github.com/user-attachments/assets/5e242713-8771-47c4-ab49-8bb5ba27ad41

| Task Type | Answer | Example |
|---|---|---|
| `noul` | yes/no, returned as a probability | `hospital_admission` — was the patient admitted? |
| `choice` | one of 2–255 named options | `primary_diagnostic_modality` — imaging, histopathology, laboratory, … |
| `score` | an ordered level | `symptom_severity` — mild / moderate / severe |

### How MedJev differs from general LLMs (e.g., ChatGPT et al.)

| | MedJev | A general LLM |
|---|---|---|
| **What you get back** | Every field filled in with one of the answers you defined, ready to go straight into a spreadsheet or database. | A written reply that somebody still has to read and transcribe into fields. |
| **Can it go off-script?** | No. It can only pick from the answers you listed. | Sometimes. It may invent a category, hedge ("moderate to severe"), or word the same answer differently from one note to the next. |
| **Does it tell you when it is unsure?** | Yes. Every answer comes with a confidence, so you can accept the clear-cut charts automatically and send only the doubtful ones for human review. | It tends to sound equally confident whether it is right or wrong. |
| **Speed and running cost** | Thousands of notes in minutes on a single computer, with no per-use fee. | One request per note, slower, and more expensive. |
| **Where the notes go** | Nowhere. It runs on your own hardware, so patient text never leaves your network. | Usually sent to an outside company's servers. |
| **Adding a new field** | Needs a few hundred example charts labelled first, then a few hours of retraining. | Just describe the new field in a sentence. |
| **Asking anything else** | It only answers the fields it was built for. | Summarise, explain, draft, and answer open-ended questions. |

---

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate

pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

Check:

```bash
python -c "import torch, transformers, peft, fla; print(torch.__version__, torch.cuda.is_available())"
nvcc -V
```


**Optional: hosted Jev baseline from [TypeSafe AI](https://typesafe.ai/)**. (Everything else runs fully offline.)

```bash
echo 'TYPESAFE_API_KEY=apikey_...' > .env && chmod 600 .env
```

---

## Training

The dataset is ready for training at `data/medjev-v1/` (23,719 / 2,997 / 2,895 records from [Augmented Clinical Notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes))

```bash
# download Qwen3.5-0.8B-Base
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B-Base

# single GPU (>=16G)
python -m medjev.train --out runs/medjev-0.8b \
    --epochs 2 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 \
    --questions_per_record 4 --save_every single

# multi GPUs, the current recipe (~2,048-token states, all ~9 questions per record)
scripts/train_3gpu.sh runs/medjev-0.8b
```


| Flag | Default | Why you would change it |
|---|---|---|
| `--questions_per_record N` | `0` (all) | Trains a random N of a record's ~9 questions per epoch. Cost is `questions × state tokens`, so this scales runtime almost linearly — the main throughput knob. |
| `--max_state` / `--max_branch` | `2048` / `+512` | The state budget. Notes are median ~670 tokens, p95 ~1,337. `max_branch` is the state **+** branch total, not the branch alone. |
| `--dtype bf16 --checkpointing 1` | `bf16`, on | Halves memory |
| `--ord_w` | `0.3` | Ordinal RPS loss weight for the three genuinely ordered `score` variables. |
| `--class_weight` | `none` | Inverse-frequency weighting, for the skewed binaries. |
| `--val_every` / `--val_records` | `250` / `400` | In-training validation on development. |
| `--wandb 1` | off | Weights & Biases logging (`--wandb_project`, `--wandb_name`). |

Select checkpoints on **development**. The test split is locked behind `--allow-test` in every
script; keep it that way.

---

## Evaluation

```bash
# score a checkpoint (development by default)
python -m medjev.evaluate --run runs/medjev-0.8b --split development

# a specific intermediate checkpoint, in the serving dtype
python -m medjev.evaluate --run runs/medjev-0.8b/step-5000 \
    --split development --dtype bf16

# the locked test split, once, after selection is settled
python -m medjev.evaluate --run runs/medjev-0.8b --split test --allow-test

# tabulate every system on one split
python -m medjev.compare --split test
```

`medjev.evaluate` writes a JSON report with micro accuracy, per-question accuracy, macro-F1, Brier,
ECE, mean absolute level error for `score` questions, the majority-class floor each question has to
clear, and `latency_ms_per_record`. It deliberately runs the **serving** path — the state is encoded
once and each question branch reads its cache — so the latency it reports is the real per-record cost
of answering all of a record's variables.

`medjev.compare` normalises the differently-shaped reports from `eval_baseline`, `base_probe`,
`eval_jev` and `evaluate` into one markdown table; point it at specific files with `--rule`,
`--probe`, `--instruct`, `--jev` and `--medjev`.

Per-category serving latency:

```bash
python -m medjev.bench_runtime --records 300 --jev-records 200 --workers 1
```

---

## Inference


### From Python

```python
import torch
from medjev.checkpoint import load
from medjev.model import MAX_BRANCH, MAX_STATE
from medjev.records import materialize

tok, model = load("runs/medjev-0.8b/step-5000", "cuda", dtype=torch.bfloat16)
model.lm.config.use_cache = True

request = {
    "state": "A 36-year old female patient was admitted with severe left hip pain. MRI showed a "
             "lesion; biopsy confirmed osteosarcoma. She underwent resection and received "
             "chemotherapy, with complete resolution at 6 months.",
    "questions": {
        # `label` and `src` are required by materialize(); the label is not read at inference
        "hospital_admission": {
            "type": "noul",
            "instructions": "Was this patient admitted to a hospital or other care centre?",
            "criteria": {"true": "Admitted as an inpatient", "false": "Outpatient visit only"},
            "label": True, "src": "demo",
        },
        "symptom_severity": {
            "type": "score",
            "instructions": "How severe is the patient's presentation overall?",
            "criteria": ["Mild", "Moderate", "Severe"],
            "label": 0, "src": "demo",
        },
    },
}

enc = model.encode(tok, materialize(request), max_state=MAX_STATE, max_branch=MAX_BRANCH)
with torch.no_grad():
    probs, _ = model.probs_and_prefix(enc)   # state encoded once, branches read its cache

for p, (qid, q) in zip(probs, request["questions"].items()):
    print(qid, [round(float(x), 3) for x in p])

# hospital_admission [0.0, 1.0]        noul:   [P(false), P(true)]
# symptom_severity   [0.0, 0.0, 1.0]   score:  one entry per level (choice: per option)
```

The question schema is free-form: reuse the 11 specs in `medjev.labels.QUESTIONS`, or pass your own
instructions and criteria in the same shape. Option order is shuffled during training, so `choice`
answers are order-robust, but the model is fine-tuned on this corpus's vocabulary — new option sets
work best after a short further fine-tune.

### Interactive comparison UI

```bash
python -m medjev.serve_compare --run runs/medjev-0.8b/step-5000 --preload
```

Serves <http://127.0.0.1:8765>, a three-way comparison of base Qwen3.5-0.8B (zero-shot letter
logits), hosted Jev and the MedJev checkpoint. Batch-score 50, 100, 1,000 or all cases with live
per-system progress, then drill into any single case to see each system's answers, confidences and
latency beside the gold label. Jev answers are replayed from `data/results/jev-<split>.jsonl` rather than
re-bought from the API. stdlib HTTP only, no extra dependencies; models load lazily, every GPU call
is serialised behind one lock, and the GPU is masked for you. `--host`, `--port`, `--dtype` and
`--split` are configurable.

---

## Citation


```bibtex
@software{ma_medjev_2026,
  author  = {Ma, Jun},
  title   = {{MedJev: Ultra-fast Clinical Variable Extraction from Free-text Notes}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/JunMa11/MedJev}
}
```

Please also cite the work MedJev is built on: the
[kev](https://github.com/jaredpalmer/kev) decision-model architecture, the
[Augmented Clinical Notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes)
corpus, and [PMC-Patients](https://arxiv.org/abs/2202.13876), the source of the underlying notes.

---

## Acknowledgements

- **[kev](https://github.com/jaredpalmer/kev)** by Jared Palmer — the decision-model architecture
  (LoRA + pointer head, question isolation, the System One request format) and the training and
  serving code MedJev is built from. Apache-2.0. `medjev/model.py`, `api.py`, `records.py` and
  `checkpoint.py` are adapted from it; `NOTICE` records exactly which files and what changed.
- **[Augmented Clinical Notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes)**
  by Antoine Bonnet and Paul Boulenger (EPFL). Its structured patient summaries, from
  which MedJev's labels are derived, were generated with GPT-4 against a medical template designed
  with Prof. Mary-Anne Hartley. MIT licence.
- **[PMC-Patients](https://arxiv.org/abs/2202.13876)** — the underlying clinical notes, extracted
  from open-access PubMed Central case reports.
- **[Qwen3.5](https://huggingface.co/Qwen)** by Alibaba — the base models. Apache-2.0.
- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** — the Gated
  DeltaNet kernels that make this trainable in hours rather than days.

