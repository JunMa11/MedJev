# `data/results/` — inference output of the reference systems

The answers the two external reference systems gave on the MedJev splits: the hosted **Jev** API and
the zero-shot **Qwen3.5-0.8B** probes. They sit under `data/` rather than `runs/` because they are
fixed evaluation *inputs* — nothing here is an artifact of a MedJev training run, and the Jev answers
were paid for once and are replayed thereafter.

MedJev's own checkpoints, its eval reports and the rule baseline stay in
[`runs/`](../../runs/README.md).

Commands below assume the virtualenv is active (`source .venv/bin/activate`).

## Hosted Jev (`jev-1.13.0`, TypeSafe System One)

| Path | Size | What it is |
|---|---|---|
| `jev-test.jsonl` | 4.7 M | **Paid API answers**, one line per record: every question's probability distribution, per-request latency and token usage. 2,895 records, 26,286 questions, 0 failures. |
| `jev-development.jsonl` | 4.9 M | The same for the development split — what the prior correction is fitted on. |
| `jev-{test,development}-runinfo.json` | — | Wall time, throughput, latency percentiles and token totals for those two runs. |
| `jev-test-report.json` | — | `medjev.eval_jev`: Jev scored against the gold labels (micro 0.6016). |
| `jev-test-calibrated.json` | — | `medjev.calibrate_jev`: the same answers after a per-question temperature + class prior fitted on development (0.6016 → 0.6913). |

**Do not delete the two `.jsonl` files.** Regenerating them costs money and network, and
`medjev.serve_compare` replays them rather than re-calling the API.

```bash
python -m medjev.run_jev  --split test --allow-test --workers 6   # needs TYPESAFE_API_KEY in .env
python -m medjev.eval_jev --split test
python -m medjev.calibrate_jev
```

### Calibration for Jev

`jev-test-calibrated.json` the same paid answers in `jev-test.jsonl`, rescored after two numbers are fitted per question on *development* set. Jev is never re-called and never retrained; only the probability vector it already
returned is reshaped:

    argmax over options of  (p ** (1/T)) * w,  renormalised

| fitted per question | what it is | grid |
|---|---|---|
| `T` — temperature | sharpens (`T<1`) or flattens (`T>1`) Jev's distribution | 0.5 … 3.0 step 0.1 |
| `w` — class-prior weight | `(train label frequency / Jev's own mean predicted frequency) ** mix`, so systematically over-predicted options are damped | `mix` ∈ 0, 0.25, 0.5, 0.75, 1.0 |

Both are chosen by the coordinate search in `medjev.calibrate_jev.fit`, maximising **development**
accuracy. `temperature` and `dev_accuracy` in the report record what was selected and what it scored
where it was fitted; every other field is test.


| variable | Jev predicts the top level | labels have it |
|---|---|---|
| `diagnostic_workup_intensity` | 81.1% | 23.0% |
| `treatment_response` | 55.9% | 10.5% |

`diagnostic_workup_intensity` bins by *counting entries in the structured summary*; Jev reads a dense
case report and calls almost every work-up extensive. That is a threshold disagreement, not a reading
failure, and a prior is exactly the right instrument for it. Correcting it moves Jev 0.6016 → 0.6913
micro without touching the model — so **that 0.09 is the part of Jev's deficit that was never about
clinical comprehension**, and the remaining gap to MedJev is what fine-tuning actually bought.

It also keeps the comparison honest: the rule baseline in `runs/` is itself development-tuned, so
scoring raw zero-shot Jev against it would be the weaker system handicapped.

### Per-question effect (test)

| variable | type | T | raw | calibrated | Δ |
|---|---|---|---|---|---|
| `follow_up_planned` | noul | 2.4 | 0.6314 | 0.8891 | **+0.258** |
| `diagnostic_workup_intensity` | score | 3.0 | 0.3551 | 0.6076 | **+0.253** |
| `treatment_response` | score | 2.6 | 0.2865 | 0.4397 | **+0.153** |
| `prior_comorbidity` | noul | 2.5 | 0.7368 | 0.8577 | +0.121 |
| `drug_therapy` | noul | 3.0 | 0.6964 | 0.7351 | +0.039 |
| `symptom_severity` | score | 2.5 | 0.5939 | 0.6242 | +0.030 |
| `surgical_management` | noul | 2.8 | 0.8097 | 0.8387 | +0.029 |
| `principal_medical_therapy` | choice | 1.7 | 0.3655 | 0.3896 | +0.024 |
| `smoking_status` | choice | 1.5 | 0.9435 | 0.9504 | +0.007 |
| `hospital_admission` | noul | 2.9 | 0.7986 | 0.8024 | +0.004 |
| `primary_diagnostic_modality` | choice | 1.5 | 0.4898 | 0.4905 | +0.001 |



- **Every gain is positive**, and the three largest are the convention-bound variables. Nothing is
  lost, because the identity transform (`T=1`, `mix=0`) is inside the search grid.
- **The `choice` variables barely move** (+0.001 to +0.024). Picking among 8 diagnostic modalities is
  a genuine reading problem; no reweighting fixes it. `smoking_status` is already at 0.94 — it is
  lexically explicit, so there was nothing to correct.
- **Fitted `T` is high (2.4–3.0) on the noul and score questions**: Jev is badly overconfident there,
  and flattening lets the prior actually change the argmax.
- **Accuracy improves while `calibrated_brier` stays poor** on `choice` (0.85, 1.02). The transform
  is fitted for argmax accuracy, not for probability quality, so these confidences should not be used
  as calibrated probabilities — MedJev's own eval reports are where to look for that.


## Qwen3.5-0.8B zero-shot probes

`medjev.base_probe` answers the same questions through next-token letter logits — the floor the
fine-tune has to beat. Each directory holds `report.json` (the summary) and `rows.json` (per-question
raw output, which is what lets the metrics be re-cut without spending GPU time again).

| Path | Size | Model and prompt | Test micro |
|---|---|---|---|
| `base-qwen3.5-0.8b-test/` | 6.7 M | `Qwen3.5-0.8B-Base`, plain prompt | 0.5002 |
| `instruct-qwen3.5-0.8b-test-chat/` | 6.8 M | `Qwen3.5-0.8B` instruct, chat template | 0.5172 |


```bash
python -m medjev.base_probe --model Qwen3.5-0.8B --prompt chat \
    --split test --allow-test --out data/results/instruct-qwen3.5-0.8b-test-chat
```
