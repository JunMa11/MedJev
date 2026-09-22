"""CPU-only checks for the parts of MedJev that do not need the model.

    .venv/bin/python -m pytest tests/test_medjev.py -q

Anything touching the backbone needs a GPU (flash-linear-attention's DeltaNet
kernels are Triton-only), so the model path is covered by a smoke train/evaluate
run instead, not here.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medjev.build_dataset import jsonl_line, split_of  # noqa: E402
from medjev.labels import QUESTIONS, build_questions, parse_summary  # noqa: E402
from medjev.train import class_weights, label_key, subsample_questions  # noqa: E402

DATA = ROOT / "data" / "medjev-v1"


# --------------------------------------------------------------------------- #
# dataset integrity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("split", ["train", "development", "test"])
def test_one_record_per_line(split):
    """splitlines() must agree with split('\\n'): a raw U+2028 inside a note would
    silently cut a record in half for any reader that uses the former (this bit us
    once, and load_records now splits on "\\n" only)."""
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not built")
    text = path.read_text()
    assert len(text.splitlines()) == text.count("\n")
    for line in text.split("\n"):
        if line.strip():
            json.loads(line)


def test_jsonl_line_survives_exotic_separators():
    line = jsonl_line({"state": "a b c\x85d", "n": 1})
    assert len(line.splitlines()) == 1
    assert json.loads(line)["state"] == "a b c\x85d"


def test_splits_are_deterministic_and_disjoint():
    assert split_of("133948") == split_of("133948")
    seen = {split_of(str(i)) for i in range(2000)}
    assert seen == {"train", "development", "test"}


@pytest.mark.parametrize("split", ["development", "test"])
def test_labels_are_in_range(split):
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not built")
    with path.open() as f:
        for n, line in enumerate(f):
            if n >= 200:
                break
            rec = json.loads(line)
            assert isinstance(rec["state"], str) and rec["state"]
            for qid, q in rec["questions"].items():
                assert q["type"] == QUESTIONS[qid]["type"]
                if q["type"] == "noul":
                    assert isinstance(q["label"], bool)
                elif q["type"] == "choice":
                    assert q["label"] in q["criteria"]
                else:
                    assert 0 <= q["label"] < len(q["criteria"])


# --------------------------------------------------------------------------- #
# label derivation
# --------------------------------------------------------------------------- #

def test_parse_summary_recovers_raw_newlines():
    assert parse_summary('{"a": "line1\nline2"}') == {"a": "line1\nline2"}
    assert parse_summary('{"a": 1,}') == {"a": 1}
    assert parse_summary("not json at all") is None


def test_build_questions_omits_undeterminable():
    """A question with no derivable label is dropped, not defaulted."""
    qs = build_questions({})
    assert set(qs) <= set(QUESTIONS)
    assert all("label" in q for q in qs.values())


def test_smoking_undocumented_is_gated():
    summary = {"patient medical history": {"smoking status": "None"}}
    without = build_questions(summary, keep_undocumented_smoking=False)
    with_ = build_questions(summary, keep_undocumented_smoking=True)
    assert "smoking_status" not in without
    assert with_.get("smoking_status", {}).get("label") == "not_documented"


# --------------------------------------------------------------------------- #
# training helpers
# --------------------------------------------------------------------------- #

def test_subsample_keeps_k_questions():
    import random
    req = {"questions": {f"q{i}": {"label": i} for i in range(9)}}
    rng = random.Random(0)
    assert len(subsample_questions(req, rng, 4)["questions"]) == 4
    assert subsample_questions(req, rng, 0)["questions"] == req["questions"]
    assert subsample_questions(req, rng, 99)["questions"] == req["questions"]


def test_subsample_is_deterministic_per_seed():
    import random
    req = {"questions": {f"q{i}": {"label": i} for i in range(9)}}
    a = sorted(subsample_questions(req, random.Random(7), 4)["questions"])
    b = sorted(subsample_questions(req, random.Random(7), 4)["questions"])
    assert a == b


def test_class_weights_are_inverse_frequency_and_mean_one():
    reqs = [{"questions": {"q": {"label": True}}}] * 90 + [{"questions": {"q": {"label": False}}}] * 10
    w = class_weights(reqs)["q"]
    assert w["False"] > w["True"]
    assert abs(0.9 * w["True"] + 0.1 * w["False"] - 1.0) < 1e-9


def test_label_key_matches_class_weights_keys():
    """class_weights keys off the raw request label; label_key must reproduce it
    from the materialized question, or weighting would KeyError mid-run."""
    assert label_key({"qtype": "noul", "label": 1, "keys": ["false", "true"]}) == "True"
    assert label_key({"qtype": "noul", "label": 0, "keys": ["false", "true"]}) == "False"
    assert label_key({"qtype": "choice", "label": 2, "keys": ["a", "b", "imaging"]}) == "imaging"
    assert label_key({"qtype": "score", "label": 1, "keys": ["0", "1", "2"]}) == "1"


# --------------------------------------------------------------------------- #
# end to end, no model
# --------------------------------------------------------------------------- #

def test_records_round_trip_through_kev():
    """A record must survive load_records + materialize with the labels landing on
    the right option index — the contract the trainer and scorer both rely on."""
    path = DATA / "development.jsonl"
    if not path.exists():
        pytest.skip("dataset not built")
    from medjev.records import load_records, materialize
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(path.read_text().split("\n")[:20]))
        tmp = f.name
    reqs = load_records(tmp, source="medjev")
    assert len(reqs) == 20
    for req in reqs:
        rec = materialize(req)
        assert len(rec["questions"]) == len(req["questions"])
        for q, (qid, src) in zip(rec["questions"], req["questions"].items()):
            assert q["qid"] == qid
            assert 0 <= q["label"] < len(q["options"])
            assert label_key(q) == (str(src["label"]) if src["type"] != "noul" else str(bool(src["label"])))


def test_compare_runs_on_existing_reports():
    if not (ROOT / "runs" / "baseline-test.json").exists():
        pytest.skip("no baseline report")
    out = subprocess.run([sys.executable, "-m", "medjev.compare", "--split", "test"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "micro accuracy" in out.stdout


# --------------------------------------------------------------------------- #
# self-containedness
# --------------------------------------------------------------------------- #

MODULES = ["api", "model", "records", "checkpoint", "train", "evaluate", "compare", "labels",
           "build_dataset", "baseline", "eval_baseline", "base_probe", "serve_compare",
           "bench_runtime", "run_jev", "eval_jev", "calibrate_jev", "merge_shards", "figures"]


def test_package_does_not_import_kev():
    """medjev must stand alone: the kev/ checkout beside it is a reference, not a
    dependency. Runs in a subprocess with an import hook that raises on `kev`, so a
    re-introduced import fails here rather than on a machine without kev/ present."""
    script = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'kev' or name.startswith('kev.'):\n"
        "            raise ImportError('medjev must not import ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import importlib\n"
        f"for m in {MODULES!r}:\n"
        "    importlib.import_module('medjev.' + m)\n"
        "assert not [k for k in sys.modules if k == 'kev' or k.startswith('kev.')]\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_no_kev_imports_in_source():
    """A grep-level guard, so the failure names the offending line."""
    offenders = []
    for path in sorted((ROOT / "medjev").glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import kev", "from kev")) or "medjev.kevlib" in stripped:
                offenders.append(f"{path.name}:{n}: {stripped}")
    assert not offenders, "\n".join(offenders)
