"""Request shapes and their conversion into the model's internal record.

Derived from `kev/kev/api.py` (Jared Palmer, Apache-2.0) — see NOTICE. MedJev
vendors it so the package does not depend on the `kev/` checkout. Trimmed to what
MedJev uses: the TypeSafe System One request shape, and `to_record`, which turns a
request into the `{state, questions: [{instr, options, label}]}` form `encode()`
consumes.

    noul   -> 2 options, [no, yes];             answer = p(yes)
    choice -> one option per named criterion;    answer = argmax over the names
    score  -> one option per ordered level;      answer = the level index

Training examples and inference requests both pass through here, so the text the
model is trained on is byte-identical to the text it is served.
"""
import json
from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class Request(BaseModel):
    state: JSONContent
    model: str = "medjev-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees, keeping field
    names as labels. MedJev states are plain strings (`full_note`), but the
    question criteria go through here too."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list))
                     else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def to_record(req: Request):
    """-> (internal record for `encode`, per-question metadata to map probabilities
    back to option names)."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        instr = render(q.instructions)
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            meta.append({"id": qid, "type": "noul"})
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
            meta.append({"id": qid, "type": "choice", "keys": list(q.criteria.keys())})
        else:
            opts = [render(x) for x in q.criteria]
            meta.append({"id": qid, "type": "score",
                         "legend": {str(i): render(x) for i, x in enumerate(q.criteria)}})
        qs.append({"instr": instr, "options": opts, "label": 0})
    return {"state": render(req.state), "questions": qs}, meta


def choice_confidence(p) -> float:
    """(p_max - 1/K) / (1 - 1/K); a single option has confidence 1."""
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p) -> float:
    """1 - E|level - mode| / (L - 1): how concentrated the distribution is around
    its modal level. An approximation of TypeSafe's unpublished formula."""
    levels = len(p)
    mode = max(range(levels), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (levels - 1)


def to_answers(probs, meta) -> dict:
    """Probabilities -> the System One answer shape, for serving and for the
    comparison UI. Neither confidence field is a measured accuracy rate."""
    out = {}
    for p, m in zip(probs, meta):
        p = [float(x) for x in p]
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": p[1]}
        elif m["type"] == "choice":
            out[m["id"]] = {"type": "choice",
                            "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                            "confidence": choice_confidence(p),
                            "probabilities": dict(zip(m["keys"], p))}
        else:
            out[m["id"]] = {"type": "score", "score": sum(i * pi for i, pi in enumerate(p)),
                            "legend": m["legend"], "confidence": score_confidence(p),
                            "probabilities": {str(i): v for i, v in enumerate(p)}}
    return out


def output_tokens(tok, answers: dict) -> int:
    """Billing-style figure: tokens of the serialised answers. There is no
    generation, so this is not a count of generated tokens."""
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
