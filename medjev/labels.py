"""Label derivation for MedJev step 1.

Input  : one record of augmented_notes_30K.jsonl
Output : (state, questions) in Kev fine-tuning format.

The `summary` field is GPT-4 structured output over `full_note`; we use it as
weak-supervision gold and normalise its free text into noul / choice / score
labels. `note` and `conversation` are unused.
"""
import json
import re

# --------------------------------------------------------------------------- #
# summary parsing
# --------------------------------------------------------------------------- #

NULLS = {
    "none", "n/a", "na", "", "null", "unknown", "not mentioned",
    "not applicable", "not specified", "not provided", "not reported", "-",
}


def parse_summary(raw):
    """~26% of summaries contain raw newlines inside strings; strict=False fixes
    those, a trailing-comma strip fixes the rest. Returns None if unusable."""
    try:
        return json.loads(raw, strict=False)
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", raw), strict=False)
        except Exception:
            return None


def s_(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def is_null(x):
    return not isinstance(x, str) or x.strip().lower() in NULLS


def entries(summary, key):
    """Non-empty entries of a list-valued template section."""
    lst = summary.get(key) or []
    if isinstance(lst, dict):
        lst = [lst]
    return [e for e in lst
            if isinstance(e, dict) and any(not is_null(v) for v in e.values())]


def section(summary, key):
    d = summary.get(key) or {}
    return d if isinstance(d, dict) else {}


def _join(items, *fields):
    """Join field values, dropping the literal "None" placeholders the GPT-4
    template writes for absent values."""
    vals = [s_(e.get(f)) for e in items for f in fields]
    return " ".join(v.strip() for v in vals if not is_null(v)).strip()


def _first(text, table, default):
    for name, pattern in table:
        if pattern.search(text):
            return name
    return default


def _rx(table):
    return [(n, re.compile(p, re.I)) for n, p in table]


# --------------------------------------------------------------------------- #
# vocabularies (priority-ordered: most specific first)
# --------------------------------------------------------------------------- #

MODALITY = _rx([
    ("histopathology", r"biops|histopath|cytolog|immunohistochem|fine[- ]needle|pathologic(al)? (exam|analysis)|frozen section|specimen"),
    ("genetic_molecular", r"genetic (test|analysis)|sequenc|mutation analysis|karyotyp|\bpcr\b|\bfish\b|genomic|molecular (test|analysis)|chromosom"),
    ("endoscopy", r"endoscop|colonoscop|bronchoscop|cystoscop|gastroscop|arthroscop|hysteroscop"),
    ("imaging", r"\b(ct|c\.t\.|mri|m\.r\.i|x-?ray|ultrasound|ultrasonograph|sonograph|pet(-| )?(ct|scan)?|scintigraph|angiograph|echocardiograph|radiograph|mammograph|doppler|tomograph|imaging|scan)\b"),
    ("functional_testing", r"\b(ecg|ekg|electrocardiogra|eeg|emg|electromyograph|spirometr|pulmonary function|nerve conduction|audiometr|urodynamic|stress test)\b"),
    ("laboratory", r"blood|serum|plasma|\bcbc\b|\bwbc\b|h(a)?emoglobin|creatinin|culture|urinalysis|urine|liver function|electrolyte|\bcrp\b|\besr\b|antibod|titer|level[s]?\b|count|glucose|enzyme|marker"),
])

TREATMENT = _rx([
    ("radiotherapy", r"radiotherap|radiation therap|irradiat|brachytherap|\bgy\b|fractions"),
    ("chemotherapy", r"chemotherap|cisplatin|carboplatin|doxorubicin|cyclophosphamid|methotrexat|paclitaxel|docetaxel|rituximab|fluorouracil|vincristin|etoposid|gemcitabin|imatinib|immunotherap|tyrosine kinase"),
    ("surgery", r"surger|surgical|resect|excis|ectomy|otomy|ostomy|plasty|graft|transplant|amputat|drainage|debrid|fixation|implantat|operat|reconstruct|laparotom|laparoscop"),
    ("antimicrobial", r"antibiotic|antifungal|antiviral|antimicrobial|cillin|ceftriax|cefta|vancomyc|metronidaz|azithromy|acyclovir|fluconaz|doxycyclin|meropenem|levofloxac|rifampi|isoniaz"),
    ("supportive_palliative", r"supportive|palliativ|conservative (management|treatment)|observation|physiotherap|rehabilit|transfusion|oxygen|analgesi|pain (control|management)|hydration|nutrition(al)? support"),
    ("other_pharmacotherapy", r"\bmg\b|tablet|capsule|injection|infusion|steroid|prednis|dexamethason|insulin|anticoagul|heparin|warfarin|statin|therapy|administered|prescrib|treated with|medication"),
])

DRUG_EVIDENCE = re.compile(
    r"\bmg\b|\bmcg\b|\bg/|tablet|capsule|injection|infusion|oral|intravenous|\biv\b"
    r"|steroid|prednis|antibiotic|chemotherap|insulin|daily|twice|dose", re.I)

SEVERE = re.compile(r"severe|marked|intense|extensive|excruciat|profound|critical|high[- ]grade|stage iv|grade (iii|iv|3|4)|debilitat|unbearable", re.I)
MODERATE = re.compile(r"moderate|significant|progressiv|worsening|increasing|persistent|considerable|intermittent|recurrent", re.I)
MILD = re.compile(r"mild|slight|minimal|low[- ]grade|painless|subtle|occasional|transient", re.I)

RESP_BAD = re.compile(r"\b(no |not |without |lack of |failed|fail(s|ed)? to|unsuccessful|ineffective|refractory|did not|minimal|poor|deterior|worsen|progress(ed|ion)|relapse|recurren|died|death|discontinued|intoleran|adverse|side effect|complication)", re.I)
RESP_FULL = re.compile(r"complete|resolv|cured|full recovery|successful|excellent|remission|asymptomatic|uneventful|significant improvement|responded well|good response|marked improvement|no residual", re.I)

SMOKE_NEVER = re.compile(r"non[- ]?smok|never smok|denie[sd]|no (history of |significant )?(smok|tobacco)|not? smok|no tobacco", re.I)
SMOKE_FORMER = re.compile(r"\b(ex|former|quit|ceased|stopped|past|previously)[- ]?(smok|tobacco)?|smoked (until|for \d+ years)", re.I)
SMOKE_CURRENT = re.compile(r"\b(current|active|heavy|chronic|daily)?\s?(smok(er|es|ing)|tobacco (use|abuse|consumption)|pack[- ]year|cigarette)", re.I)


# --------------------------------------------------------------------------- #
# per-variable label functions -> label value or None (question omitted)
# --------------------------------------------------------------------------- #

def lab_hospital_admission(s):
    return bool(entries(s, "admission"))


def lab_surgical_management(s):
    return bool(entries(s, "surgeries"))


def lab_drug_therapy(s):
    return any(DRUG_EVIDENCE.search(_join([e], "name", "dosage", "frequency"))
               for e in entries(s, "treatments"))


def lab_prior_comorbidity(s):
    return not is_null(s_(section(s, "patient medical history").get("physiological context")))


def lab_follow_up_planned(s):
    return not is_null(s_(section(s, "discharge").get("follow up")))


def lab_primary_modality(s):
    tests = _join(entries(s, "diagnosis tests"), "test", "result", "details")
    if not tests:
        return "not_reported"
    return _first(tests, MODALITY, "other")


def lab_principal_treatment(s):
    """Therapy recorded in the treatment plan. Operations recorded only in the
    `surgeries` section are covered by the `surgical_management` question, so
    surgical-only cases land in `none_recorded` here."""
    tx = _join(entries(s, "treatments"), "name", "dosage", "details")
    if not tx:
        return "none_recorded"
    return _first(tx, TREATMENT, "other")


def lab_smoking_status(s):
    raw = s_(section(s, "patient medical history").get("smoking status"))
    if is_null(raw):
        return "not_documented"
    if SMOKE_NEVER.search(raw):
        return "never"
    if SMOKE_FORMER.search(raw):
        return "former"
    if SMOKE_CURRENT.search(raw):
        return "current"
    return None  # unclear wording -> drop the question for this record


def lab_symptom_severity(s):
    text = (_join(entries(s, "symptoms"), "intensity of symptom") + " " +
            _join(entries(s, "diagnosis tests"), "severity")).strip()
    if not text:
        return None
    if SEVERE.search(text):
        return 2
    if MODERATE.search(text):
        return 1
    if MILD.search(text):
        return 0
    return None


def lab_treatment_response(s):
    text = _join(entries(s, "treatments"), "reaction to treatment")
    if not text:
        return None
    if RESP_BAD.search(text):
        return 0
    if RESP_FULL.search(text):
        return 2
    return 1


def lab_workup_intensity(s):
    n = len(entries(s, "diagnosis tests")) + len(entries(s, "medical examinations"))
    return 0 if n <= 2 else (1 if n <= 5 else 2)


# --------------------------------------------------------------------------- #
# question specifications (Kev request shape)
# --------------------------------------------------------------------------- #

QUESTIONS = {
    "hospital_admission": {
        "type": "noul",
        "instructions": "Was this patient admitted to a hospital or other care centre during the episode described?",
        "criteria": {"true": "Admitted as an inpatient or to a care centre",
                     "false": "Outpatient, clinic or emergency visit only, or no admission described"},
        "fn": lab_hospital_admission,
    },
    "surgical_management": {
        "type": "noul",
        "instructions": "Did the patient undergo a surgical or interventional procedure?",
        "criteria": {"true": "One or more operations or invasive procedures were performed",
                     "false": "Managed without any surgical or interventional procedure"},
        "fn": lab_surgical_management,
    },
    "drug_therapy": {
        "type": "noul",
        "instructions": "Was the patient treated with a named drug or a dosed medication?",
        "criteria": {"true": "At least one medication with a name, dose or schedule was given",
                     "false": "No pharmacological treatment described"},
        "fn": lab_drug_therapy,
    },
    "prior_comorbidity": {
        "type": "noul",
        "instructions": "Does the note describe relevant pre-existing medical history or comorbidity, separate from the current presenting problem?",
        "criteria": {"true": "Prior physiological conditions or comorbidities documented",
                     "false": "No prior medical history documented"},
        "fn": lab_prior_comorbidity,
    },
    "follow_up_planned": {
        "type": "noul",
        "instructions": "Does the note describe a planned follow-up, review or surveillance after discharge?",
        "criteria": {"true": "A follow-up appointment, review or surveillance plan is described",
                     "false": "No follow-up plan described"},
        "fn": lab_follow_up_planned,
    },
    "primary_diagnostic_modality": {
        "type": "choice",
        "instructions": "Which type of investigation was most central to reaching the diagnosis?",
        "criteria": {
            "histopathology": "Biopsy, cytology, immunohistochemistry or specimen pathology",
            "genetic_molecular": "Genetic, molecular, sequencing or cytogenetic testing",
            "endoscopy": "Endoscopic visualisation (colonoscopy, bronchoscopy, cystoscopy, arthroscopy, …)",
            "imaging": "Radiological imaging (CT, MRI, X-ray, ultrasound, PET, angiography, …)",
            "functional_testing": "Physiological or functional tests (ECG, EEG, EMG, spirometry, audiometry, …)",
            "laboratory": "Blood, serum, urine or other laboratory analyses",
            "other": "A diagnostic test that fits none of the above categories",
            "not_reported": "No diagnostic test is described",
        },
        "fn": lab_primary_modality,
    },
    "principal_medical_therapy": {
        "type": "choice",
        "instructions": "Which modality best describes the therapy given to this patient in the treatment plan?",
        "criteria": {
            "radiotherapy": "Radiotherapy, irradiation or brachytherapy",
            "chemotherapy": "Cytotoxic chemotherapy, targeted therapy or immunotherapy",
            "surgery": "An operation or interventional procedure",
            "antimicrobial": "Antibiotic, antifungal, antiviral or other antimicrobial therapy",
            "supportive_palliative": "Supportive, conservative, palliative or rehabilitative care",
            "other_pharmacotherapy": "Any other drug treatment",
            "other": "A treatment that fits none of the above categories",
            "none_recorded": "No therapy is recorded in the treatment plan",
        },
        "fn": lab_principal_treatment,
    },
    "smoking_status": {
        "type": "choice",
        "instructions": "What is the patient's smoking status as documented in the note?",
        "criteria": {
            "never": "Never smoked, or smoking explicitly denied",
            "former": "Smoked in the past but has stopped",
            "current": "Currently smokes or uses tobacco",
            "not_documented": "Smoking status is not mentioned at all",
        },
        "fn": lab_smoking_status,
    },
    "symptom_severity": {
        "type": "score",
        "instructions": "How severe is the patient's presentation overall, judged from the documented symptoms and findings?",
        "criteria": ["Mild — slight, minimal, low-grade or painless findings",
                     "Moderate — significant, persistent or progressive findings",
                     "Severe — severe, marked, extensive, high-grade or critical findings"],
        "fn": lab_symptom_severity,
    },
    "treatment_response": {
        "type": "score",
        "instructions": "How well did the patient respond to the treatment given?",
        "criteria": ["No response or deterioration — treatment failed, was ineffective, or the condition worsened",
                     "Partial response — some improvement, or an ongoing or mixed response",
                     "Complete response — symptoms resolved, disease in remission, or recovery uneventful"],
        "fn": lab_treatment_response,
    },
    "diagnostic_workup_intensity": {
        "type": "score",
        "instructions": "How extensive was the diagnostic work-up (examinations and tests) for this patient?",
        "criteria": ["Limited — at most two examinations or tests",
                     "Moderate — three to five examinations or tests",
                     "Extensive — six or more examinations or tests"],
        "fn": lab_workup_intensity,
    },
}


def build_questions(summary, keep_undocumented_smoking=False):
    """Return the Kev `questions` object for one record, omitting any question
    whose label cannot be determined."""
    out = {}
    for qid, spec in QUESTIONS.items():
        label = spec["fn"](summary)
        if label is None:
            continue
        if (qid == "smoking_status" and label == "not_documented"
                and not keep_undocumented_smoking):
            continue
        q = {"type": spec["type"], "instructions": spec["instructions"]}
        if spec.get("criteria") is not None:
            q["criteria"] = spec["criteria"]
        q["label"] = label
        out[qid] = q
    return out
