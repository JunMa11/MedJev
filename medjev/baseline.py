"""Rule-based baseline for MedJev clinical variable extraction.

Predicts every variable in `medjev.labels.QUESTIONS` from `full_note` text
only, using hand-written regular expressions and counting heuristics. This is
the floor a fine-tuned Kev model has to beat.

    uv run python -m medjev.baseline --split development
    uv run python -m medjev.baseline --split test --allow-test
"""
import re

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def rx(p):
    return re.compile(p, re.I)


def sentences(note):
    return re.split(r"(?<=[.!?])\s+|\n+", note)


def near(note, trigger, window=200):
    """Text windows around each trigger match — lets rules read local context
    instead of the whole note."""
    return [note[max(0, m.start() - window): m.end() + window]
            for m in trigger.finditer(note)]


NEG = rx(r"\b(no|not|non|never|without|denie[sd]|denying|negative for|free of|absence of|ruled out|unremarkable)\b")


# --------------------------------------------------------------------------- #
# noul rules
# --------------------------------------------------------------------------- #

ADMIT = rx(r"\b(was |were |been )?(admitted|hospitali[sz]ed)\b|\badmission\b|\binpatient\b"
           r"|\bhospital course\b|\b(intensive care unit|\bicu\b)\b|\btransferred to (our|the) (hospital|ward|unit|department)\b"
           r"|\bpostoperative day\b|\b(on|during) (the )?(first|second|third|\d+)(st|nd|rd|th)? (hospital|postoperative) day\b"
           r"|\bdischarged\b|\bward\b")
ADMIT_NEG = rx(r"\boutpatient (department|clinic|basis|setting)\b|\bas an outpatient\b|\bday[- ]case\b")

SURGERY = rx(r"\b\w*(ectomy|ostomy|otomy|plasty|pexy|desis|rraphy)\b"
             r"|\b(surger(y|ies|ical)|operat(ion|ive|ed)|resect(ed|ion)|excis(ed|ion)|amputat|graft(ing|ed)?"
             r"|transplant(ation|ed)?|implant(ation|ed)|debride|drainage|fixation|reconstruct|anastomos"
             r"|laparotom|laparoscop|thoracotom|craniotom|arthroscop|intraoperative|postoperative|perioperative)")

DOSE = rx(r"\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|iu|units?)\b(/|\s?per\s?)?(kg|m2|day|d|h|hr)?"
          r"|\b(mg|g)/(kg|m2|day|dl)\b")
DRUGNAME = rx(r"\b\w+(cillin|mycin|micin|azole|oxacin|cycline|prazole|olol|pril|sartan|statin|ipine"
              r"|tinib|mab|ximab|zumab|umab|dipine|floxacin|cephin|penem)\b"
              r"|\b(prednis(one|olone)|dexamethasone|methotrexate|cisplatin|carboplatin|cyclophosphamide"
              r"|doxorubicin|paclitaxel|docetaxel|vincristine|etoposide|gemcitabine|fluorouracil|5-fu"
              r"|heparin|warfarin|aspirin|insulin|morphine|ibuprofen|paracetamol|acetaminophen"
              r"|metformin|furosemide|amoxicillin|ceftriaxone|vancomycin|metronidazole|acyclovir"
              r"|fluconazole|rituximab|imatinib|tacrolimus|cyclosporine|azathioprine|mycophenolate)\b")
ADMINISTER = rx(r"\b(was |were )?(administered|prescribed|started on|initiated on|commenced on|treated with"
                r"|given|received|put on|placed on)\b.{0,60}\b(therapy|treatment|drug|medication|tablet|capsule"
                r"|injection|infusion|antibiotic|chemotherapy|steroid|dose)")

HISTORY = rx(r"\b(past|previous|prior) (medical |surgical |clinical )?history\b"
             r"|\bhistory of\b|\bknown case of\b|\bknown (diabetic|hypertensive|asthmatic|epileptic)\b"
             r"|\bhad been diagnosed\b|\bpreviously diagnosed\b|\bcomorbid\b|\bco-?morbidit"
             r"|\b(suffered|suffering) from\b|\bhas a \d+[- ]year history\b|\bwas diagnosed with .{0,40}\b(in|at the age of) \d{4}\b"
             r"|\bunderlying (disease|condition|illness)\b|\bmedical background\b")
HISTORY_NEG = rx(r"\b(no|unremarkable|negative|denied any)\s+(significant\s+|relevant\s+|contributory\s+)?"
                 r"(past |previous |prior )?(medical |surgical )?history\b"
                 r"|\bpast medical history was (unremarkable|non-contributory|negative)\b")

FOLLOWUP = rx(r"\bfollow(ed|ing)?[- ]?up\b|\bfollowed up\b|\bat (the )?\d+[- ]?(month|year|week)s? (of )?follow"
              r"|\bsurveillance\b|\bregular (check|review|monitoring|visits)\b|\breviewed (in|at|after)\b"
              r"|\bscheduled (for|to) (a )?(review|visit|appointment)\b|\bre-?evaluat(ed|ion) (at|after|in)\b"
              r"|\bremains? (asymptomatic|well|disease[- ]free) (at|after)\b|\bno (evidence of )?recurrence (at|after)\b")


# --------------------------------------------------------------------------- #
# choice rules (priority-ordered, first match wins)
# --------------------------------------------------------------------------- #

MODALITY = [
    ("histopathology", rx(r"\bbiops(y|ies|ied)\b|histopatholog|histolog|cytolog|immunohistochem|\bihc\b"
                          r"|fine[- ]needle aspirat|\bfnac?\b|pathologic(al)? examination|frozen section"
                          r"|surgical specimen|microscopic examination")),
    ("genetic_molecular", rx(r"\bgenetic (test|analysis|screening|studies)\b|\bsequenc(ing|ed)\b|mutation analysis"
                             r"|karyotyp|\bpcr\b|\bfish\b analysis|molecular (test|analysis|studies)|genomic"
                             r"|chromosomal analysis|whole[- ]exome|next[- ]generation sequencing")),
    ("endoscopy", rx(r"\bendoscop|colonoscop|bronchoscop|cystoscop|gastroscop|arthroscop|hysteroscop|laryngoscop"
                     r"|sigmoidoscop|\bercp\b")),
    ("imaging", rx(r"\b(ct|c\.t\.|mri|m\.r\.i\.?|x-?ray|ultrasound|ultrasonograph|sonograph|pet[- /]?(ct|scan)?"
                   r"|scintigraph|angiograph|echocardiograph|radiograph|mammograph|doppler|tomograph|spect"
                   r"|imaging|\bmr imaging\b)\b|\bscan (showed|revealed|demonstrated|was performed)\b")),
    ("functional_testing", rx(r"\b(ecg|ekg|electrocardiogra|eeg|electroencephalogra|emg|electromyograph|spirometr"
                              r"|pulmonary function test|nerve conduction|audiometr|urodynamic|exercise stress test"
                              r"|visual field test|electroretinogra)\b")),
    ("laboratory", rx(r"\b(blood (test|count|culture|sample|work)|serum|plasma|cbc|wbc|h(a)?emoglobin|creatinine"
                      r"|urinalysis|urine (test|culture|analysis)|liver function test|renal function test"
                      r"|electrolyte|crp|esr|c-reactive protein|antibod(y|ies)|titer|titre|tumou?r marker"
                      r"|laborator(y|ies)|blood gas|platelet count|leukocyt)\b")),
]

THERAPY = [
    ("radiotherapy", rx(r"\bradiotherap|radiation therap|irradiat|brachytherap|\b\d+\s?gy\b|\bgy in \d+ fractions"
                        r"|external beam|radiosurger|gamma knife|cyberknife")),
    ("chemotherapy", rx(r"\bchemotherap|chemoradi|cisplatin|carboplatin|doxorubicin|cyclophosphamide|methotrexate"
                        r"|paclitaxel|docetaxel|rituximab|fluorouracil|5-fu\b|vincristine|etoposide|gemcitabine"
                        r"|imatinib|immunotherap|tyrosine kinase inhibitor|\bchop\b|\bfolfox\b|targeted therapy"
                        r"|\w+(tinib|mab)\b")),
    ("antimicrobial", rx(r"\bantibiotic|antifungal|antiviral|antimicrobial|anti-?tubercul|\w*cillin\b|ceftriaxone"
                         r"|ceftazidime|cefepime|vancomycin|metronidazole|azithromycin|acyclovir|fluconazole"
                         r"|doxycycline|meropenem|levofloxacin|ciprofloxacin|rifampic|isoniazid|gentamicin"
                         r"|clindamycin|piperacillin")),
    ("surgery", rx(r"\b\w*(ectomy|ostomy|otomy|plasty|pexy|rraphy)\b|\b(surgical (excision|resection|removal|treatment)"
                   r"|underwent (an? )?(operation|surgery|procedure)|operative (management|treatment)"
                   r"|embolizat|stent(ing|ed)?|catheter(izat|isat)|drainage was performed|debridement"
                   r"|reduction and fixation|graft|transplantation)")),
    ("supportive_palliative", rx(r"\bsupportive (care|treatment|management|therapy)|palliativ|conservative (management|treatment|therapy)"
                                 r"|physiotherap|rehabilitat|blood transfusion|oxygen (therapy|supplementation)"
                                 r"|analgesi|pain (control|management|relief)|intravenous (fluids|hydration)"
                                 r"|nutritional support|observation alone|watchful waiting")),
    ("other_pharmacotherapy", rx(r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml|iu|units?)\b|\b(tablet|capsule|injection|infusion"
                                 r"|steroid|prednis|dexamethasone|insulin|anticoagula|heparin|warfarin|statin"
                                 r"|was (started|treated|commenced) (on|with)|prescribed|medication|drug therapy)")),
]

SMOKE_CUE = rx(r"\bsmok(e|ed|er|ers|ing)\b|\btobacco\b|\bcigarett|\bpack[- ]?year|\bnicotine\b")
SMOKE_NEVER = rx(r"\b(non[- ]?smoker|never smoked|never a smoker|no (history of |significant )?(smoking|tobacco)"
                 r"|denie[sd] (any )?(smoking|tobacco|cigarette)|does not smoke|did not smoke|no smoking history"
                 r"|negative for (smoking|tobacco)|no tobacco (use|abuse))\b")
SMOKE_FORMER = rx(r"\b(ex[- ]?smoker|former smoker|previous smoker|quit smoking|stopped smoking|ceased smoking"
                  r"|gave up smoking|smoked (until|for \d+ years)|history of smoking .{0,30}(quit|stopped|ceased)"
                  r"|had smoked)\b")
SMOKE_CURRENT = rx(r"\b(current smoker|active smoker|heavy smoker|chronic smoker|is a smoker|smokes\b"
                   r"|smoking \d+|\d+ pack[- ]?years?|\d+ cigarettes|daily smoker|tobacco (use|abuse|consumption)"
                   r"|smoking history of)\b")


# --------------------------------------------------------------------------- #
# score rules
# --------------------------------------------------------------------------- #

SEVERE = rx(r"\bsevere|marked(ly)?|intense|excruciating|extensive|profound|critical|life[- ]threatening"
            r"|high[- ]grade|grade (iii|iv|3|4)\b|stage (iii|iv|3|4)\b|massive|florid|debilitating|unbearable"
            r"|emergenc|resuscitat|shock\b|\bicu\b")
MODERATE = rx(r"\bmoderate|significant|considerable|progressiv|worsening|increasing|persistent|recurrent"
              r"|intermittent|gradually|moderately")
MILD = rx(r"\bmild|slight|minimal|low[- ]grade|painless|subtle|occasional|transient|trivial|small\b|minor\b")

RESP_GOOD = rx(r"\b(complete (response|remission|resolution|recovery)|resolved|resolution of|cured|full recovery"
               r"|uneventful (recovery|postoperative)|asymptomatic|remission|symptom[- ]free|disease[- ]free"
               r"|excellent (response|outcome|result)|responded well|marked improvement|significant improvement"
               r"|no evidence of (recurrence|residual|disease)|successful(ly)?|discharged in good|made a full recovery"
               r"|returned to normal|normali[sz]ed)\b")
RESP_BAD = rx(r"\b(no (response|improvement|change|relief|benefit)|did not (improve|respond)|failed to (improve|respond)"
              r"|unsuccessful|ineffective|refractory|resistant to|deteriorat|worsen(ed|ing)|progress(ed|ion) of (the )?disease"
              r"|relapse[d]?|recurren(ce|t)|died|death|expired|deceased|fatal|succumb|poor (response|outcome|prognosis)"
              r"|complication[s]?|adverse (event|reaction|effect)|discontinued due to|intoleran)\b")
RESP_PARTIAL = rx(r"\b(partial (response|remission|improvement|recovery)|some improvement|gradual improvement"
                  r"|slowly improv|improved (slightly|gradually)|stable disease|ongoing (treatment|therapy)"
                  r"|continues? (on|to receive)|symptoms (decreased|reduced|lessened))\b")

# any distinct investigation mention, used as a count proxy for work-up intensity
WORKUP_ITEM = rx(r"\b(ct|mri|x-?ray|ultrasound|ultrasonograph|sonograph|pet|scintigraph|angiograph|echocardiograph"
                 r"|radiograph|mammograph|doppler|tomograph|spect|endoscop|colonoscop|bronchoscop|cystoscop"
                 r"|gastroscop|biops|histopatholog|cytolog|immunohistochem|blood (test|count|culture)|serum|cbc"
                 r"|urinalysis|urine culture|liver function|renal function|electrolyte|crp|esr|ecg|ekg|eeg|emg"
                 r"|spirometr|pulmonary function|nerve conduction|audiometr|culture|examination revealed"
                 r"|physical examination|laborator|genetic (test|analysis)|sequencing|karyotyp)\b")



# --------------------------------------------------------------------------- #
# tuned rule bodies (thresholds and cue sets fitted on the development split)
# --------------------------------------------------------------------------- #

ADMIT_STRONG = rx(r"\b(was|were|been)\s+(admitted|hospitali[sz]ed)\b|\badmitted to\b|\bon admission\b"
                  r"|\bhospital course\b|\bintensive care unit\b|\b(icu|hdu)\b|\binpatient\b"
                  r"|\bpostoperative day\b|\b(hospital|postoperative) day \d+\b|\bwas discharged\b"
                  r"|\bdischarged (home|on|after|from)\b|\bward\b|\blength of stay\b|\bstayed in (the )?hospital\b")
ADMIT_WEAK = rx(r"\badmission\b|\bhospitali[sz]ation\b|\btransferred to (our|the|another) (hospital|centre|center|unit|ward|department)\b"
                r"|\breferred to our (hospital|institution|centre|center)\b|\bemergency (department|room|unit)\b"
                r"|\bunder general an(a)?esthesia\b|\bpreoperative(ly)?\b|\bintraoperative(ly)?\b")

CHRONIC_DISEASE = rx(r"\b(hypertension|hypertensive|diabet(es|ic)|asthma|copd|chronic (kidney|renal|liver|obstructive)"
                     r"|cirrhosis|hepatitis [bc]|hiv|ischemic heart|coronary artery disease|atrial fibrillation"
                     r"|heart failure|epilep(sy|tic)|rheumatoid|lupus|thyroid disease|hypothyroid|hyperthyroid"
                     r"|dyslipid(a)?emia|hypercholesterol|obesity|tuberculosis|psoriasis|crohn|ulcerative colitis)\b")
HISTORY_STRONG = rx(r"\b(past|previous|prior) (medical|surgical|clinical) history\b|\bhistory of\b|\bknown case of\b"
                    r"|\bknown (diabetic|hypertensive|asthmatic|epileptic|case)\b|\bhad been diagnosed\b"
                    r"|\bpreviously diagnosed\b|\bcomorbid|\bco-?morbidit|\b(suffered|suffering) from\b"
                    r"|\b\d+[- ]year history\b|\bunderlying (disease|condition|illness)\b|\bmedical background\b"
                    r"|\bhad undergone\b|\bhas been on\b|\bwas on (regular |long[- ]term )?(medication|treatment|therapy)\b"
                    r"|\b(diagnosed|treated) .{0,30}\b(years?|months?) (ago|prior|previously|earlier)\b"
                    r"|\bin (19|20)\d\d\b.{0,40}\b(diagnos|underwent|treated)")

FOLLOWUP_PLAN = rx(r"\bfollow(ed|ing)?[- ]?up (visit|appointment|examination|assessment|care|plan|schedule|protocol|period)\b"
                   r"|\b(regular|periodic|close|long[- ]term|annual|monthly|routine) follow[- ]?up\b"
                   r"|\bfollow[- ]?up (was|is|will be) (arranged|scheduled|planned|advised|recommended|continued|ongoing)\b"
                   r"|\b(scheduled|arranged|advised|planned|recommended) for .{0,30}follow[- ]?up\b"
                   r"|\b(will be|to be|is being) followed[- ]?up\b|\bunder (regular )?(follow[- ]?up|surveillance)\b"
                   r"|\badvised to (return|come back|report)\b|\bre-?examin(ed|ation) (was )?(scheduled|planned)\b"
                   r"|\bfollow[- ]?up (at|after|every) \d+\s?(week|month|year)")

DRUG_ADMIN = rx(r"\b(was|were)\s+(administered|prescribed|started on|initiated on|commenced on|given|treated with|put on|placed on)\b"
                r"|\b(therapy|treatment) (was|were) (started|initiated|commenced|administered|given)\b"
                r"|\breceived .{0,40}\b(therapy|chemotherapy|antibiotics?|steroids?|infusion|injection)\b")
DOSE_STRICT = rx(r"\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|iu|units?)\b(\s?(/|per)\s?(kg|m2|day|d|h|hr|dose))?"
                 r"|\b(mg|g|mcg)\s?/\s?(kg|m2|day|dl)\b|\b(once|twice|three times|\d+ times) (a |per )?(daily|day|week)\b")

SEVERE_STRICT = rx(r"\bsevere(ly)?\b|\bexcruciating\b|\bunbearable\b|\bintense\b|\bprofound\b|\bdebilitating\b"
                   r"|\blife[- ]threatening\b|\bhigh[- ]grade\b|\bgrade (iii|iv|3|4)\b|\bstage (iii|iv|3|4)\b"
                   r"|\bextensive\b|\bmassive\b|\bmarked(ly)?\b|\bcritical(ly)?\b|\bfulminant\b|\bflorid\b")
MILD_STRICT = rx(r"\bmild(ly)?\b|\bslight(ly)?\b|\bminimal(ly)?\b|\blow[- ]grade\b|\bpainless\b|\bsubtle\b"
                 r"|\boccasional(ly)?\b|\btransient\b|\btrivial\b|\bminor\b|\bwell[- ]tolerated\b|\bgrade (i|1)\b")

# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #

def _first(note, table, default):
    for name, pattern in table:
        if pattern.search(note):
            return name
    return default


def _argmax_mentions(note, table, default, priority=()):
    """Pick the modality mentioned most often; a single lexical hit anywhere
    (the `_first` rule) badly over-predicts biopsy and surgery."""
    counts = {name: len(pattern.findall(note)) for name, pattern in table}
    if not any(counts.values()):
        return default
    best = max(counts.values())
    tied = [n for n, c in counts.items() if c == best]
    for n in priority:
        if n in tied:
            return n
    return tied[0]


def predict(note):
    """full_note -> {question_id: predicted label} for all 11 variables.

    Every threshold and cue-set choice below was selected by grid search on the
    development split only; the test split was read once, afterwards."""
    p = {}

    # admitted unless the note reads as a purely outpatient episode
    p["hospital_admission"] = not (ADMIT_NEG.search(note) and not ADMIT_STRONG.search(note)
                                   and not SURGERY.search(note))
    p["surgical_management"] = bool(SURGERY.search(note))
    p["drug_therapy"] = bool(DOSE_STRICT.search(note)) or bool(
        DRUG_ADMIN.search(note) and DRUGNAME.search(note))
    p["prior_comorbidity"] = (len(HISTORY_STRONG.findall(note))
                              + len(CHRONIC_DISEASE.findall(note))) >= 1
    # a single "follow-up" mention is usually narrative; two are a plan
    p["follow_up_planned"] = len(FOLLOWUP_PLAN.findall(note)) >= 2

    p["primary_diagnostic_modality"] = _first(note, MODALITY, "not_reported")
    p["principal_medical_therapy"] = _first(note, THERAPY, "none_recorded")

    windows = " ".join(near(note, SMOKE_CUE, 120))
    if not windows:
        p["smoking_status"] = "not_documented"
    elif SMOKE_NEVER.search(windows):
        p["smoking_status"] = "never"
    elif SMOKE_FORMER.search(windows):
        p["smoking_status"] = "former"
    elif SMOKE_CURRENT.search(windows):
        p["smoking_status"] = "current"
    else:
        p["smoking_status"] = "not_documented"

    p["symptom_severity"] = 2 if SEVERE_STRICT.search(note) else (
        0 if MILD_STRICT.search(note) else 1)

    tail = note[len(note) // 2:]  # outcome wording lives in the second half
    good, bad = len(RESP_GOOD.findall(tail)), len(RESP_BAD.findall(tail))
    p["treatment_response"] = 0 if bad > good else (2 if good > bad + 1 else 1)

    n_items = len(set(m.group(0).lower() for m in WORKUP_ITEM.finditer(note)))
    p["diagnostic_workup_intensity"] = 0 if n_items == 0 else (1 if n_items <= 4 else 2)

    return p
