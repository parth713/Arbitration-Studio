"""MACT (Motor Accident Claims Tribunal) document classification.

Mirrors ``documents.py`` but works over the Delhi High Court Scheme for Motor
Accident Claims document set (FAR, IAR, DAR, FIR, medical and income records,
etc.) instead of arbitration pleadings. Extraction and the heuristic-scoring
helpers are reused from ``documents.py`` so the two domains stay consistent.
"""

from typing import Dict, Iterable, List, Tuple

from arbitration_studio.documents import (
    SourceDocument,
    _apply_patterns,
    _normalize,
    extract_file,
)
from arbitration_studio.ocr import extract_pdf_pages


# Canonical MACT document kinds the classifier can assign.
MACT_KINDS = [
    "First Accident Report (FAR)",
    "Interim Accident Report (IAR)",
    "Detailed Accident Report (DAR)",
    "Claim Petition",
    "FIR",
    "Charge Sheet",
    "Post-Mortem Report",
    "Death Certificate",
    "Medico-Legal Case (MLC)",
    "Disability Certificate",
    "Medical Record / Bills",
    "Income Proof",
    "Insurance Policy / Form",
    "Driver's Form",
    "Owner's Form",
    "Victim's Form",
    "Site Plan",
    "Mechanical Inspection Report",
    "Tribunal Award / Order",
]

# Filename signals: (regex, weight, reason). Filenames are short and reliable.
_FILENAME_PATTERNS: Dict[str, List[Tuple[str, int, str]]] = {
    "First Accident Report (FAR)": [
        (r"\bfar\b", 6, "filename contains FAR"),
        (r"\bfirst\s+accident\s+report\b", 7, "filename says first accident report"),
    ],
    "Interim Accident Report (IAR)": [
        (r"\biar\b", 6, "filename contains IAR"),
        (r"\binterim\s+accident\s+report\b", 7, "filename says interim accident report"),
    ],
    "Detailed Accident Report (DAR)": [
        (r"\bdar\b", 6, "filename contains DAR"),
        (r"\bdetailed\s+accident\s+report\b", 7, "filename says detailed accident report"),
    ],
    "Claim Petition": [
        (r"\bclaim\s+petition\b", 7, "filename says claim petition"),
        (r"\bmact\b", 3, "filename references MACT"),
        (r"\bsection\s+166\b", 5, "filename references section 166"),
        (r"\bdava\b", 3, "filename references dava/petition"),
    ],
    "FIR": [
        (r"\bfir\b", 7, "filename contains FIR"),
        (r"\bfirst\s+information\s+report\b", 7, "filename says first information report"),
    ],
    "Charge Sheet": [
        (r"\bcharge\s*sheet\b", 7, "filename says charge sheet"),
        (r"\bsection\s+173\b", 5, "filename references section 173 CrPC"),
    ],
    "Post-Mortem Report": [
        (r"\bpost\s*mortem\b", 7, "filename says post-mortem"),
        (r"\bpm\s+report\b", 5, "filename says PM report"),
        (r"\bautopsy\b", 6, "filename says autopsy"),
    ],
    "Death Certificate": [
        (r"\bdeath\s+certificate\b", 8, "filename says death certificate"),
    ],
    "Medico-Legal Case (MLC)": [
        (r"\bmlc\b", 7, "filename contains MLC"),
        (r"\bmedico\s*legal\b", 7, "filename says medico-legal"),
    ],
    "Disability Certificate": [
        (r"\bdisability\s+certificate\b", 8, "filename says disability certificate"),
        (r"\bdisability\b", 4, "filename says disability"),
    ],
    "Medical Record / Bills": [
        (r"\bdischarge\s+summary\b", 6, "filename says discharge summary"),
        (r"\bmedical\s+(?:bill|record|report)\b", 6, "filename says medical bill/record"),
        (r"\bhospital\b", 4, "filename references hospital"),
        (r"\btreatment\b", 4, "filename references treatment"),
    ],
    "Income Proof": [
        (r"\bsalary\s+slip\b", 7, "filename says salary slip"),
        (r"\bincome\b", 5, "filename says income"),
        (r"\bitr\b", 6, "filename contains ITR"),
        (r"\bform\s*16\b", 6, "filename says Form 16"),
        (r"\bpay\s*slip\b", 6, "filename says pay slip"),
    ],
    "Insurance Policy / Form": [
        (r"\binsurance\b", 6, "filename says insurance"),
        (r"\bpolicy\b", 5, "filename says policy"),
        (r"\bcover\s*note\b", 5, "filename says cover note"),
    ],
    "Driver's Form": [
        (r"\bdriver'?s?\s+form\b", 7, "filename says driver's form"),
        (r"\bform\s*[-\s]*iii\b", 5, "filename says Form III"),
    ],
    "Owner's Form": [
        (r"\bowner'?s?\s+form\b", 7, "filename says owner's form"),
        (r"\bform\s*[-\s]*iv\b", 5, "filename says Form IV"),
    ],
    "Victim's Form": [
        (r"\bvictim'?s?\s+form\b", 7, "filename says victim's form"),
        (r"\bform\s*[-\s]*via?\b", 5, "filename says Form VI"),
    ],
    "Site Plan": [
        (r"\bsite\s+plan\b", 8, "filename says site plan"),
    ],
    "Mechanical Inspection Report": [
        (r"\bmechanical\s+inspection\b", 8, "filename says mechanical inspection"),
        (r"\bmir\b", 4, "filename contains MIR"),
    ],
    "Tribunal Award / Order": [
        (r"\baward\b", 6, "filename says award"),
        (r"\border\b", 4, "filename says order"),
        (r"\bjudgment\b", 5, "filename says judgment"),
    ],
}

# Body-text signals: matched against the opening of each document.
_BODY_PATTERNS: Dict[str, List[Tuple[str, int, str]]] = {
    "First Accident Report (FAR)": [
        (r"\bfirst\s+accident\s+report\b", 10, "opening says first accident report"),
        (r"\bform\s*[-\s]*i\b.{0,40}\baccident\b", 5, "Form I accident report"),
    ],
    "Interim Accident Report (IAR)": [
        (r"\binterim\s+accident\s+report\b", 10, "opening says interim accident report"),
    ],
    "Detailed Accident Report (DAR)": [
        (r"\bdetailed\s+accident\s+report\b", 10, "opening says detailed accident report"),
        (r"\bform\s*[-\s]*vii\b", 4, "references Form VII"),
    ],
    "Claim Petition": [
        (r"\bclaim\s+petition\b", 8, "opening says claim petition"),
        (r"\bunder\s+section\s+166\b.{0,60}\bmotor\s+vehicles\s+act\b", 8, "petition under s.166 MV Act"),
        (r"\bpetition\s+for\s+(?:grant\s+of\s+)?compensation\b", 7, "petition for compensation"),
        (r"\bclaims?\s+tribunal\b", 3, "references claims tribunal"),
    ],
    "FIR": [
        (r"\bfirst\s+information\s+report\b", 10, "opening says first information report"),
        (r"\bunder\s+section\s+279\b", 5, "FIR u/s 279 IPC"),
        (r"\bunder\s+section\s+304[\s-]*a\b", 6, "FIR u/s 304-A IPC (death)"),
        (r"\bunder\s+section\s+337\b|\bunder\s+section\s+338\b", 5, "FIR u/s 337/338 IPC (injury)"),
    ],
    "Charge Sheet": [
        (r"\bcharge\s*sheet\b", 9, "opening says charge sheet"),
        (r"\bunder\s+section\s+173\b", 6, "report u/s 173 CrPC"),
    ],
    "Post-Mortem Report": [
        (r"\bpost[\s-]*mortem\b", 9, "opening says post-mortem"),
        (r"\bcause\s+of\s+death\b", 6, "states cause of death"),
        (r"\bautopsy\b", 7, "autopsy report"),
    ],
    "Death Certificate": [
        (r"\bcertificate\s+of\s+death\b|\bdeath\s+certificate\b", 9, "death certificate"),
        (r"\bregistration\s+of\s+births?\s+and\s+deaths?\b", 6, "births and deaths registration"),
    ],
    "Medico-Legal Case (MLC)": [
        (r"\bmedico[\s-]*legal\b", 9, "opening says medico-legal"),
        (r"\bmlc\s+no\b", 7, "MLC number"),
    ],
    "Disability Certificate": [
        (r"\bdisability\s+certificate\b", 9, "disability certificate"),
        (r"\bpermanent\s+(?:physical\s+)?disability\b", 7, "permanent disability"),
        (r"\b\d{1,3}\s*%\s*disability\b", 6, "percentage disability stated"),
    ],
    "Medical Record / Bills": [
        (r"\bdischarge\s+summary\b", 8, "discharge summary"),
        (r"\bdiagnosis\b", 4, "diagnosis"),
        (r"\bbill\s+no\b|\btotal\s+amount\b", 4, "billing details"),
        (r"\badmitted\b.{0,30}\bdischarged\b", 5, "admission/discharge dates"),
    ],
    "Income Proof": [
        (r"\bsalary\s+(?:slip|certificate)\b", 8, "salary slip/certificate"),
        (r"\bincome\s+tax\s+return\b", 7, "income tax return"),
        (r"\bgross\s+(?:salary|income)\b", 6, "gross salary/income"),
        (r"\bform\s*16\b", 6, "Form 16"),
    ],
    "Insurance Policy / Form": [
        (r"\bpolicy\s+no\b", 7, "policy number"),
        (r"\binsurance\s+(?:policy|company)\b", 6, "insurance policy/company"),
        (r"\bcertificate\s+of\s+insurance\b", 7, "certificate of insurance"),
    ],
    "Driver's Form": [
        (r"\bdriver'?s?\s+form\b", 9, "driver's form"),
        (r"\bdriving\s+licen[cs]e\b.{0,40}\bvalidity\b", 5, "driving licence validity (Form III)"),
    ],
    "Owner's Form": [
        (r"\bowner'?s?\s+form\b", 9, "owner's form"),
    ],
    "Victim's Form": [
        (r"\bvictim'?s?\s+form\b", 9, "victim's form"),
    ],
    "Site Plan": [
        (r"\bsite\s+plan\b", 9, "site plan"),
        (r"\bscale\b.{0,30}\broad\b", 4, "scaled road layout"),
    ],
    "Mechanical Inspection Report": [
        (r"\bmechanical\s+inspection\s+report\b", 9, "mechanical inspection report"),
        (r"\bmechanical\s+(?:condition|fitness)\b", 6, "mechanical condition"),
    ],
    "Tribunal Award / Order": [
        (r"\bmotor\s+accident\s+claims?\s+tribunal\b", 5, "before the MACT"),
        (r"\baward\b.{0,40}\bcompensation\b", 6, "award of compensation"),
        (r"\bit\s+is\s+(?:hereby\s+)?ordered\b", 5, "operative order language"),
    ],
}


def classify_mact_document(filename: str, text: str) -> str:
    return classify_mact_document_detailed(filename, text)[0]


def classify_mact_document_detailed(filename: str, text: str) -> Tuple[str, int, str]:
    normalized_filename = _normalize(filename)
    opening = _normalize(text[:4000])

    scores: Dict[str, int] = {kind: 0 for kind in MACT_KINDS}
    evidence: Dict[str, List[str]] = {kind: [] for kind in MACT_KINDS}

    for kind, patterns in _FILENAME_PATTERNS.items():
        _apply_patterns(scores, evidence, kind, normalized_filename, patterns)
    for kind, patterns in _BODY_PATTERNS.items():
        _apply_patterns(scores, evidence, kind, opening, patterns)

    best, best_score = max(scores.items(), key=lambda item: item[1])
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    if best_score < 5 or best_score - runner_up < 2:
        return "Supporting Document", best_score, _evidence_text(scores, evidence, "Supporting Document")
    return best, best_score, _evidence_text(scores, evidence, best)


def make_mact_source_documents(
    uploaded_files: Iterable,
    *,
    google_api_key: str = "",
    ocr_model: str = "",
    ocr_dpi: int = 300,
    enable_ocr: bool = True,
) -> List[SourceDocument]:
    """Classify uploaded MACT files, OCR'ing scanned PDF pages via Gemini.

    Scanned documents (FIRs, post-mortems, DARs) carry no extractable text, so
    pages that ``pypdf`` reads as empty/garbled are transcribed with Gemini when
    a Google API key is configured. Digital PDFs incur no OCR calls.
    """
    documents = []
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        ocr_used = False
        if uploaded_file.name.lower().endswith(".pdf"):
            pages, ocr_used = extract_pdf_pages(
                uploaded_file.getvalue(),
                api_key=google_api_key,
                model=ocr_model,
                dpi=ocr_dpi,
                enable_ocr=enable_ocr and bool(google_api_key),
            )
        else:
            pages = extract_file(uploaded_file)
        text = "\n".join(page.text for page in pages)
        kind, confidence, evidence = classify_mact_document_detailed(uploaded_file.name, text)
        if ocr_used:
            evidence = f"[Gemini OCR] {evidence}"
        documents.append(
            SourceDocument(
                doc_id=f"D{idx}",
                filename=uploaded_file.name,
                kind=kind,
                pages=pages,
                confidence=confidence,
                evidence=evidence,
            )
        )
    return documents


# Signals that decide whether the case is a death or an injury claim. The
# tribunal computes under Form XV (death) or Form XVI (injury) accordingly.
_DEATH_KINDS = {"Post-Mortem Report", "Death Certificate"}
_INJURY_KINDS = {"Disability Certificate"}
_DEATH_TEXT = [
    r"\bpost[\s-]*mortem\b",
    r"\bdeath\s+certificate\b",
    r"\bcause\s+of\s+death\b",
    r"\bdeceased\b",
    r"\bsection\s+304[\s-]*a\b",
    r"\blegal\s+(?:heirs?|representatives?)\b",
]
_INJURY_TEXT = [
    r"\bdisability\s+certificate\b",
    r"\bpermanent\s+disability\b",
    r"\bgrievous\s+(?:hurt|injur)",
    r"\bsection\s+337\b",
    r"\bsection\s+338\b",
    r"\bamputat",
]


def detect_case_type(documents: Iterable[SourceDocument]) -> Tuple[str, str]:
    """Return (case_type, evidence) where case_type is 'death', 'injury' or 'unknown'."""
    import re

    death_hits: List[str] = []
    injury_hits: List[str] = []
    for doc in documents:
        if doc.kind in _DEATH_KINDS:
            death_hits.append(f"{doc.kind} present ({doc.filename})")
        if doc.kind in _INJURY_KINDS:
            injury_hits.append(f"{doc.kind} present ({doc.filename})")
        opening = _normalize(doc.full_text[:6000])
        for pattern in _DEATH_TEXT:
            if re.search(pattern, opening):
                death_hits.append(f"'{pattern.strip(chr(92)+'b')}' in {doc.filename}")
                break
        for pattern in _INJURY_TEXT:
            if re.search(pattern, opening):
                injury_hits.append(f"'{pattern.strip(chr(92)+'b')}' in {doc.filename}")
                break

    death_score = len(death_hits)
    injury_score = len(injury_hits)
    if death_score == 0 and injury_score == 0:
        return "unknown", "No death or injury signals found in the record."
    if death_score >= injury_score:
        return "death", "; ".join(list(dict.fromkeys(death_hits))[:6])
    return "injury", "; ".join(list(dict.fromkeys(injury_hits))[:6])


# Documents the Delhi HC Scheme expects in the record, used for the gap checklist.
_EXPECTED_COMMON = [
    "First Accident Report (FAR)",
    "Detailed Accident Report (DAR)",
    "FIR",
    "Insurance Policy / Form",
]
_EXPECTED_DEATH = ["Post-Mortem Report", "Death Certificate", "Income Proof"]
_EXPECTED_INJURY = [
    "Medico-Legal Case (MLC)",
    "Disability Certificate",
    "Medical Record / Bills",
    "Income Proof",
]


def missing_documents(documents: Iterable[SourceDocument], case_type: str) -> List[str]:
    present = {doc.kind for doc in documents}
    expected = list(_EXPECTED_COMMON)
    if case_type == "death":
        expected += _EXPECTED_DEATH
    elif case_type == "injury":
        expected += _EXPECTED_INJURY
    return [kind for kind in dict.fromkeys(expected) if kind not in present]


def _evidence_text(scores: Dict[str, int], evidence: Dict[str, List[str]], selected: str) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:4]
    score_text = ", ".join(f"{kind}: {score}" for kind, score in ranked if score > 0) or "no signals"
    if selected == "Supporting Document":
        return f"No dominant MACT document signal. Top scores: {score_text}"
    reasons = "; ".join(dict.fromkeys(evidence[selected])) or "No evidence captured"
    return f"{reasons}. Top scores: {score_text}"
