"""Extract the cause-title parties of a MACT case.

Mirrors ``parties.py`` for arbitration. In a MACT claim the petitioner(s) are
the injured victim or the legal representatives of the deceased, and the
respondent(s) are the driver, owner and insurer of the offending vehicle. The
finer compensation facts (age, income, dependents, disability) are extracted
separately by ``mact_compensation`` from the graph-retrieved context.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List

from arbitration_studio.documents import SourceDocument
from arbitration_studio.parties import _clean_name, _is_noise

# Documents whose first page carries a cause title we can parse.
_TITLED_KINDS = {
    "Claim Petition",
    "Detailed Accident Report (DAR)",
    "Tribunal Award / Order",
}


@dataclass
class MactParty:
    name: str
    role: str
    evidence: str


def extract_mact_parties(documents: Iterable[SourceDocument]) -> List[MactParty]:
    parties: Dict[str, MactParty] = {}
    # Prefer titled documents, but fall back to any document if needed.
    ordered = sorted(
        documents,
        key=lambda doc: (doc.kind not in _TITLED_KINDS, doc.doc_id),
    )
    for doc in ordered:
        if not doc.pages:
            continue
        title = doc.pages[0].text[:4000]
        petitioner, respondent = _extract_versus(title)
        if petitioner and "Petitioner" not in parties:
            parties["Petitioner"] = MactParty(petitioner, "Petitioner / Claimant", f"cause title in {doc.filename}")
        if respondent and "Respondent" not in parties:
            parties["Respondent"] = MactParty(respondent, "Respondent", f"cause title in {doc.filename}")
        if "Petitioner" in parties and "Respondent" in parties:
            break
    return [parties[role] for role in ("Petitioner", "Respondent") if role in parties]


def mact_party_rows(parties: List[MactParty]) -> List[Dict[str, object]]:
    return [{"Role": p.role, "Party": p.name, "Evidence": p.evidence} for p in parties]


def party_name(parties: List[MactParty], role_prefix: str) -> str:
    for p in parties:
        if p.role.startswith(role_prefix) and p.name:
            return p.name
    return ""


_PETITIONER_LABEL = r"petitioners?|claimants?|applicants?"
_RESPONDENT_LABEL = r"respondents?|opposite\s+part(?:y|ies)"


def _extract_versus(text: str) -> tuple:
    compact = re.sub(r"\s+", " ", text)
    # Anchor on the "Versus" token; the petitioner sits just before it (after any
    # case-header boilerplate) and the respondent just after it.
    match = re.search(r"\b(?:versus|vs\.?|v/s|v\.)\b", compact, flags=re.IGNORECASE)
    if not match:
        return "", ""
    return _name_before(compact[: match.start()]), _name_after(compact[match.end():])


def _name_before(left: str) -> str:
    # Walk sentence-like segments backwards, skipping label-only fragments, and
    # take the nearest real name preceding the "Versus" token.
    for seg in reversed(re.split(r"\.\s+", left)):
        seg = seg.strip()
        if not seg or re.fullmatch(rf"(?:the\s+)?(?:{_PETITIONER_LABEL})\.?", seg, flags=re.IGNORECASE):
            continue
        seg = re.sub(rf"\b(?:{_PETITIONER_LABEL})\b\.?\s*$", "", seg, flags=re.IGNORECASE).strip()
        name = _trim_party(seg)
        if name:
            return name
    return ""


def _name_after(right: str) -> str:
    for seg in re.split(r"\.\s+", right):
        seg = seg.strip()
        if not seg:
            continue
        seg = re.sub(rf"\b(?:{_RESPONDENT_LABEL})\b\.?\s*$", "", seg, flags=re.IGNORECASE).strip()
        name = _trim_party(seg)
        if name:
            return name
    return ""


def _trim_party(raw: str) -> str:
    # Keep the lead name plus an "& Ors." marker if present, drop boilerplate.
    raw = re.split(r"\bin\s+the\s+matter\s+of\b|\bclaim\s+petition\b|\bmact\b", raw, flags=re.IGNORECASE)[-1]
    has_ors = bool(re.search(r"&\s*(?:ors?|others)\b|and\s+others\b", raw, flags=re.IGNORECASE))
    lead = re.split(r"&|\band\s+others\b|\bthrough\b|\bs/o\b|\bw/o\b|\bd/o\b|\br/o\b", raw, flags=re.IGNORECASE)[0]
    name = _clean_name(lead)
    if not name or _is_noise(name):
        return ""
    if has_ors:
        name = f"{name} & Ors."
    return name
