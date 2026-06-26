from dataclasses import dataclass
import re
from typing import Dict, Iterable, List

from arbitration_studio.documents import SourceDocument


PLEADING_TYPES = {"Statement of Claim", "Statement of Defence", "Rejoinder"}
ADDRESS_PATTERN = re.compile(
    r"\b(?:"
    r"address|office|registered\s+office|principal\s+office|place\s+of\s+business|"
    r"having|situated|located|resident|residing|r/o|through|represented|"
    r"floor|road|street|avenue|lane|marg|nagar|tower|building|complex|"
    r"city|district|state|pin|pincode|postal|email|phone|mobile"
    r")\b|[@#]|\b\d{4,}\b",
    flags=re.IGNORECASE,
)
NAME_SUFFIX_PATTERN = re.compile(
    r"\b(?:"
    r"limited|ltd\.?|private\s+limited|pvt\.?\s+ltd\.?|llp|llc|inc\.?|"
    r"corporation|corp\.?|company|co\.?|gmbh|s\.?a\.?|plc|fze|fzc|trust|firm"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass
class PartyCandidate:
    name: str
    suggested_role: str
    score: int
    evidence: str


def extract_party_candidates(documents: Iterable[SourceDocument]) -> List[PartyCandidate]:
    parties: Dict[str, PartyCandidate] = {}
    for doc in documents:
        if doc.kind not in PLEADING_TYPES:
            continue
        cause_title = _cause_title_text(doc)
        if not cause_title:
            continue
        extracted = _extract_from_cause_title(cause_title)
        for role, name in extracted.items():
            if name:
                _set_party(parties, role, name, f"cause title in {doc.filename}")
        if "Claimant" in parties and "Respondent" in parties:
            break

    return [parties[role] for role in ("Claimant", "Respondent") if role in parties]


def rows_from_candidates(candidates: List[PartyCandidate]) -> List[Dict[str, object]]:
    return [
        {
            "Role": candidate.suggested_role,
            "Party": candidate.name,
            "Confidence": candidate.score,
            "Evidence": candidate.evidence,
        }
        for candidate in candidates
    ]


def selected_party(rows: List[Dict[str, object]], role: str) -> str:
    for row in rows:
        if row.get("Role") == role and row.get("Party"):
            return str(row["Party"])
    return ""


def _cause_title_text(doc: SourceDocument) -> str:
    if not doc.pages:
        return ""
    text = doc.pages[0].text[:4500]
    match = re.search(r"\bstatement\s+of\s+(?:claim|defen[cs]e)\b|\brejoinder\b", text, flags=re.IGNORECASE)
    if match and match.start() > 120:
        text = text[: match.start()]
    return text[:3000]


def _extract_from_cause_title(text: str) -> Dict[str, str]:
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    block = "\n".join(lines)

    parties = _extract_between_block(block)
    if "Claimant" not in parties:
        parties["Claimant"] = _extract_labeled_name(lines, "Claimant")
    if "Respondent" not in parties:
        parties["Respondent"] = _extract_labeled_name(lines, "Respondent")
    return {role: name for role, name in parties.items() if name}


def _extract_between_block(block: str) -> Dict[str, str]:
    compact = re.sub(r"\s+", " ", block)
    pattern = (
        r"\bbetween\s+(.{3,220}?)\s+(?:claimants?|petitioner|applicant)\s+"
        r"(?:and|v\.?|vs\.?|versus)\s+(.{3,220}?)\s+respondents?\b"
    )
    match = re.search(pattern, compact, flags=re.IGNORECASE)
    if not match:
        return {}
    return {
        "Claimant": _clean_name(match.group(1)),
        "Respondent": _clean_name(match.group(2)),
    }


def _extract_labeled_name(lines: List[str], role: str) -> str:
    label = r"claimants?|petitioner|applicant" if role == "Claimant" else r"respondents?"
    label_group = rf"(?:{label})"

    for index, line in enumerate(lines):
        colon_match = re.search(rf"\b{label_group}\b\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if colon_match:
            name = _clean_name(colon_match.group(1))
            if name:
                return name

        suffix_match = re.search(rf"^(.+?)\s*,?\s*(?:\.{2,})?\s*\b{label_group}\b\.?$", line, flags=re.IGNORECASE)
        if suffix_match:
            name = _clean_name(suffix_match.group(1))
            if name:
                return name

        if re.fullmatch(rf"(?:\.{{2,}}\s*)?(?:the\s+)?{label_group}\.?", line, flags=re.IGNORECASE):
            previous = _party_name_before_label(lines, index)
            if previous:
                return previous

    return ""


def _party_name_before_label(lines: List[str], index: int) -> str:
    block: List[str] = []
    for previous in reversed(lines[max(0, index - 8) : index]):
        if _is_party_boundary(previous):
            break
        block.insert(0, previous)

    candidates = [_clean_name(line) for line in block if _looks_like_name_line(line)]
    candidates = [name for name in candidates if name and not _is_noise(name)]
    if not candidates:
        return ""

    legal_names = [name for name in candidates if NAME_SUFFIX_PATTERN.search(name)]
    if legal_names:
        return legal_names[-1]
    return candidates[-1]


def _is_party_boundary(line: str) -> bool:
    if re.fullmatch(r"(?:and|v\.?|vs\.?|versus)", line, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(?:claimants?|petitioner|applicant|respondents?)\b", line, flags=re.IGNORECASE):
        return True
    if re.search(r"\bbefore\s+(?:the\s+)?(?:arbitral\s+tribunal|sole\s+arbitrator)\b", line, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_name_line(line: str) -> bool:
    cleaned = _clean_name(line)
    if not cleaned or _is_noise(cleaned):
        return False
    if len(cleaned.split()) > 10:
        return False
    if NAME_SUFFIX_PATTERN.search(cleaned):
        return True
    if ADDRESS_PATTERN.search(line) and cleaned == line.strip(" .,:;-"):
        return False
    return bool(re.search(r"[A-Z][a-zA-Z]", cleaned)) and not re.search(r"\d", cleaned)


def _set_party(parties: Dict[str, PartyCandidate], role: str, name: str, evidence: str) -> None:
    if not name or _is_noise(name):
        return
    current = parties.get(role)
    score = 20
    if not current or score > current.score:
        parties[role] = PartyCandidate(name=name, suggested_role=role, score=score, evidence=evidence)


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line)
    return line.strip(" \t:-;,")


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^between\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(?:m/s\.?|the)\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(?:claimants?|petitioner|applicant|respondents?)\b\.?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(?:and|v\.?|vs\.?|versus)\s+", "", name, flags=re.IGNORECASE)
    name = re.split(
        r"\b(?:having|through|represented|acting|address|registered\s+office|registered|office|"
        r"principal\s+office|place\s+of\s+business|situated|located|resident|residing|r/o|"
        r"incorporated|hereinafter|under)\b",
        name,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return name.strip(" .,:;-")


def _is_noise(name: str) -> bool:
    lowered = name.lower()
    if len(name) < 3 or len(name.split()) > 14:
        return True
    return lowered in {
        "statement of claim",
        "statement of defence",
        "statement of defense",
        "rejoinder",
        "arbitral tribunal",
        "before the arbitral tribunal",
    }
