"""MACT statutory-timeline and limitation compliance checker.

Deterministic (no LLM): given the dates read from the record, it computes the
Delhi HC Scheme deadlines from the accident date (t0) and the Section 166(3)
Motor Vehicles Act limitation, and flags each as on-time / delayed / missing.

The limitation logic encodes the point that decided the sample case
(*Priyanka Tanwar v. Sandeep*): under Section 166(4), **any** accident report
forwarded to the Tribunal — including the FAR — is treated as the application
for compensation. So limitation is satisfied if the *earliest* report (usually
the FAR) reached the Tribunal within six months, even where the DAR was filed
later. IAR/DAR deadlines are themselves extendable by the Tribunal (Scheme
cl. 17), so a late DAR is flagged as delayed-but-curable, not fatal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import re
from typing import List, Optional, TYPE_CHECKING

from arbitration_studio.mact_compensation import normalize_numerals
from arbitration_studio.mact_ontology import (
    EXTENDABLE_FORMS,
    FORMS,
    LIMITATION_MONTHS,
    TIMELINE_FORMS,
)

if TYPE_CHECKING:
    from arbitration_studio.mact_compensation import CaseFacts


# Accepted date formats (Indian records are usually dd.mm.yyyy / dd-mm-yyyy).
_DATE_FORMATS = [
    "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y",
    "%d.%m.%y", "%d/%m/%y", "%d-%m-%y",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B, %Y", "%d %b, %Y",
]


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse a date string (incl. Hindi/Urdu numerals) into a ``date``."""
    if not value:
        return None
    text = normalize_numerals(str(value)).strip()
    # Pull a dd[sep]mm[sep]yyyy or yyyy-mm-dd token out of surrounding words.
    match = re.search(r"\d{1,4}[.\-/]\d{1,2}[.\-/]\d{2,4}", text)
    token = match.group(0) if match else text
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    # Fall back to trying the whole (worded) string for month-name formats.
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def add_months(start: date, months: int) -> date:
    """Add calendar months, clamping the day to the target month's length."""
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, _days_in_month(year, month))
    return date(year, month, day)


@dataclass
class ComplianceItem:
    label: str
    deadline: Optional[date]
    actual: Optional[date]
    status: str  # "On time" | "Delayed" | "Date not on record" | "N/A"
    note: str = ""


@dataclass
class ComplianceReport:
    accident_date: Optional[date]
    items: List[ComplianceItem] = field(default_factory=list)
    limitation_status: str = "Unknown"  # "Satisfied" | "At risk" | "Unknown"
    limitation_deadline: Optional[date] = None
    limitation_note: str = ""


# Map CaseFacts date fields → the Form whose deadline they answer.
_FACT_DATE_FOR_FORM = {
    "I": "far_date",
    "V": "iar_date",
    "VII": "dar_date",
}


def check_compliance(facts: "CaseFacts") -> ComplianceReport:
    accident = parse_date(facts.date_of_accident)
    report = ComplianceReport(accident_date=accident)

    if accident is None:
        report.limitation_note = "Accident date not on record — timeline cannot be assessed."
        return report

    for form_key in TIMELINE_FORMS:
        meta = FORMS[form_key]
        days = meta.get("deadline_days")
        deadline = accident + _timedelta_days(days) if isinstance(days, int) else None
        actual = parse_date(getattr(facts, _FACT_DATE_FOR_FORM.get(form_key, ""), None))
        label = f"{meta['name']} — due +{days}d"
        if actual is None:
            status, note = "Date not on record", ""
        elif deadline is not None and actual <= deadline:
            status, note = "On time", ""
        else:
            status = "Delayed"
            note = "Extendable by the Tribunal (Scheme cl. 17)." if form_key in EXTENDABLE_FORMS else "Delay to be explained."
        report.items.append(ComplianceItem(label, deadline, actual, status, note))

    _assess_limitation(facts, accident, report)
    return report


def _assess_limitation(facts: "CaseFacts", accident: date, report: ComplianceReport) -> None:
    deadline = add_months(accident, LIMITATION_MONTHS)
    report.limitation_deadline = deadline

    # Earliest accident report reaching the Tribunal (Section 166(4) cure).
    reports = {
        "FAR": parse_date(facts.far_date),
        "IAR": parse_date(facts.iar_date),
        "DAR": parse_date(facts.dar_date),
        "claim filing": parse_date(facts.filing_date),
    }
    dated = {name: d for name, d in reports.items() if d is not None}
    if not dated:
        report.limitation_status = "Unknown"
        report.limitation_note = (
            f"No report/filing date on record. Limitation is {LIMITATION_MONTHS} months from "
            f"{accident.isoformat()} (by {deadline.isoformat()}); confirm the FAR/DAR filing date."
        )
        return

    earliest_name = min(dated, key=lambda k: dated[k])
    earliest = dated[earliest_name]
    if earliest <= deadline:
        report.limitation_status = "Satisfied"
        report.limitation_note = (
            f"{earliest_name} filed {earliest.isoformat()} ≤ {deadline.isoformat()} "
            f"(6 months). Section 166(4) treats any accident report as the application, "
            f"so limitation is satisfied even if a later report (e.g. DAR) was filed beyond 6 months."
        )
    else:
        report.limitation_status = "At risk"
        report.limitation_note = (
            f"Earliest report ({earliest_name}) filed {earliest.isoformat()} is beyond the "
            f"6-month limit ({deadline.isoformat()}). Verify any extension (Scheme cl. 17) or "
            f"condonation of delay before treating the claim as maintainable."
        )


def _timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=days)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days
