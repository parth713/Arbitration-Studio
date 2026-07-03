"""MACT compensation: grounded fact extraction + deterministic computation.

Two stages, deliberately separated so the arithmetic is auditable:

1. ``extract_case_facts`` uses the LLM purely to *read facts off the record*
   (age, income, dependents, disability, expenses) grounded on graph-retrieved
   chunks, returning each fact with the citation it came from (or ``None`` when
   the record is silent).
2. ``compute_death_award`` / ``compute_injury_award`` apply the settled
   multiplier method in plain Python — no LLM in the loop — so the numbers are
   reproducible and every line carries the rule that produced it.

Legal basis for the constants:
- Multiplier table and personal-expense deduction: *Sarla Verma v. DTC*,
  (2009) 6 SCC 121.
- Future prospects and the conventional heads (loss of estate, consortium,
  funeral): *National Insurance Co. v. Pranay Sethi*, (2017) 16 SCC 680, with
  the convention of enhancing the conventional heads by 10% every three years.
- Separate "loss of love and affection" is not awarded: *Magma General
  Insurance v. Nanu Ram*, (2018) 18 SCC 130 / *United India v. Satinder Kaur*,
  (2021) 11 SCC 780 — kept as a zero line the tribunal may adjust.

All constants live in this module so they are easy to update as the case law
evolves, and every computed figure is surfaced to the judge for override.
"""

from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
import json
import re
from typing import Dict, List, Optional

from openai import OpenAI

from arbitration_studio.graph_rag import Chunk, GraphIndex, retrieve_context


# --------------------------------------------------------------------------- #
# Statutory constants                                                         #
# --------------------------------------------------------------------------- #

# Pranay Sethi conventional heads, 2017 base figures (rupees).
PRANAY_SETHI_BASE_YEAR = 2017
CONVENTIONAL_HEADS_BASE = {
    "loss_of_estate": 15000,
    "consortium_per_claimant": 40000,
    "funeral_expenses": 15000,
}
CONVENTIONAL_ENHANCEMENT = 0.10  # +10% per completed 3-year block.


def multiplier_for_age(age: Optional[float]) -> Optional[int]:
    """Sarla Verma multiplier by age band."""
    if age is None:
        return None
    if age <= 25:
        return 18
    if age <= 30:
        return 17
    if age <= 35:
        return 16
    if age <= 40:
        return 15
    if age <= 45:
        return 14
    if age <= 50:
        return 13
    if age <= 55:
        return 11
    if age <= 60:
        return 9
    if age <= 65:
        return 7
    return 5


def future_prospects_rate(age: Optional[float], employment_type: Optional[str]) -> float:
    """Pranay Sethi future-prospects addition as a fraction of income."""
    if age is None:
        return 0.0
    salaried = (employment_type or "").lower().startswith("salar") or (employment_type or "").lower() == "permanent"
    if age < 40:
        return 0.50 if salaried else 0.40
    if age <= 50:
        return 0.30 if salaried else 0.25
    if age <= 60:
        return 0.15 if salaried else 0.10
    return 0.0  # Above 60: nil under Pranay Sethi (tribunal may adjust).


def personal_expense_fraction(
    num_dependents: Optional[int],
    marital_status: Optional[str],
    bachelor_with_dependent_family: bool = False,
) -> Optional[Fraction]:
    """Sarla Verma deduction for the deceased's personal/living expenses.

    Bachelor: normally 1/2, but 1/3 in the recognised exception where the
    deceased supported a widowed mother and younger non-earning siblings.
    Married: 1/3 (2-3 dependants), 1/4 (4-6), 1/5 (>6).
    """
    status = (marital_status or "").lower()
    if status.startswith("unmarr") or status in {"bachelor", "single", "spinster"}:
        return Fraction(1, 3) if bachelor_with_dependent_family else Fraction(1, 2)
    if num_dependents is None:
        return None
    if num_dependents <= 3:  # 2-3 members of the family
        return Fraction(1, 3)
    if num_dependents <= 6:  # 4-6 members
        return Fraction(1, 4)
    return Fraction(1, 5)  # > 6 members


def _accident_year(facts: "CaseFacts") -> Optional[int]:
    """The year of the accident (drives conventional-head escalation)."""
    for value in (facts.date_of_accident, facts.filing_date, facts.dar_date, facts.fir_date):
        if not value:
            continue
        years = re.findall(r"(?:19|20)\d{2}", normalize_numerals(str(value)))
        if years:
            return int(years[-1])
    return None


def _bachelor_family_exception(facts: "CaseFacts") -> bool:
    """True where a bachelor deceased supported a widowed mother + siblings."""
    status = (facts.marital_status or "").lower()
    if not (status.startswith("unmarr") or status in {"bachelor", "single", "spinster"}):
        return False
    relations = " ".join(
        (d.get("relation") or "").lower() for d in facts.dependents if isinstance(d, dict)
    )
    has_mother = "mother" in relations
    has_sibling = any(term in relations for term in ("brother", "sister", "sibling"))
    return has_mother and has_sibling


def conventional_heads(award_year: int = 2024) -> Dict[str, float]:
    blocks = max(0, (award_year - PRANAY_SETHI_BASE_YEAR) // 3)
    factor = (1 + CONVENTIONAL_ENHANCEMENT) ** blocks
    return {key: round(base * factor) for key, base in CONVENTIONAL_HEADS_BASE.items()}


# --------------------------------------------------------------------------- #
# Extracted facts                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class CaseFacts:
    case_type: str  # "death" | "injury"
    name: Optional[str] = None
    age: Optional[float] = None
    occupation: Optional[str] = None
    employment_type: Optional[str] = None  # "salaried" | "self_employed"
    monthly_income: Optional[float] = None
    marital_status: Optional[str] = None
    num_dependents: Optional[int] = None
    dependents: List[Dict[str, object]] = field(default_factory=list)
    date_of_accident: Optional[str] = None
    # Procedural dates for the statutory-timeline / limitation check.
    fir_date: Optional[str] = None
    far_date: Optional[str] = None
    iar_date: Optional[str] = None
    dar_date: Optional[str] = None
    filing_date: Optional[str] = None
    vehicle_number: Optional[str] = None
    insurer: Optional[str] = None
    # Actors (ontology Module 1)
    driver_name: Optional[str] = None
    owner_name: Optional[str] = None
    investigating_officer: Optional[str] = None
    police_station: Optional[str] = None
    hospital: Optional[str] = None
    eyewitness: Optional[str] = None
    # Injury-specific
    nature_of_injury: Optional[str] = None
    disability_percent: Optional[float] = None
    functional_disability_percent: Optional[float] = None
    treatment_months: Optional[float] = None
    # Expenses (rupees)
    medical_expenses: Optional[float] = None
    conveyance_expenses: Optional[float] = None
    special_diet_expenses: Optional[float] = None
    attendant_expenses: Optional[float] = None
    artificial_limb_cost: Optional[float] = None
    # Provenance + diagnostics
    sources: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)


# Fields the LLM should try to read off the record.
_EXTRACT_FIELDS = [
    "name", "age", "occupation", "employment_type", "monthly_income",
    "marital_status", "num_dependents", "date_of_accident", "fir_date",
    "far_date", "iar_date", "dar_date", "filing_date", "vehicle_number",
    "insurer", "driver_name", "owner_name", "investigating_officer",
    "police_station", "hospital", "eyewitness", "nature_of_injury",
    "disability_percent", "functional_disability_percent", "treatment_months",
    "medical_expenses", "conveyance_expenses", "special_diet_expenses",
    "attendant_expenses", "artificial_limb_cost",
]
_NUMERIC_FIELDS = {
    "age", "monthly_income", "num_dependents", "disability_percent",
    "functional_disability_percent", "treatment_months", "medical_expenses",
    "conveyance_expenses", "special_diet_expenses", "attendant_expenses",
    "artificial_limb_cost",
}


def extract_case_facts(
    index: GraphIndex,
    api_key: str,
    chat_model: str,
    embedding_model: str,
    case_type: str,
) -> CaseFacts:
    query = (
        "motor accident victim deceased injured name age occupation income salary "
        "dependents legal representatives marital status disability percentage "
        "hospitalisation medical expenses conveyance attendant vehicle insurance policy "
        "date of accident"
    )
    chunks = retrieve_context(index, query, api_key=api_key, embedding_model=embedding_model, top_k=24)
    facts = CaseFacts(case_type=case_type, chunks=chunks)

    if not api_key:
        facts.notes.append("OPENAI_API_KEY missing — no facts extracted; enter figures manually.")
        return facts

    context = "\n\n".join(f"[{chunk.citation}]\n{chunk.text}" for chunk in chunks) or "No context."
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=chat_model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "You extract structured facts from an Indian Motor Accident Claims Tribunal "
                    "case record for compensation assessment. Use ONLY the provided context. "
                    "The record may be in English, Hindi (Devanagari) or Urdu (Nastaliq). "
                    "Return a single JSON object. For each field provide an object "
                    '{"value": <value or null>, "citation": "<citation>"}. '
                    "The citation MUST be copied verbatim from the bracketed tag that sits above the "
                    "text you relied on. Each tag has the form 'C<n> | <filename> p. <page>' and "
                    "identifies three things: the graph node id (C<n>), the document name, and the "
                    "page number — e.g. 'C7 | DAR.pdf p. 3'. Whenever value is not null you MUST "
                    "return this full citation (graph node + document + page); never shorten it, "
                    "never invent one, and never cite a tag that is not in the context. If a value "
                    "comes from more than one place, give the most specific tag. "
                    "Use null (with citation '') when the record does not state the fact — never guess. "
                    "Convert any Hindi/Urdu numerals to Western digits, and read amounts written "
                    "in words (e.g. 'pandrah hazaar' / 'पंद्रह हज़ार' = 15000). "
                    "Transliterate person and place names to Roman/English script for the output "
                    "values (the English award is drawn from these). "
                    "monthly_income in rupees per month; if only annual income is given, divide by 12. "
                    "employment_type must be 'salaried' or 'self_employed'. "
                    "num_dependents is the count of legal representatives/dependents. "
                    "Percentages as plain numbers (e.g. 40 for 40%). "
                    "Dates (date_of_accident, fir_date, far_date, iar_date, dar_date, filing_date) "
                    "as the date the document was made/filed, in DD-MM-YYYY; fir_date is the FIR date, "
                    "far_date the First Accident Report date, dar_date the Detailed Accident Report "
                    "filing date, filing_date the claim petition / DAR registration date. "
                    "Actors: driver_name (driver of the offending vehicle), owner_name (registered "
                    "owner), investigating_officer (IO name), police_station, hospital (treating "
                    "hospital), eyewitness (name(s)) — read in any script, transliterated to Roman."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Case type: {case_type}.\n"
                    f"Fields to extract: {', '.join(_EXTRACT_FIELDS)}.\n"
                    "Also return a 'dependents' array of objects {name, age, relation} for the "
                    "deceased's legal representatives if present.\n\n"
                    f"Context:\n{context}"
                ),
            },
        ],
    )
    _apply_extraction(facts, response.output_text)
    return facts


def _apply_extraction(facts: CaseFacts, output_text: str) -> None:
    data = _parse_json(output_text)
    if not data:
        facts.notes.append("Could not parse extraction output as JSON; enter figures manually.")
        return

    for key in _EXTRACT_FIELDS:
        entry = data.get(key)
        value, citation = _unwrap(entry)
        if value is None:
            continue
        if key in _NUMERIC_FIELDS:
            value = _to_number(value)
            if value is None:
                continue
            if key == "num_dependents":
                value = int(value)
        setattr(facts, key, value)
        if citation:
            facts.sources[key] = citation

    deps = data.get("dependents")
    if isinstance(deps, dict):
        deps = deps.get("value")
    if isinstance(deps, list):
        facts.dependents = [d for d in deps if isinstance(d, dict)]
        if facts.num_dependents is None and facts.dependents:
            facts.num_dependents = len(facts.dependents)


# --------------------------------------------------------------------------- #
# Computation                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class LineItem:
    label: str
    amount: Optional[float]
    basis: str
    editable: bool = True
    in_total: bool = True  # False for intermediate working lines (e.g. multiplier).


@dataclass
class Computation:
    case_type: str
    line_items: List[LineItem]
    total: float
    summary: Dict[str, object]
    missing_fields: List[str]


def compute_death_award(facts: CaseFacts, award_year: Optional[int] = None) -> Computation:
    items: List[LineItem] = []
    missing: List[str] = []

    # Conventional heads escalate with the ACCIDENT year, not a fixed year.
    year = award_year or _accident_year(facts)
    if year is None:
        year = date.today().year
        missing.append("date of accident (year drives conventional heads)")
    heads = conventional_heads(year)

    income = facts.monthly_income
    if income is None:
        missing.append("monthly_income")
    age = facts.age
    if age is None:
        missing.append("age")

    fp_rate = future_prospects_rate(age, facts.employment_type)
    bachelor_exception = _bachelor_family_exception(facts)
    pe_fraction = personal_expense_fraction(facts.num_dependents, facts.marital_status, bachelor_exception)
    if pe_fraction is None:
        missing.append("num_dependents/marital_status")
    multiplier = multiplier_for_age(age)

    A = income or 0.0
    B = A * fp_rate
    pe_frac_val = float(pe_fraction) if pe_fraction is not None else 0.0
    C = (A + B) * pe_frac_val
    D = (A + B) - C
    annual = D * 12
    E = multiplier or 0
    F = annual * E

    emp = facts.employment_type or "unknown"
    # Intermediate working lines (in_total=False): they build up to F, which is summed.
    items.append(LineItem("Income of deceased (A) — monthly", income, "From record", True, in_total=False))
    items.append(LineItem(
        "Add: Future prospects (B)", round(B) if income else None,
        f"{int(fp_rate*100)}% (Pranay Sethi, age {age}, {emp})", False, in_total=False))
    pe_basis = f"deduction {pe_fraction if pe_fraction else '?'} (Sarla Verma"
    status = (facts.marital_status or "").lower()
    if status.startswith("unmarr") or status in {"bachelor", "single", "spinster"}:
        pe_basis += ", bachelor exception 1/3" if bachelor_exception else ", bachelor 1/2"
    else:
        pe_basis += f", {facts.num_dependents} dependents"
    pe_basis += ")"
    items.append(LineItem(
        "Less: Personal expenses (C)", round(C) if income else None, pe_basis, False, in_total=False))
    items.append(LineItem("Monthly loss of dependency (D = A+B−C)", round(D) if income else None, "computed", False, in_total=False))
    items.append(LineItem("Annual loss of dependency (D×12)", round(annual) if income else None, "computed", False, in_total=False))
    items.append(LineItem("Multiplier (E)", E or None, f"Sarla Verma, age {age}", False, in_total=False))
    items.append(LineItem("Total loss of dependency (F = D×12×E)", round(F) if income else None, "computed", False, in_total=True))
    items.append(LineItem("Medical expenses (G)", facts.medical_expenses, "From record", True, in_total=True))
    consortium_claimants = facts.num_dependents or 1
    H = heads["consortium_per_claimant"] * consortium_claimants
    items.append(LineItem(
        "Loss of consortium (H)", H,
        f"{consortium_claimants} × ₹{heads['consortium_per_claimant']:,} (Pranay Sethi, accident year {year})", True, in_total=True))
    items.append(LineItem(
        "Loss of love and affection (I)", 0,
        "No separate award (Magma General v. Nanu Ram); adjust if tribunal differs", True, in_total=True))
    items.append(LineItem(
        "Loss of estate (J)", heads["loss_of_estate"],
        f"₹{heads['loss_of_estate']:,} (Pranay Sethi, accident year {year})", True, in_total=True))
    items.append(LineItem(
        "Funeral expenses (K)", heads["funeral_expenses"],
        f"₹{heads['funeral_expenses']:,} (Pranay Sethi, accident year {year})", True, in_total=True))

    total = _sum_items(items)
    summary = {
        "accident_year": year,
        "future_prospects_rate": fp_rate,
        "personal_expense_fraction": str(pe_fraction) if pe_fraction else None,
        "bachelor_exception_applied": bachelor_exception,
        "multiplier": E,
        "total_loss_of_dependency": round(F) if income else None,
    }
    return Computation("death", items, round(total), summary, missing)


def compute_injury_award(facts: CaseFacts, award_year: int = 2024) -> Computation:
    items: List[LineItem] = []
    missing: List[str] = []

    income = facts.monthly_income
    age = facts.age
    multiplier = multiplier_for_age(age)
    disability = facts.functional_disability_percent
    if disability is None:
        disability = facts.disability_percent
    if disability is None:
        missing.append("disability_percent")
    if income is None:
        missing.append("monthly_income")
    if age is None:
        missing.append("age")

    # Pecuniary loss
    loss_during_treatment = None
    if income is not None and facts.treatment_months is not None:
        loss_during_treatment = round(income * facts.treatment_months)
    annual_income = income * 12 if income is not None else None
    loss_future_income = None
    if annual_income is not None and disability is not None and multiplier:
        loss_future_income = round(annual_income * (disability / 100.0) * multiplier)

    items.append(LineItem("Expenditure on treatment", facts.medical_expenses, "From record", True))
    items.append(LineItem("Expenditure on conveyance", facts.conveyance_expenses, "From record", True))
    items.append(LineItem("Expenditure on special diet", facts.special_diet_expenses, "From record", True))
    items.append(LineItem("Cost of nursing / attendant", facts.attendant_expenses, "From record", True))
    items.append(LineItem("Cost of artificial limb", facts.artificial_limb_cost, "From record", True))
    items.append(LineItem(
        "Loss of income during treatment", loss_during_treatment,
        f"income × {facts.treatment_months} months" if loss_during_treatment is not None else "treatment period not stated", False))
    items.append(LineItem(
        "Loss of future income", loss_future_income,
        (f"annual income ₹{annual_income:,.0f} × {disability}% × multiplier {multiplier}"
         if loss_future_income is not None else "needs income, disability % and age"), False))
    # Non-pecuniary heads — discretionary, left for the tribunal to set.
    items.append(LineItem("Pain and suffering", None, "Tribunal discretion — enter figure", True))
    items.append(LineItem("Loss of amenities of life", None, "Tribunal discretion — enter figure", True))
    items.append(LineItem("Disfiguration", None, "Tribunal discretion — enter figure", True))

    total = _sum_items(items)
    summary = {
        "multiplier": multiplier,
        "disability_percent_applied": disability,
        "loss_of_future_income": loss_future_income,
    }
    return Computation("injury", items, round(total), summary, missing)


def field_gaps(facts: CaseFacts) -> List[str]:
    """Human-readable list of compensation-relevant facts absent from the record."""
    checks = {
        "Age of victim": facts.age,
        "Monthly income": facts.monthly_income,
        "Occupation / employment type": facts.occupation or facts.employment_type,
        "Date of accident": facts.date_of_accident,
        "Offending vehicle number": facts.vehicle_number,
        "Insurer": facts.insurer,
    }
    if facts.case_type == "death":
        checks["Number of dependents / legal representatives"] = facts.num_dependents
    else:
        checks["Nature of injury"] = facts.nature_of_injury
        checks["Disability percentage"] = (
            facts.functional_disability_percent if facts.functional_disability_percent is not None
            else facts.disability_percent
        )
    return [label for label, value in checks.items() if value in (None, "", [])]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _sum_items(items: List[LineItem]) -> float:
    return sum(item.amount for item in items if item.in_total and item.amount is not None)


def _unwrap(entry):
    if isinstance(entry, dict):
        return entry.get("value"), str(entry.get("citation") or "")
    return entry, ""


# Devanagari (०-९), Extended Arabic-Indic / Urdu (۰-۹) and Arabic-Indic (٠-٩)
# digits → ASCII, so incomes/ages written in Hindi or Urdu numerals survive.
_NUMERAL_MAP = {
    **{ord("०") + i: str(i) for i in range(10)},  # Devanagari U+0966..U+096F
    **{ord("۰") + i: str(i) for i in range(10)},  # Urdu       U+06F0..U+06F9
    **{ord("٠") + i: str(i) for i in range(10)},  # Arabic     U+0660..U+0669
}


def normalize_numerals(text: str) -> str:
    """Convert Devanagari / Urdu / Arabic-Indic digits to ASCII digits."""
    return text.translate(_NUMERAL_MAP)


def _to_number(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", normalize_numerals(value).replace(",", ""))
        if cleaned and cleaned != ".":
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None
