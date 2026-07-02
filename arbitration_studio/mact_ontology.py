"""MACT domain ontology — a typed schema for the case knowledge graph.

Ported (no Neo4j) from the MACT knowledge-graph ontology notes and the Delhi HC
Scheme. It provides three things the rest of the app reuses:

1. ``FORMS`` — the statutory Form I–XIX set with filer, recipient and the
   deadline (days from the accident) — the single source of truth also used by
   ``mact_timeline`` for the compliance/limitation checker.
2. ``ENTITY_CLASSES`` / ``PRECEDENTS`` — the ontology's actor classes and the
   multiplier-doctrine lineage (Susamma Thomas → Trilok Chandra → Sarla Verma →
   Reshma Kumari → Pranay Sethi), for typing entities and citing authority.
3. ``enrich_case_graph`` — turns the generic per-case graph into a *typed*
   knowledge graph: document nodes are tagged with their ontology Form/class,
   and the extracted facts become typed role nodes (Deceased, Insurer, Vehicle…)
   hung off a central Case node.

This replaces generic capitalized-token entities with ontology-typed nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from arbitration_studio.documents import SourceDocument
    from arbitration_studio.mact_compensation import CaseFacts


# --------------------------------------------------------------------------- #
# Statutory forms (Delhi HC Scheme). deadline_days = days from accident (t0);
# None where the deadline is relative to another event (noted) or open-ended.
# --------------------------------------------------------------------------- #

FORMS: Dict[str, Dict[str, object]] = {
    "I":    {"name": "First Accident Report (FAR)", "filed_by": "Investigating Officer", "filed_to": "Claims Tribunal", "deadline_days": 2, "ontology_class": "AccidentReport"},
    "II":   {"name": "Rights of Victims + Flow Chart", "filed_by": "Investigating Officer", "filed_to": "Victim", "deadline_days": 10, "ontology_class": "CourtFiling"},
    "III":  {"name": "Driver's Form", "filed_by": "Driver", "filed_to": "Investigating Officer", "deadline_days": 30, "ontology_class": "CourtFiling"},
    "IV":   {"name": "Owner's Form", "filed_by": "Owner", "filed_to": "Investigating Officer", "deadline_days": 30, "ontology_class": "CourtFiling"},
    "V":    {"name": "Interim Accident Report (IAR)", "filed_by": "Investigating Officer", "filed_to": "Claims Tribunal", "deadline_days": 50, "ontology_class": "AccidentReport"},
    "VIA":  {"name": "Victim's Form", "filed_by": "Victim", "filed_to": "Investigating Officer", "deadline_days": 60, "ontology_class": "CourtFiling"},
    "VII":  {"name": "Detailed Accident Report (DAR)", "filed_by": "Investigating Officer", "filed_to": "Claims Tribunal", "deadline_days": 90, "ontology_class": "AccidentReport"},
    "VIII": {"name": "Site Plan", "filed_by": "Investigating Officer", "filed_to": "Claims Tribunal", "deadline_days": 90, "ontology_class": "Document"},
    "IX":   {"name": "Mechanical Inspection Report", "filed_by": "Investigating Officer", "filed_to": "Claims Tribunal", "deadline_days": 90, "ontology_class": "Document"},
    "X":    {"name": "Verification Report", "filed_by": "Investigating Officer", "filed_to": "Claims Tribunal", "deadline_days": 90, "ontology_class": "Document"},
    "XI":   {"name": "Insurance Form", "filed_by": "Insurance Company", "filed_to": "Claims Tribunal", "deadline_days": None, "note": "within 30 days of DAR", "ontology_class": "CourtFiling"},
    "XII":  {"name": "Victim Impact Report (VIR)", "filed_by": "SLSA", "filed_to": "Criminal Court", "deadline_days": None, "note": "within 30 days of conviction", "ontology_class": "CourtFiling"},
    "XIII": {"name": "Written submissions (death)", "filed_by": "Parties", "filed_to": "Claims Tribunal", "deadline_days": None, "ontology_class": "CourtFiling"},
    "XIV":  {"name": "Written submissions (injury)", "filed_by": "Parties", "filed_to": "Claims Tribunal", "deadline_days": None, "ontology_class": "CourtFiling"},
    "XV":   {"name": "Award summary (death)", "filed_by": "Claims Tribunal", "filed_to": "—", "deadline_days": None, "ontology_class": "CompensationConstruct"},
    "XVI":  {"name": "Award summary (injury)", "filed_by": "Claims Tribunal", "filed_to": "—", "deadline_days": None, "ontology_class": "CompensationConstruct"},
    "XVII": {"name": "Compliance record", "filed_by": "Claims Tribunal", "filed_to": "—", "deadline_days": None, "ontology_class": "CourtFiling"},
    "XVIII":{"name": "Record of awards", "filed_by": "Claims Tribunal", "filed_to": "—", "deadline_days": None, "ontology_class": "CourtFiling"},
    "XIX":  {"name": "Annuity Deposit (MACAD)", "filed_by": "Claims Tribunal", "filed_to": "—", "deadline_days": None, "ontology_class": "CompensationConstruct"},
}

# Map the classifier's MACT_KINDS to a statutory Form key.
KIND_TO_FORM: Dict[str, str] = {
    "First Accident Report (FAR)": "I",
    "Interim Accident Report (IAR)": "V",
    "Detailed Accident Report (DAR)": "VII",
    "Driver's Form": "III",
    "Owner's Form": "IV",
    "Victim's Form": "VIA",
    "Site Plan": "VIII",
    "Mechanical Inspection Report": "IX",
    "Insurance Policy / Form": "XI",
}

# Ontology actor/entity classes → top-level class (for typed graph nodes).
ENTITY_CLASSES: Dict[str, str] = {
    "Deceased": "NaturalPerson",
    "Injured": "NaturalPerson",
    "Claimant": "NaturalPerson",
    "Driver": "NaturalPerson",
    "Owner": "NaturalPerson",
    "Eyewitness": "NaturalPerson",
    "InvestigatingOfficer": "Organization",
    "Insurer": "Organization",
    "Hospital": "Organization",
    "Tribunal": "AdjudicatoryBody",
    "Vehicle": "Object",
}

# Multiplier-doctrine lineage (the reasoning backbone behind the calculator).
PRECEDENTS: List[Dict[str, object]] = [
    {"name": "G.M. Kerala SRTC v. Susamma Thomas", "citation": "(1994) 2 SCC 176", "bench": 3, "holds": "Multiplier method established; max multiplier 16."},
    {"name": "U.P. SRTC v. Trilok Chandra", "citation": "(1996) 4 SCC 362", "bench": 3, "holds": "Max multiplier 18; Second Schedule is a guide only / defective."},
    {"name": "Sarla Verma v. DTC", "citation": "(2009) 6 SCC 121", "bench": 2, "holds": "Standardised age→multiplier table; personal-expense deduction (1/3, 1/4, 1/5)."},
    {"name": "Reshma Kumari v. Madan Mohan", "citation": "(2013) 9 SCC 65", "bench": 3, "holds": "Approves the Sarla Verma table."},
    {"name": "National Insurance Co. v. Pranay Sethi", "citation": "(2017) 16 SCC 680", "bench": 5, "holds": "Future-prospects %; conventional heads (estate/consortium/funeral); Second Schedule redundant."},
]

# Section 166(3) Motor Vehicles Act, 1988 (as amended 2019): limitation period.
LIMITATION_MONTHS = 6

# Timeline milestones measured from the accident (t0). Used by mact_timeline.
# Each references a Form and its statutory deadline; extendable = Scheme cl. 17.
TIMELINE_FORMS = ["I", "V", "VII"]  # FAR, IAR, DAR — the tribunal-facing deadlines.
EXTENDABLE_FORMS = {"V", "VII"}     # IAR / DAR extendable by the Tribunal (cl. 17).


def form_meta(form_key: str) -> Dict[str, object]:
    return FORMS.get(form_key, {})


def form_for_kind(kind: str) -> Optional[str]:
    return KIND_TO_FORM.get(kind)


# --------------------------------------------------------------------------- #
# Typed case-graph enrichment (no Neo4j — mutates a NetworkX graph in place)
# --------------------------------------------------------------------------- #

_ROLE_COLOR_TYPE = "entity"  # rendered red by render_graph_html


def enrich_case_graph(
    graph: "nx.Graph",
    facts: "CaseFacts",
    documents: "List[SourceDocument]",
) -> "nx.Graph":
    """Add ontology types to a per-case graph and return it.

    - Tags each document node with its Form key + ontology class.
    - Adds a central ``CASE`` node linked to every document.
    - Adds typed role nodes from the extracted facts (Deceased/Injured, Driver,
      Owner, Insurer, Vehicle) linked to the Case.

    Operate on a *copy* of the index graph if you don't want to mutate it.
    """
    for doc in documents:
        if doc.doc_id in graph:
            form = form_for_kind(doc.kind)
            if form:
                graph.nodes[doc.doc_id]["ontology_form"] = form
                graph.nodes[doc.doc_id]["ontology_class"] = FORMS[form].get("ontology_class", "Document")

    case_id = "CASE"
    victim_role = "Deceased" if facts.case_type == "death" else "Injured"
    case_label = f"CASE · {facts.name or 'Unknown'} ({victim_role.lower()})"
    graph.add_node(case_id, label=case_label, node_type="document", kind="Case")
    for doc in documents:
        if doc.doc_id in graph:
            graph.add_edge(case_id, doc.doc_id, relation="record")

    typed = case_ontology_rows(facts)
    for row in typed:
        node_id = f"ONT:{row['Role']}"
        graph.add_node(node_id, label=f"{row['Role']}: {row['Value']}", node_type=_ROLE_COLOR_TYPE, kind=row["Class"])
        graph.add_edge(case_id, node_id, relation=row["Role"].lower())
    return graph


def case_ontology_rows(facts: "CaseFacts") -> List[Dict[str, str]]:
    """Typed entities extracted for the case (for the graph + a table)."""
    victim_role = "Deceased" if facts.case_type == "death" else "Injured"
    candidates = [
        (victim_role, facts.name),
        ("Insurer", facts.insurer),
        ("Vehicle", facts.vehicle_number),
    ]
    rows: List[Dict[str, str]] = []
    for role, value in candidates:
        if value:
            rows.append({"Role": role, "Class": ENTITY_CLASSES.get(role, "Thing"), "Value": str(value)})
    return rows
