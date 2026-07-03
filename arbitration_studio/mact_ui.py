"""Shared MACT UI — the Streamlit render logic for the MACT compensation flow.

Used by both the combined app (`app.py`, MACT mode) and the standalone MACT
app (`mact_app.py`), so the flow is defined once and never diverges.
"""

from dataclasses import replace

import streamlit as st
import streamlit.components.v1 as components

from arbitration_studio.graph_rag import GraphIndex, build_graph_index, graph_stats, render_graph_html
from arbitration_studio.mact_documents import (
    detect_case_type,
    make_mact_source_documents,
    missing_documents,
)
from arbitration_studio.mact_parties import extract_mact_parties, mact_party_rows, party_name
from arbitration_studio.mact_compensation import (
    Computation,
    compute_death_award,
    compute_injury_award,
    extract_case_facts,
    field_gaps,
)
from arbitration_studio.mact_generator import generate_mact_award
from arbitration_studio.mact_ontology import case_ontology_json, case_ontology_rows, enrich_case_graph
from arbitration_studio.mact_timeline import check_compliance


def render_mact(settings, has_key: bool, has_google_key: bool) -> None:
    uploaded_files = st.file_uploader(
        "Upload all documents for a single MACT case (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="mact_uploader",
    )

    if not uploaded_files:
        st.info("Upload the case file (FAR, IAR, DAR, FIR, medical and income records, etc.) to begin.")
        return

    enable_ocr = st.checkbox(
        "Use Gemini OCR for scanned PDFs",
        value=has_google_key,
        disabled=not has_google_key,
        help="Scanned FIRs, post-mortems and DARs have no extractable text. "
        "Gemini transcribes those pages. Requires GOOGLE_API_KEY.",
    )
    if enable_ocr and not has_google_key:
        st.warning("GOOGLE_API_KEY is not set — scanned pages cannot be OCR'd and will be read as empty.")

    if st.button("Index documents", type="primary", key="mact_index_btn"):
        with st.spinner("Extracting, OCR'ing scanned pages, classifying, embedding, and building the graph..."):
            documents = make_mact_source_documents(
                uploaded_files,
                google_api_key=settings.google_api_key,
                ocr_model=settings.ocr_model,
                ocr_dpi=settings.ocr_dpi,
                enable_ocr=enable_ocr,
                openai_api_key=settings.openai_api_key if has_key else "",
                chat_model=settings.chat_model,
                enable_llm_classify=has_key,
            )
            index = build_graph_index(
                documents,
                api_key=settings.openai_api_key if has_key else "",
                embedding_model=settings.embedding_model,
            )
            case_type, ct_evidence = detect_case_type(documents)
            st.session_state["mact_documents"] = documents
            st.session_state["mact_index"] = index
            st.session_state["mact_graph_html"] = render_graph_html(index)
            st.session_state["mact_parties"] = extract_mact_parties(documents)
            st.session_state["mact_case_type"] = case_type
            st.session_state["mact_ct_evidence"] = ct_evidence
            # Invalidate any prior extraction/computation for a new bundle.
            st.session_state.pop("mact_facts", None)
            st.session_state.pop("mact_computation", None)

    index = st.session_state.get("mact_index")
    documents = st.session_state.get("mact_documents")
    if not index or not documents:
        return

    st.subheader("Document Classification")
    st.dataframe(
        [
            {
                "ID": doc.doc_id,
                "File": doc.filename,
                "Detected type": doc.kind,
                "Confidence": doc.confidence,
                "Evidence": doc.evidence,
                "Pages": len(doc.pages),
            }
            for doc in documents
        ],
        width="stretch",
        hide_index=True,
    )

    metric_cols = st.columns(4)
    for col, (label, value) in zip(metric_cols, graph_stats(index).items()):
        col.metric(label, value)

    st.subheader("Graph RAG")
    components.html(st.session_state["mact_graph_html"], height=680, scrolling=True)

    parties = st.session_state.get("mact_parties", [])
    if parties:
        st.subheader("Parties")
        st.dataframe(mact_party_rows(parties), width="stretch", hide_index=True)

    st.subheader("Compensation Assessment")
    detected = st.session_state.get("mact_case_type", "unknown")
    st.caption(f"Auto-detected case type: **{detected}** — {st.session_state.get('mact_ct_evidence', '')}")
    options = ["death", "injury"]
    default_index = options.index(detected) if detected in options else 0
    case_type = st.selectbox("Case type", options, index=default_index)

    if st.button("Extract facts & compute", type="primary", key="mact_compute_btn"):
        with st.spinner("Reading facts from the record and computing compensation..."):
            facts = extract_case_facts(
                index=index,
                api_key=settings.openai_api_key if has_key else "",
                chat_model=settings.chat_model,
                embedding_model=settings.embedding_model,
                case_type=case_type,
            )
            computation = (
                compute_death_award(facts) if case_type == "death" else compute_injury_award(facts)
            )
            st.session_state["mact_facts"] = facts
            st.session_state["mact_computation"] = computation
            st.session_state["mact_case_type_used"] = case_type

    facts = st.session_state.get("mact_facts")
    computation = st.session_state.get("mact_computation")
    if not facts or not computation:
        return

    for note in facts.notes:
        st.warning(note)

    _render_extracted_facts(facts)

    if _looks_like_non_record(facts):
        st.info(
            "Very few case facts were found. This file looks like a **judgment, order, "
            "or cause-list**, not a case record. To compute compensation, upload the case "
            "file — DAR/claim petition, income proof, FIR, and medical/disability certificates — "
            "which state the victim's age, income and dependents."
        )

    st.subheader(f"Computation — {'Form XV (death)' if computation.case_type == 'death' else 'Form XVI (injury)'}")
    st.caption("Figures from the deterministic engine (Sarla Verma / Pranay Sethi). Edit any amount to override before drafting the award.")
    edited = _render_editable_computation(computation)

    _render_gaps(documents, facts, computation)
    _render_compliance(facts)
    _render_ontology(index, facts, documents, computation)

    st.subheader("Generate Award")
    petitioner_default = party_name(parties, "Petitioner")
    respondent_default = party_name(parties, "Respondent")
    cols = st.columns(2)
    petitioner = cols[0].text_input("Petitioner(s) / Claimant(s)", value=petitioner_default)
    respondent = cols[1].text_input("Respondent(s)", value=respondent_default)
    extra = st.text_area(
        "Drafting instructions",
        placeholder="Optional: negligence finding, contributory negligence, interest rate, apportionment.",
        key="mact_extra",
    )

    if st.button("Generate award", type="primary", key="mact_award_btn"):
        with st.spinner("Drafting the award from graph-retrieved context and the computed figures..."):
            try:
                result = generate_mact_award(
                    index=index,
                    api_key=settings.openai_api_key if has_key else "",
                    chat_model=settings.chat_model,
                    embedding_model=settings.embedding_model,
                    facts=facts,
                    computation=edited,
                    petitioner=petitioner,
                    respondent=respondent,
                    extra_instructions=extra,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.subheader("Draft Award")
        st.markdown(result["draft"])

        st.subheader("Retrieved Citations")
        st.dataframe(
            [
                {
                    "Citation": chunk.citation,
                    "File": chunk.filename,
                    "Page": chunk.page_number,
                    "Excerpt": chunk.text[:300],
                }
                for chunk in result["chunks"]
            ],
            width="stretch",
            hide_index=True,
        )


def _looks_like_non_record(facts) -> bool:
    """True when none of the core compensation facts were found (likely a
    judgment/order/cause-list rather than an actual case record)."""
    core = [facts.name, facts.age, facts.monthly_income, facts.num_dependents]
    if facts.case_type == "injury":
        core.append(facts.disability_percent or facts.functional_disability_percent)
    return all(v in (None, "") for v in core)


def _render_extracted_facts(facts) -> None:
    fields = [
        ("Name", facts.name, "name"),
        ("Age", facts.age, "age"),
        ("Occupation", facts.occupation, "occupation"),
        ("Employment type", facts.employment_type, "employment_type"),
        ("Monthly income", facts.monthly_income, "monthly_income"),
        ("Marital status", facts.marital_status, "marital_status"),
        ("Dependents", facts.num_dependents, "num_dependents"),
        ("Date of accident", facts.date_of_accident, "date_of_accident"),
        ("Vehicle number", facts.vehicle_number, "vehicle_number"),
        ("Insurer", facts.insurer, "insurer"),
    ]
    if facts.case_type == "injury":
        fields += [
            ("Nature of injury", facts.nature_of_injury, "nature_of_injury"),
            ("Disability %", facts.disability_percent, "disability_percent"),
            ("Functional disability %", facts.functional_disability_percent, "functional_disability_percent"),
        ]

    def _citation(value, key):
        if value in (None, ""):
            return ""
        return facts.sources.get(key) or "⚠ no citation"

    rows = [
        {
            "Fact": label,
            "Value": "—" if value in (None, "") else value,
            "Citation (node · doc · page)": _citation(value, key),
        }
        for label, value, key in fields
    ]
    with st.expander("Facts extracted from the record (with citations)", expanded=True):
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("Each citation is the graph node id + document name + page, e.g. `C7 | DAR.pdf p. 3`.")
        if facts.dependents:
            st.caption("Dependents / legal representatives")
            st.dataframe(facts.dependents, width="stretch", hide_index=True)


def _render_editable_computation(computation: Computation) -> Computation:
    edited_items = []
    running_total = 0.0
    for idx, item in enumerate(computation.line_items):
        cols = st.columns([3, 2, 4])
        cols[0].markdown(f"**{item.label}**")
        if item.editable:
            value = float(item.amount) if item.amount is not None else 0.0
            new_amount = cols[1].number_input(
                item.label,
                value=value,
                step=1000.0,
                format="%.0f",
                key=f"mact_li_{idx}",
                label_visibility="collapsed",
            )
            edited_items.append(replace(item, amount=new_amount))
        else:
            display = f"{item.amount:,.0f}" if item.amount is not None else "—"
            cols[1].markdown(display)
            edited_items.append(item)
        cols[2].caption(item.basis)
        if edited_items[-1].in_total and edited_items[-1].amount is not None:
            running_total += edited_items[-1].amount

    st.metric("Total compensation (₹)", f"{running_total:,.0f}")
    return Computation(
        case_type=computation.case_type,
        line_items=edited_items,
        total=round(running_total),
        summary=computation.summary,
        missing_fields=computation.missing_fields,
    )


def _render_gaps(documents, facts, computation: Computation) -> None:
    missing_docs = missing_documents(documents, computation.case_type)
    gaps = field_gaps(facts)
    if not missing_docs and not gaps:
        st.success("All expected documents and key facts are present in the record.")
        return
    with st.expander("Missing documents & facts (record gaps)", expanded=True):
        if missing_docs:
            st.markdown("**Expected documents not found:**")
            for doc in missing_docs:
                st.markdown(f"- {doc}")
        if gaps:
            st.markdown("**Compensation facts not on record:**")
            for gap in gaps:
                st.markdown(f"- {gap}")


def _render_compliance(facts) -> None:
    report = check_compliance(facts)
    st.subheader("Statutory Timeline & Limitation")
    if report.accident_date is None:
        st.info(report.limitation_note)
        return

    verdict = f"**Limitation — Section 166(3):** {report.limitation_status}. {report.limitation_note}"
    if report.limitation_status == "Satisfied":
        st.success(verdict)
    elif report.limitation_status == "At risk":
        st.error(verdict)
    else:
        st.warning(verdict)

    st.dataframe(
        [
            {
                "Form / event": item.label,
                "Due by": item.deadline.isoformat() if item.deadline else "—",
                "Actual": item.actual.isoformat() if item.actual else "—",
                "Status": item.status,
                "Note": item.note,
            }
            for item in report.items
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Accident date (t₀): {report.accident_date.isoformat()}. Deadlines per Delhi HC Scheme; extensions under cl. 17.")


def _render_ontology(index, facts, documents, computation) -> None:
    with st.expander("Typed case graph (ontology)", expanded=False):
        rows = case_ontology_rows(facts)

        st.markdown("**Extracted ontology — JSON**")
        st.json(case_ontology_json(facts, documents, computation))

        if rows:
            st.markdown("**Typed entities (with citations)**")
            st.dataframe(rows, width="stretch", hide_index=True)

        st.markdown("**Typed case graph**")
        st.caption("CASE node → documents (tagged with their statutory Form) and typed actor nodes; hover a node for its citation.")
        # Enrich a copy so the base index graph is left untouched across reruns.
        enriched = enrich_case_graph(index.graph.copy(), facts, documents)
        html = render_graph_html(
            GraphIndex(graph=enriched, chunks=index.chunks, documents=index.documents)
        )
        components.html(html, height=560, scrolling=True)
