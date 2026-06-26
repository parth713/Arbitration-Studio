import streamlit as st
import streamlit.components.v1 as components

from arbitration_studio.config import get_settings
from arbitration_studio.documents import make_source_documents
from arbitration_studio.generator import generate_pleading
from arbitration_studio.graph_rag import build_graph_index, graph_stats, render_graph_html
from arbitration_studio.parties import extract_party_candidates, rows_from_candidates, selected_party


st.set_page_config(page_title="Arbitration Studio", page_icon="⚖️", layout="wide")


def main() -> None:
    settings = get_settings()

    st.title("Arbitration Studio")
    st.caption("Graph RAG drafting for Statements of Claim, Statements of Defence, and Rejoinders.")

    with st.sidebar:
        st.header("Configuration")
        has_key = bool(settings.openai_api_key and not settings.openai_api_key.startswith("replace-"))
        st.write("OpenAI API key:", "Configured" if has_key else "Missing")
        st.write("Drafting model:", settings.chat_model)
        st.write("Embedding model:", settings.embedding_model)

    uploaded_files = st.file_uploader(
        "Upload arbitration PDFs and DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload the record to begin.")
        return

    if st.button("Index documents", type="primary"):
        with st.spinner("Extracting, classifying, embedding, and building the graph..."):
            documents = make_source_documents(uploaded_files)
            index = build_graph_index(
                documents,
                api_key=settings.openai_api_key if has_key else "",
                embedding_model=settings.embedding_model,
            )
            st.session_state["documents"] = documents
            st.session_state["index"] = index
            st.session_state["graph_html"] = render_graph_html(index)
            st.session_state["party_rows"] = rows_from_candidates(extract_party_candidates(documents))

    index = st.session_state.get("index")
    documents = st.session_state.get("documents")
    if not index or not documents:
        return

    kinds = {doc.kind for doc in documents}

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
                "Characters": len(doc.full_text),
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
    components.html(st.session_state["graph_html"], height=680, scrolling=True)

    st.subheader("Parties")
    party_rows = st.session_state.get("party_rows", [])
    if party_rows:
        st.dataframe(
            party_rows,
            width="stretch",
            hide_index=True,
        )
        extracted_party_rows = party_rows
    else:
        st.warning("No claimant/respondent names were extracted from the cause title of the uploaded pleadings.")
        extracted_party_rows = []

    next_pleading = _next_pleading(kinds)
    if not next_pleading:
        st.success("SOC, SOD, and Rejoinder are all present. Upload a different bundle to generate the next pleading.")
        return

    st.subheader(f"Generate {next_pleading}")
    party_names = [str(row["Party"]) for row in extracted_party_rows if row.get("Party")]
    claimant_default = selected_party(extracted_party_rows, "Claimant")
    respondent_default = selected_party(extracted_party_rows, "Respondent")
    claimant_options = _party_options(party_names, claimant_default)
    respondent_options = _party_options(party_names, respondent_default)

    party_cols = st.columns(2)
    claimant = party_cols[0].selectbox("Claimant", claimant_options, index=0)
    respondent = party_cols[1].selectbox("Respondent", respondent_options, index=0)
    if claimant == "Manual entry needed":
        claimant = party_cols[0].text_input("Enter claimant name")
    if respondent == "Manual entry needed":
        respondent = party_cols[1].text_input("Enter respondent name")
    extra = st.text_area("Drafting instructions", placeholder="Optional issues, tribunal style, procedural posture, or relief emphasis.")

    if st.button(f"Generate {next_pleading}", type="primary"):
        with st.spinner(f"Generating {next_pleading} from graph-retrieved context only..."):
            try:
                result = generate_pleading(
                    index=index,
                    api_key=settings.openai_api_key if has_key else "",
                    chat_model=settings.chat_model,
                    embedding_model=settings.embedding_model,
                    pleading_type=next_pleading,
                    claimant=claimant,
                    respondent=respondent,
                    extra_instructions=extra,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.subheader("Draft")
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


def _next_pleading(kinds: set) -> str:
    has_soc = "Statement of Claim" in kinds
    has_sod = "Statement of Defence" in kinds
    has_rejoinder = "Rejoinder" in kinds
    if not has_soc:
        return "Statement of Claim"
    if has_soc and not has_sod and not has_rejoinder:
        return "Statement of Defence"
    if has_soc and has_sod and not has_rejoinder:
        return "Rejoinder"
    return ""


def _party_options(party_names: list, preferred: str) -> list:
    options = []
    if preferred:
        options.append(preferred)
    options.extend(name for name in party_names if name and name not in options)
    if not options:
        options.append("Manual entry needed")
    return options


if __name__ == "__main__":
    main()
