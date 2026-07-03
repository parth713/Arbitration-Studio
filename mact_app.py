"""Standalone MACT Compensation Studio — Streamlit entry point.

Runs only the Motor Accident Claims Tribunal (MACT) flow, sharing the same
render logic as the combined app via `arbitration_studio.mact_ui`. Launch with:

    streamlit run mact_app.py
"""

import streamlit as st

from arbitration_studio.config import get_settings
from arbitration_studio.mact_ui import render_mact


st.set_page_config(page_title="MACT Compensation Studio", page_icon="⚖️", layout="wide")


def main() -> None:
    settings = get_settings()
    has_key = bool(settings.openai_api_key and not settings.openai_api_key.startswith("replace-"))
    has_google_key = bool(settings.google_api_key)

    st.title("MACT Compensation Studio")
    st.caption("Graph-RAG compensation drafting for Motor Accident Claims Tribunal (MACT) cases.")

    with st.sidebar:
        st.header("Configuration")
        st.write("OpenAI API key:", "Configured" if has_key else "Missing")
        st.write("Drafting model:", settings.chat_model)
        st.write("Embedding model:", settings.embedding_model)
        st.divider()
        st.write("Gemini OCR key:", "Configured" if has_google_key else "Missing")
        st.write("OCR model:", settings.ocr_model)

    render_mact(settings, has_key, has_google_key)


if __name__ == "__main__":
    main()
