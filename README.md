# Arbitration Studio

A Streamlit app for graph-based arbitration drafting. It ingests PDF/DOCX bundles, identifies Statements of Claim, Statements of Defence, and Rejoinders, builds a citation-aware graph RAG index, visualizes the graph, and drafts the next pleading from the uploaded record only.

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Edit `.env` and set:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

3. Run:

```bash
streamlit run app.py
```

## Flow

- Upload PDFs and DOCX files together.
- The app classifies each file as SOC, SOD, Rejoinder, or supporting material.
- It chunks and embeds all text into a graph RAG index.
- It renders the document/chunk/entity graph.
- It enables:
  - SOC generation when no SOC is present.
  - SOD generation when SOC exists but SOD/Rejoinder do not.
  - Rejoinder generation when SOC and SOD exist.

Generated drafts are instructed to rely only on graph-retrieved material and include bracketed citations such as `[C3 | Contract.pdf p. 4]`.
