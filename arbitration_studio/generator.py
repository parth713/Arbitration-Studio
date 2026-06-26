from typing import Dict, List

from openai import OpenAI

from arbitration_studio.graph_rag import Chunk, GraphIndex, retrieve_context


PLEADING_GUIDANCE = {
    "Statement of Claim": "Draft a Statement of Claim for the claimant.",
    "Statement of Defence": "Draft a Statement of Defence for the respondent, responding to the Statement of Claim and supporting record.",
    "Rejoinder": "Draft a Rejoinder for the claimant, responding to the Statement of Defence using the uploaded record.",
}


def generate_pleading(
    index: GraphIndex,
    api_key: str,
    chat_model: str,
    embedding_model: str,
    pleading_type: str,
    claimant: str,
    respondent: str,
    extra_instructions: str = "",
) -> Dict[str, object]:
    query = _query_for(pleading_type, claimant, respondent, extra_instructions)
    chunks = retrieve_context(index, query, api_key=api_key, embedding_model=embedding_model, top_k=22)
    context = _format_context(chunks)

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env before generating a pleading.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=chat_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an arbitration drafting assistant. Use only the provided graph RAG context. "
                    "Do not invent facts, dates, clauses, amounts, authorities, or procedural history. "
                    "Every material factual assertion must include one or more bracketed citations exactly as provided, "
                    "for example [C7 | Agreement.pdf p. 3]. If the context is insufficient, say so in the relevant section. "
                    "Draft in formal arbitration style with clear headings, numbered paragraphs, and a prayer for relief where appropriate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{PLEADING_GUIDANCE[pleading_type]}\n\n"
                    f"Claimant: {claimant or 'Not specified'}\n"
                    f"Respondent: {respondent or 'Not specified'}\n"
                    f"Additional instructions: {extra_instructions or 'None'}\n\n"
                    "Graph RAG context:\n"
                    f"{context}"
                ),
            },
        ],
    )
    return {"draft": response.output_text, "chunks": chunks}


def _query_for(pleading_type: str, claimant: str, respondent: str, extra_instructions: str) -> str:
    return " ".join(
        [
            pleading_type,
            claimant,
            respondent,
            "contract obligations breach notices invoices payments damages relief arbitration jurisdiction",
            extra_instructions,
        ]
    )


def _format_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return "No context retrieved from the graph."
    return "\n\n".join(f"[{chunk.citation}]\n{chunk.text}" for chunk in chunks)
