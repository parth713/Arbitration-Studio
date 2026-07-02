"""Draft a MACT compensation Award from the computed figures and the record.

Mirrors ``generator.py``. The arithmetic is already fixed by
``mact_compensation`` (deterministic Python); the LLM only writes the narrative
award around those figures and must cite the graph-retrieved record for every
factual assertion. It never recomputes or alters the amounts.
"""

from typing import Dict, List

from openai import OpenAI

from arbitration_studio.graph_rag import Chunk, GraphIndex, retrieve_context
from arbitration_studio.mact_compensation import CaseFacts, Computation


def generate_mact_award(
    index: GraphIndex,
    api_key: str,
    chat_model: str,
    embedding_model: str,
    facts: CaseFacts,
    computation: Computation,
    petitioner: str,
    respondent: str,
    extra_instructions: str = "",
) -> Dict[str, object]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env before generating an award.")

    query = (
        f"motor accident claim compensation award {facts.case_type} negligence liability "
        f"{petitioner} {respondent} income dependency disability insurance {extra_instructions}"
    )
    chunks = retrieve_context(index, query, api_key=api_key, embedding_model=embedding_model, top_k=22)
    context = _format_context(chunks)
    figures = _format_figures(computation)
    form = "Form XV (death case)" if computation.case_type == "death" else "Form XVI (injury case)"

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=chat_model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "You are assisting a Motor Accident Claims Tribunal judge in India to draft a "
                    "compensation award under the Delhi High Court Scheme for Motor Accident Claims. "
                    "The record may be in English, Hindi or Urdu; write the award in ENGLISH, "
                    "transliterating any Hindi/Urdu names to Roman script. "
                    "Use ONLY the provided record context for facts; do not invent names, dates, FIR "
                    "numbers, vehicle or policy particulars, or medical findings. "
                    "Use the COMPUTED FIGURES table verbatim for all amounts — do not recalculate or "
                    "change any number. Every factual assertion must carry a bracketed citation exactly "
                    "as provided, e.g. [C7 | DAR.pdf p. 3]. Where a fact needed for the award is not in "
                    "the record, state '[not on record]' rather than guessing. "
                    "Write a formal, structured award: cause title, brief facts of the accident, issue "
                    "of negligence and liability, finding on income and dependency, a heads-of-compensation "
                    f"table reflecting {form}, total compensation, rate of interest, and apportionment/"
                    "disbursement directions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Case type: {computation.case_type}.\n"
                    f"Petitioner(s): {petitioner or 'Not specified'}\n"
                    f"Respondent(s): {respondent or 'Not specified'}\n"
                    f"Additional instructions: {extra_instructions or 'None'}\n\n"
                    f"COMPUTED FIGURES ({form}) — use these amounts exactly:\n{figures}\n\n"
                    f"Record context:\n{context}"
                ),
            },
        ],
    )
    return {"draft": response.output_text, "chunks": chunks}


def _format_figures(computation: Computation) -> str:
    lines = []
    for item in computation.line_items:
        amount = f"₹{item.amount:,.0f}" if item.amount is not None else "—"
        lines.append(f"- {item.label}: {amount}  ({item.basis})")
    lines.append(f"- TOTAL COMPENSATION: ₹{computation.total:,.0f}")
    return "\n".join(lines)


def _format_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return "No context retrieved from the graph."
    return "\n\n".join(f"[{chunk.citation}]\n{chunk.text}" for chunk in chunks)
