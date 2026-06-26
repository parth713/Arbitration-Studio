from dataclasses import dataclass, field
import html
import math
import re
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from openai import OpenAI
from pyvis.network import Network

from arbitration_studio.documents import SourceDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str
    citation: str
    embedding: Optional[List[float]] = None
    entities: List[str] = field(default_factory=list)


@dataclass
class GraphIndex:
    graph: nx.Graph
    chunks: List[Chunk]
    documents: List[SourceDocument]


def build_graph_index(
    documents: List[SourceDocument],
    api_key: str,
    embedding_model: str,
    chunk_words: int = 280,
    overlap_words: int = 55,
) -> GraphIndex:
    chunks = _chunk_documents(documents, chunk_words=chunk_words, overlap_words=overlap_words)
    if api_key:
        _embed_chunks(chunks, api_key=api_key, embedding_model=embedding_model)

    graph = nx.Graph()
    for doc in documents:
        graph.add_node(doc.doc_id, label=doc.filename, node_type="document", kind=doc.kind)

    for chunk in chunks:
        graph.add_node(chunk.chunk_id, label=chunk.citation, node_type="chunk", text=chunk.text)
        graph.add_edge(chunk.doc_id, chunk.chunk_id, relation="contains")
        for entity in chunk.entities:
            entity_id = f"E:{entity.lower()}"
            graph.add_node(entity_id, label=entity, node_type="entity")
            graph.add_edge(chunk.chunk_id, entity_id, relation="mentions")

    _connect_similar_chunks(graph, chunks)
    return GraphIndex(graph=graph, chunks=chunks, documents=documents)


def retrieve_context(index: GraphIndex, query: str, api_key: str, embedding_model: str, top_k: int = 18) -> List[Chunk]:
    if not index.chunks:
        return []

    if api_key and all(chunk.embedding for chunk in index.chunks):
        client = OpenAI(api_key=api_key)
        query_vector = client.embeddings.create(model=embedding_model, input=query).data[0].embedding
        ranked = sorted(
            index.chunks,
            key=lambda chunk: _cosine_similarity(query_vector, chunk.embedding or []),
            reverse=True,
        )
        return ranked[:top_k]

    query_terms = set(_normalize_terms(query))
    ranked = sorted(
        index.chunks,
        key=lambda chunk: len(query_terms.intersection(_normalize_terms(chunk.text))),
        reverse=True,
    )
    return ranked[:top_k]


def render_graph_html(index: GraphIndex) -> str:
    net = Network(height="640px", width="100%", bgcolor="#ffffff", font_color="#111827")
    net.barnes_hut(gravity=-24000, central_gravity=0.18, spring_length=130, spring_strength=0.025)

    colors = {
        "document": "#2563eb",
        "chunk": "#059669",
        "entity": "#dc2626",
    }
    sizes = {
        "document": 28,
        "chunk": 14,
        "entity": 12,
    }

    for node_id, attrs in index.graph.nodes(data=True):
        node_type = attrs.get("node_type", "chunk")
        title = html.escape(attrs.get("kind") or attrs.get("text", "")[:600] or attrs.get("label", ""))
        net.add_node(
            node_id,
            label=str(attrs.get("label", node_id))[:55],
            title=title,
            color=colors.get(node_type, "#6b7280"),
            size=sizes.get(node_type, 10),
        )

    for source, target, attrs in index.graph.edges(data=True):
        net.add_edge(source, target, title=attrs.get("relation", "related"), color="#9ca3af")

    with tempfile.NamedTemporaryFile("w+", suffix=".html", delete=False, encoding="utf-8") as handle:
        net.write_html(handle.name, notebook=False)
        handle.seek(0)
        return handle.read()


def graph_stats(index: GraphIndex) -> Dict[str, int]:
    return {
        "Documents": len(index.documents),
        "Chunks": len(index.chunks),
        "Entities": sum(1 for _, attrs in index.graph.nodes(data=True) if attrs.get("node_type") == "entity"),
        "Edges": index.graph.number_of_edges(),
    }


def _chunk_documents(documents: Iterable[SourceDocument], chunk_words: int, overlap_words: int) -> List[Chunk]:
    chunks = []
    counter = 1
    for doc in documents:
        for page in doc.pages:
            words = page.text.split()
            if not words:
                continue
            start = 0
            while start < len(words):
                selected = words[start : start + chunk_words]
                text = " ".join(selected).strip()
                if text:
                    citation = f"C{counter} | {doc.filename} p. {page.page_number}"
                    chunks.append(
                        Chunk(
                            chunk_id=f"C{counter}",
                            doc_id=doc.doc_id,
                            filename=doc.filename,
                            page_number=page.page_number,
                            text=text,
                            citation=citation,
                            entities=_extract_entities(text),
                        )
                    )
                    counter += 1
                if start + chunk_words >= len(words):
                    break
                start += max(1, chunk_words - overlap_words)
    return chunks


def _embed_chunks(chunks: List[Chunk], api_key: str, embedding_model: str) -> None:
    client = OpenAI(api_key=api_key)
    batch_size = 64
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        response = client.embeddings.create(model=embedding_model, input=[chunk.text for chunk in batch])
        for chunk, item in zip(batch, response.data):
            chunk.embedding = item.embedding


def _connect_similar_chunks(graph: nx.Graph, chunks: List[Chunk], threshold: float = 0.79) -> None:
    embedded = [chunk for chunk in chunks if chunk.embedding]
    if not embedded:
        return
    for i, left in enumerate(embedded):
        scores: List[Tuple[float, Chunk]] = []
        for right in embedded[i + 1 :]:
            if left.doc_id == right.doc_id:
                continue
            score = _cosine_similarity(left.embedding or [], right.embedding or [])
            if score >= threshold:
                scores.append((score, right))
        for score, right in sorted(scores, reverse=True)[:3]:
            graph.add_edge(left.chunk_id, right.chunk_id, relation=f"similar {score:.2f}")


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    a = np.array(left)
    b = np.array(right)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or math.isnan(denom):
        return 0.0
    return float(np.dot(a, b) / denom)


def _extract_entities(text: str) -> List[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z&.,'-]+(?:\s+|$)){2,6}", text)
    cleaned = []
    stop = {"Statement Of", "The Claimant", "The Respondent", "This Agreement", "Arbitral Tribunal"}
    for candidate in candidates:
        entity = re.sub(r"\s+", " ", candidate).strip(" .,")
        if len(entity) >= 5 and entity not in stop:
            cleaned.append(entity)
    return list(dict.fromkeys(cleaned))[:12]


def _normalize_terms(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())
