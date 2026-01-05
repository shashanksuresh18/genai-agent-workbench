from __future__ import annotations

from uuid import uuid4
from typing import Any

from qdrant_client.http import models as qm

from .store import get_client, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE
from .chunking import chunk_text
from .embeddings import embed_text_hash

def ingest_text(
    *,
    text: str,
    source: str | None = None,
    doc_id: str | None = None,
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    """
    Chunks text, creates deterministic vectors, upserts into Qdrant.
    Returns doc_id and counts.
    """
    client = get_client()
    doc_id = doc_id or str(uuid4())
    source = source or "manual"

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    points: list[qm.PointStruct] = []

    for i, ch in enumerate(chunks):
        pid = str(uuid4())
        vec = embed_text_hash(ch, dim=QDRANT_VECTOR_SIZE)
        payload = {
            "doc_id": doc_id,
            "chunk_id": i,
            "source": source,
            "text": ch,
        }
        points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "points_upserted": len(points),
        "source": source,
    }

def search(
    *,
    query: str,
    top_k: int = 5,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    client = get_client()
    qvec = embed_text_hash(query, dim=QDRANT_VECTOR_SIZE)

    flt = None
    if doc_id:
        flt = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        )

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        query_filter=flt,
    )

    out: list[dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        out.append({
            "score": float(h.score),
            "doc_id": payload.get("doc_id"),
            "chunk_id": payload.get("chunk_id"),
            "source": payload.get("source"),
            "text": payload.get("text"),
        })
    return out
