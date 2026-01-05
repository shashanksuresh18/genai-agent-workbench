from __future__ import annotations

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1").strip()
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs").strip()
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "COSINE").upper().strip()

def distance_enum() -> qm.Distance:
    if QDRANT_DISTANCE == "DOT":
        return qm.Distance.DOT
    if QDRANT_DISTANCE == "EUCLID":
        return qm.Distance.EUCLID
    return qm.Distance.COSINE

def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection() -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=QDRANT_VECTOR_SIZE, distance=distance_enum()),
    )
