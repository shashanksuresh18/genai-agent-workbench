from qdrant_client.http import models as qm
from .qdrant_client import get_qdrant_client, get_collection_name

def ensure_collection(vector_size: int = 1536) -> None:
    """
    1536 is common for many OpenAI/Azure embedding models.
    We'll confirm the exact size once we pick the embedding deployment.
    """
    client = get_qdrant_client()
    name = get_collection_name()

    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )
