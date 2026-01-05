import os
from qdrant_client import QdrantClient

def get_qdrant_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "127.0.0.1")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)

def get_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "docs")
