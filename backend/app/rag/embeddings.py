from __future__ import annotations

import math
import re
import hashlib

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def embed_text_hash(text: str, dim: int = 1536) -> list[float]:
    """
    Deterministic, local embedding (NOT semantic like real embeddings).
    Good enough to validate the RAG plumbing end-to-end.
    """
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vec

    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h, "little") % dim
        vec[idx] += 1.0

    # L2 normalize for cosine similarity
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
