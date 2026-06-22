"""
embeddings.py
-------------
Wraps sentence-transformers for embedding chunk text into vectors.

Uses 'all-MiniLM-L6-v2': small (~80MB), fast on CPU, and a solid quality/speed
tradeoff for semantic search over short passages -- a standard default choice
for this kind of project.
"""

import numpy as np

_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def get_model():
    """Lazy-load the embedding model (avoids slow import/download at module
    import time, e.g. when this file is just being inspected or unit tested)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_texts(texts: list) -> np.ndarray:
    """Embed a list of strings -> (n, dim) float32 numpy array, L2-normalized
    so that cosine similarity == dot product (lets us use a plain FAISS
    inner-product index instead of a separate cosine index)."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string -> (1, dim) float32 array."""
    return embed_texts([query])
