"""Thin wrapper around sentence-transformers: lazy-loads the model once per process
and exposes normalized embeddings for both indexing (batch) and querying (single text)."""
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts as unit-normalized vectors (so cosine similarity == dot product)."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 20)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Goes through the same batch path as indexing so query
    and corpus vectors are always produced identically."""
    return embed_texts([text])[0]
