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
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 20)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
