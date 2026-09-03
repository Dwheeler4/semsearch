import json
import os

import numpy as np

from .ann import HNSWIndex
from .embed import MODEL_NAME
from .ingest import Chunk

INDEX_DIRNAME = ".semsearch_index"


def _index_paths(root: str) -> tuple[str, str, str]:
    index_dir = os.path.join(root, INDEX_DIRNAME)
    return (
        os.path.join(index_dir, "embeddings.npy"),
        os.path.join(index_dir, "metadata.json"),
        os.path.join(index_dir, "hnsw.pkl"),
    )


def find_index_root(start: str = ".") -> str:
    """Walk upward from `start` looking for a .semsearch_index directory, like git does for .git."""
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, INDEX_DIRNAME)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                "No .semsearch_index found in this directory or any parent. Run 'semsearch index <directory>' first."
            )
        current = parent


def save_index(root: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    embeddings_path, metadata_path, ann_path = _index_paths(root)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, embeddings)
    metadata = {
        "model": MODEL_NAME,
        "chunks": [
            {"file": c.file, "start_line": c.start_line, "end_line": c.end_line, "text": c.text}
            for c in chunks
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    ann_index = HNSWIndex()
    for vector in embeddings:
        ann_index.add(vector)
    ann_index.save(ann_path)


def load_index(root: str) -> tuple[list[Chunk], np.ndarray]:
    embeddings_path, metadata_path, _ = _index_paths(root)
    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No index found under {root}. Run 'index' first.")
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chunks = [Chunk(file=c["file"], start_line=c["start_line"], end_line=c["end_line"], text=c["text"]) for c in metadata["chunks"]]
    return chunks, embeddings


def load_ann_index(root: str) -> HNSWIndex | None:
    _, _, ann_path = _index_paths(root)
    if not os.path.exists(ann_path):
        return None
    return HNSWIndex.load(ann_path)


def search(query_embedding: np.ndarray, embeddings: np.ndarray, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
    """Exact brute-force cosine similarity search: O(n) per query, always fully accurate."""
    if len(chunks) == 0:
        return []
    scores = embeddings @ query_embedding
    top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(chunks[i], float(scores[i])) for i in top_indices]


def ann_search(
    query_embedding: np.ndarray,
    ann_index: HNSWIndex,
    chunks: list[Chunk],
    top_k: int = 5,
    ef: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Approximate search via the HNSW graph: sub-linear per query, may occasionally miss a true nearest neighbor.

    `ef` is the search-time beam width: larger trades latency for recall. Defaults
    (None) to the index's own heuristic (max(top_k, ef_construction // 2)).
    """
    results = ann_index.search(query_embedding, top_k=top_k, ef=ef)
    return [(chunks[node_id], 1.0 - distance) for node_id, distance in results]
