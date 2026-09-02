"""Tests for the two search paths and index discovery (semsearch/store.py)."""
import os

import numpy as np
import pytest

from semsearch.ann import HNSWIndex
from semsearch.ingest import Chunk
from semsearch.store import ann_search, find_index_root, search


def _chunk(i: int) -> Chunk:
    return Chunk(file=f"f{i}.py", start_line=1, end_line=2, text=f"chunk {i}")


def test_brute_force_search_ranks_exact_match_first():
    embeddings = np.eye(4, dtype=np.float32)  # orthonormal, so match score is 1.0
    chunks = [_chunk(i) for i in range(4)]

    results = search(embeddings[2], embeddings, chunks, top_k=3)
    assert results[0][0] is chunks[2]
    assert results[0][1] == pytest.approx(1.0)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_brute_force_search_on_empty_corpus():
    assert search(np.zeros(4, dtype=np.float32), np.empty((0, 4), dtype=np.float32), []) == []


def test_ann_search_returns_chunks_and_similarity_scores():
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((50, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = [_chunk(i) for i in range(50)]

    index = HNSWIndex(seed=0)
    for v in embeddings:
        index.add(v)

    results = ann_search(embeddings[10], index, chunks, top_k=5)
    assert results[0][0] is chunks[10]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(-1.000001 <= s <= 1.000001 for _, s in results)


def test_find_index_root_walks_upward(tmp_path):
    (tmp_path / ".semsearch_index").mkdir()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    found = find_index_root(str(nested))
    assert os.path.realpath(found) == os.path.realpath(str(tmp_path))


def test_find_index_root_raises_when_absent(tmp_path):
    nested = tmp_path / "x"
    nested.mkdir()
    with pytest.raises(FileNotFoundError):
        find_index_root(str(nested))
