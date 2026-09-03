"""Tests for the from-scratch HNSW index (semsearch/ann.py).

All vectors here are synthetic, so the suite never touches the sentence-transformer
model. The index is approximate, so recall/self-retrieval assertions leave a small
margin rather than demanding exactness.
"""
import numpy as np
import pytest

from semsearch.ann import HNSWIndex


def _unit_rows(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _brute_force_top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    sims = vectors @ query
    return set(np.argsort(-sims)[:k].tolist())


def test_empty_index_returns_no_results():
    index = HNSWIndex(seed=1)
    assert len(index) == 0
    assert index.search(np.zeros(8, dtype=np.float32)) == []


def test_single_element_index_returns_that_element():
    index = HNSWIndex(seed=1)
    vec = _unit_rows(1, 16)[0]
    assert index.add(vec) == 0
    assert len(index) == 1

    results = index.search(vec, top_k=5)
    assert [node_id for node_id, _ in results] == [0]
    assert results[0][1] == pytest.approx(0.0, abs=1e-5)


def test_exact_match_is_top_hit_for_every_stored_vector():
    vectors = _unit_rows(200, 32, seed=2)
    index = HNSWIndex(seed=2)
    for v in vectors:
        index.add(v)

    misses = sum(
        1
        for i, v in enumerate(vectors)
        if (top := index.search(v, top_k=1)) == [] or top[0][0] != i
    )
    assert misses <= 2  # approximate index: essentially always finds a stored vector


def test_recall_against_brute_force_is_high():
    vectors = _unit_rows(400, 32, seed=3)
    index = HNSWIndex(seed=3)
    for v in vectors:
        index.add(v)

    queries = _unit_rows(60, 32, seed=99)
    k = 5
    recall = 0.0
    for q in queries:
        approx = {node_id for node_id, _ in index.search(q, top_k=k)}
        recall += len(approx & _brute_force_top_k(vectors, q, k)) / k
    # With the Algorithm-4 neighbour-selection heuristic this set is ~1.0;
    # the bar guards against regressing to naive "k closest" graph building.
    assert recall / len(queries) >= 0.98


def test_search_returns_sorted_distances_and_respects_top_k():
    index = HNSWIndex(seed=4)
    for v in _unit_rows(120, 16, seed=4):
        index.add(v)

    results = index.search(_unit_rows(1, 16, seed=7)[0], top_k=8)
    assert len(results) == 8
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_neighbor_degree_bounds_are_enforced():
    index = HNSWIndex(m=8, seed=5)
    for v in _unit_rows(300, 24, seed=5):
        index.add(v)

    for node_id in range(len(index)):
        for layer, neighbor_set in enumerate(index.neighbors[node_id]):
            cap = index.m0 if layer == 0 else index.m
            assert len(neighbor_set) <= cap


def test_construction_is_deterministic_for_a_fixed_seed():
    vectors = _unit_rows(150, 16, seed=6)

    def build() -> HNSWIndex:
        idx = HNSWIndex(seed=123)
        for v in vectors:
            idx.add(v)
        return idx

    a, b = build(), build()
    assert a.levels == b.levels
    assert a.entry_point == b.entry_point
    assert a.neighbors == b.neighbors


def test_save_and_load_roundtrip(tmp_path):
    index = HNSWIndex(seed=8)
    for v in _unit_rows(80, 16, seed=8):
        index.add(v)

    path = str(tmp_path / "hnsw.pkl")
    index.save(path)
    reloaded = HNSWIndex.load(path)

    assert len(reloaded) == len(index)
    query = _unit_rows(1, 16, seed=13)[0]
    assert [n for n, _ in reloaded.search(query)] == [n for n, _ in index.search(query)]
