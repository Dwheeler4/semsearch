"""A from-scratch HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index.

Reference: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs" (2018). This is a simplified, single-threaded,
in-memory implementation for cosine similarity over normalized vectors.
"""
import heapq
import math
import pickle
import random

import numpy as np


class HNSWIndex:
    def __init__(self, m: int = 16, ef_construction: int = 200, seed: int | None = None):
        self.m = m  # max neighbors per node, per layer (above layer 0)
        self.m0 = 2 * m  # max neighbors per node at layer 0
        self.ef_construction = ef_construction
        self._level_mult = 1 / math.log(m)
        self._rng = random.Random(seed)

        self.vectors: list[np.ndarray] = []
        self.levels: list[int] = []
        self.neighbors: list[list[set[int]]] = []  # neighbors[node_id][layer] -> neighbor ids
        self.entry_point: int | None = None
        self.max_level = -1

    def __len__(self) -> int:
        return len(self.vectors)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # vectors are unit-normalized, so cosine distance reduces to 1 - dot product
        return 1.0 - float(np.dot(a, b))

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self._level_mult)

    def add(self, vector: np.ndarray) -> int:
        node_id = len(self.vectors)
        level = self._random_level()
        self.vectors.append(vector)
        self.levels.append(level)
        self.neighbors.append([set() for _ in range(level + 1)])

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return node_id

        entry = self.entry_point
        for layer in range(self.max_level, level, -1):
            entry = self._greedy_closest(vector, entry, layer)

        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, entry, self.ef_construction, layer)
            max_conn = self.m0 if layer == 0 else self.m
            selected = self._select_neighbors(vector, candidates, max_conn)
            for neighbor_id in selected:
                self.neighbors[node_id][layer].add(neighbor_id)
                self.neighbors[neighbor_id][layer].add(node_id)
                if len(self.neighbors[neighbor_id][layer]) > max_conn:
                    pruned = self._select_neighbors(
                        self.vectors[neighbor_id], list(self.neighbors[neighbor_id][layer]), max_conn
                    )
                    self.neighbors[neighbor_id][layer] = set(pruned)
            if selected:
                entry = selected[0]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id
        return node_id

    def _greedy_closest(self, vector: np.ndarray, entry: int, layer: int) -> int:
        """Single-best hill-climb toward `vector`, used to descend through upper layers."""
        current = entry
        current_dist = self._distance(vector, self.vectors[current])
        improved = True
        while improved:
            improved = False
            for neighbor_id in self.neighbors[current][layer]:
                d = self._distance(vector, self.vectors[neighbor_id])
                if d < current_dist:
                    current, current_dist = neighbor_id, d
                    improved = True
        return current

    def _search_layer(self, vector: np.ndarray, entry: int, ef: int, layer: int) -> list[int]:
        """Beam search at a single layer, returning up to `ef` nearest node ids found.

        Note: neighbor sets here are small (m/m0 = 16-32 nodes), so batching their distance
        computations into one np.stack() + matmul call was tried and measured *slower* than
        this simple per-neighbor loop -- the stack/copy overhead outweighs the saved dispatch
        calls at this batch size. Reverted; see README benchmark section.
        """
        entry_dist = self._distance(vector, self.vectors[entry])
        visited = {entry}
        candidates = [(entry_dist, entry)]  # min-heap: nodes still worth expanding
        best = [(-entry_dist, entry)]  # max-heap (negated) of the best `ef` found so far

        while candidates:
            dist, current = heapq.heappop(candidates)
            if dist > -best[0][0] and len(best) >= ef:
                break
            for neighbor_id in self.neighbors[current][layer]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                d = self._distance(vector, self.vectors[neighbor_id])
                if len(best) < ef or d < -best[0][0]:
                    heapq.heappush(candidates, (d, neighbor_id))
                    heapq.heappush(best, (-d, neighbor_id))
                    if len(best) > ef:
                        heapq.heappop(best)

        return [node_id for _, node_id in best]

    def _nearest(self, vector: np.ndarray, candidate_ids: list[int], k: int) -> list[int]:
        return sorted(candidate_ids, key=lambda cid: self._distance(vector, self.vectors[cid]))[:k]

    def _select_neighbors(self, vector: np.ndarray, candidate_ids: list[int], k: int) -> list[int]:
        """Pick up to `k` neighbours for `vector` using the paper's heuristic
        (Malkov & Yashunin, Algorithm 4): walking candidates nearest-first, keep
        one only if it is closer to `vector` than to every neighbour already
        kept. This spreads connections across directions instead of piling them
        into the single nearest cluster -- the difference between ~65% and ~95%
        recall@10 on clustered data like text embeddings. Naive "k closest" is
        still used for final result ranking in search(), just not for building
        the graph.
        """
        by_dist = sorted(
            ((self._distance(vector, self.vectors[cid]), cid) for cid in candidate_ids),
            key=lambda t: t[0],
        )
        selected: list[int] = []
        for dist_to_query, cid in by_dist:
            if len(selected) >= k:
                break
            if all(dist_to_query < self._distance(self.vectors[cid], self.vectors[s]) for s in selected):
                selected.append(cid)
        return selected

    def search(self, query: np.ndarray, top_k: int = 5, ef: int | None = None) -> list[tuple[int, float]]:
        if self.entry_point is None:
            return []
        ef = ef or max(top_k, self.ef_construction // 2)
        entry = self.entry_point
        for layer in range(self.max_level, 0, -1):
            entry = self._greedy_closest(query, entry, layer)
        candidates = self._search_layer(query, entry, ef, 0)
        ranked = self._nearest(query, candidates, top_k)
        return [(node_id, self._distance(query, self.vectors[node_id])) for node_id in ranked]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HNSWIndex":
        with open(path, "rb") as f:
            return pickle.load(f)
