# semsearch

A command-line tool for semantic search over local notes and code. It chunks files, embeds
them with a sentence-transformer model, and indexes the embeddings with a from-scratch
implementation of HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor
search, based on Malkov & Yashunin's original paper.

## Usage

```
semsearch index [directory]              # build an index (defaults to cwd)
semsearch search "how does auth work"    # query it (auto-discovers the nearest index, like git)
semsearch search                         # interactive REPL mode
semsearch search --exact "..."           # brute-force instead of the HNSW index
semsearch benchmark                      # compare HNSW vs brute-force speed and recall
```

The index lives in a `.semsearch_index/` directory, auto-discovered by walking upward from
the current directory the same way `git` finds `.git`.

## Architecture

- `ingest.py` — walks a directory, chunks text files (800 chars/chunk, 150 char overlap) into
  `Chunk` objects carrying file path and line range.
- `embed.py` — wraps `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) to embed chunks and
  queries as unit-normalized vectors.
- `ann.py` — the HNSW index itself: layered graph construction, greedy descent through upper
  layers, beam search at layer 0.
- `store.py` — persistence (`.npy`/`json`/`pickle`) and the two search paths: exact brute-force
  cosine similarity and approximate HNSW search.
- `cli.py` — argparse-based CLI wiring the above together, plus `benchmark`.

## Benchmark: HNSW vs. brute force

`semsearch benchmark` compares the from-scratch HNSW index against exact brute-force search on
the currently-indexed corpus, reporting query latency and recall@k. The numbers below are from
a standalone measurement (`m=16`, `ef_construction=200`, 384-dim unit vectors, top-5, 200
queries per corpus size, each query a stored vector perturbed with Gaussian noise and
re-normalized rather than reused verbatim — see "methodology fix" below):

| Corpus size | HNSW build time | Exact search | HNSW search | Recall@5 |
|---|---|---|---|---|
| 1,000   | 1.1s  | 0.03 ms/query | 0.76 ms/query | 99.7% |
| 5,000   | 9.4s  | 0.08 ms/query | 1.64 ms/query | 81.6% |
| 20,000  | 66.3s | 0.49 ms/query | 2.71 ms/query | 46.9% |

Two things worth being upfront about, because they're the actual finding, not a bug being
glossed over:

**HNSW is slower than brute force here, at every size tested.** Brute-force search is a single
vectorized `embeddings @ query` matrix-vector product, handled by BLAS in one call. This HNSW
implementation is pure Python: each query walks the graph doing per-neighbor `np.dot()` calls,
heap pushes, and set lookups in a Python loop. At these corpus sizes (up to 20k vectors, 384
dims), the constant-factor overhead of that loop outweighs HNSW's sub-linear comparison count.
Textbook HNSW asymptotics assume comparisons dominate cost; here, Python/NumPy dispatch
overhead does. Tried batching each node's neighbor-set distance calls into one
`np.stack()` + matmul to cut that overhead (see git history) — measured build time got
*worse* (66s → 75s at 20k), because the neighbor sets are small (16-32 nodes) and the stack/copy
cost outweighs the saved call overhead at that batch size. Reverted rather than shipped, since
it made the code more complex for a measured loss. At a large enough corpus this would likely
cross over (that's the whole premise of HNSW), but it didn't within the sizes tested here — a
real, measured property of this implementation, not an assumed one.

**Recall degrades as the corpus grows**, because the search-time beam width (`ef`, defaulting
to 100) doesn't scale with corpus size. Beam-searching 100 candidates out of 1,000 nodes covers
the graph far more thoroughly than 100 out of 20,000. This is a known, expected HNSW trade-off
— recall could be recovered by scaling `ef` with corpus size, at the cost of extra query
latency — not "fixed" here since demonstrating the trade-off honestly, rather than picking one
operating point and hiding the rest of the curve, was the point of measuring it at multiple
sizes in the first place.

**Methodology fix:** the benchmark originally sampled a stored embedding and searched with it
verbatim — which is a graph node searching for itself, guaranteeing a trivial top-1 hit and
inflating recall regardless of index quality. Fixed to perturb the sampled vector with Gaussian
noise before re-normalizing, so the query resembles-but-isn't a stored chunk, closer to what an
actual text query looks like.

## Known limitations

- The HNSW implementation is single-threaded, in-memory, and not incrementally deletable
  (no node removal, only insertion) — fine for a local CLI re-indexing a directory, not
  intended as a production vector store.
- No tests currently cover `ann.py`'s graph construction directly (recall is checked
  empirically via the benchmark above, not asserted in a test suite).
- Chunking is purely character-count-based (800 chars, 150 overlap) with no awareness of code
  structure (functions, classes) or markdown structure (headings) — a chunk can split a
  function mid-body.
