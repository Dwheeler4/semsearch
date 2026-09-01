import argparse
import os
import random
import textwrap
import time

import numpy as np

from .ann import HNSWIndex
from .embed import embed_query, embed_texts
from .ingest import Chunk, ingest_directory
from .store import ann_search, find_index_root, load_ann_index, load_index, save_index, search


def cmd_index(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.directory)
    print(f"Scanning {root} ...")
    chunks = ingest_directory(root)
    if not chunks:
        print("No supported files found (looked for .md, .txt, .py, .cpp, .c, .h, .hpp, .rst, .org).")
        return
    print(f"Found {len(chunks)} chunks across the directory. Embedding...")
    embeddings = embed_texts([c.text for c in chunks])
    save_index(root, chunks, embeddings)
    print(f"Index saved to {os.path.join(root, '.semsearch_index')}")


def _print_results(
    query: str,
    root: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    ann_index: HNSWIndex | None,
    top_k: int,
    exact: bool,
) -> None:
    query_embedding = embed_query(query)
    if exact or ann_index is None:
        results = search(query_embedding, embeddings, chunks, top_k=top_k)
    else:
        results = ann_search(query_embedding, ann_index, chunks, top_k=top_k)
    if not results:
        print("No results.")
        return
    for chunk, score in results:
        rel_path = os.path.relpath(chunk.file, root)
        print(f"\n[{score:.3f}] {rel_path}:{chunk.start_line}-{chunk.end_line}")
        snippet = chunk.text.strip()
        print(textwrap.indent(textwrap.shorten(snippet, width=300, placeholder=" ..."), "    "))


def cmd_search(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.directory) if args.directory else find_index_root(".")
    chunks, embeddings = load_index(root)
    ann_index = load_ann_index(root)

    if args.query:
        _print_results(args.query, root, chunks, embeddings, ann_index, args.top_k, args.exact)
        return

    mode = "exact brute-force" if args.exact or ann_index is None else "approximate (HNSW)"
    print(f"Searching index at {root} ({len(chunks)} chunks, {mode}). Type a query, or 'q' to quit.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"q", "quit", "exit"}:
            break
        _print_results(query, root, chunks, embeddings, ann_index, args.top_k, args.exact)


def cmd_benchmark(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.directory) if args.directory else find_index_root(".")
    chunks, embeddings = load_index(root)
    ann_index = load_ann_index(root)
    if ann_index is None:
        print("No ANN index found for this directory. Re-run 'semsearch index' to build one.")
        return

    sample_size = min(args.samples, len(chunks))
    sample_indices = random.Random(42).sample(range(len(chunks)), sample_size)
    rng = np.random.default_rng(42)

    exact_times, ann_times, recalls = [], [], []
    for idx in sample_indices:
        # Perturb the sampled chunk's embedding rather than reusing it verbatim as the query.
        # A verbatim embedding is a graph node itself, so both exact and approximate search
        # trivially find it as their own top hit -- that inflates recall to ~100% regardless
        # of how good the index actually is. A small amount of noise (then re-normalized,
        # since embeddings are unit vectors) approximates what a real query looks like: close
        # to a chunk's meaning, but not identical to it.
        query_embedding = embeddings[idx] + rng.standard_normal(embeddings.shape[1]) * 0.35
        query_embedding = (query_embedding / np.linalg.norm(query_embedding)).astype(embeddings.dtype)

        start = time.perf_counter()
        exact_results = search(query_embedding, embeddings, chunks, top_k=args.top_k)
        exact_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        approx_results = ann_search(query_embedding, ann_index, chunks, top_k=args.top_k)
        ann_times.append(time.perf_counter() - start)

        exact_ids = {id(c) for c, _ in exact_results}
        approx_ids = {id(c) for c, _ in approx_results}
        recalls.append(len(exact_ids & approx_ids) / args.top_k)

    print(f"Benchmark: {sample_size} queries, top-{args.top_k}, corpus size {len(chunks)} chunks")
    print(f"  Exact (brute-force):  {sum(exact_times) / sample_size * 1000:.3f} ms/query avg")
    print(f"  Approx (HNSW):        {sum(ann_times) / sample_size * 1000:.3f} ms/query avg")
    print(f"  Recall@{args.top_k}:          {sum(recalls) / sample_size:.1%} (fraction of exact top-{args.top_k} that HNSW also found)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="semsearch", description="Semantic search over your notes and code.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build a semantic index for a directory")
    index_parser.add_argument("directory", nargs="?", default=".", help="Directory to index (default: current directory)")
    index_parser.set_defaults(func=cmd_index)

    search_parser = subparsers.add_parser("search", help="Search a previously built index (omit query for interactive mode)")
    search_parser.add_argument("query", nargs="?", default=None, help="Natural-language search query (omit to enter interactive mode)")
    search_parser.add_argument("directory", nargs="?", default=None, help="Directory whose index to search (default: auto-detect from current directory upward)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    search_parser.add_argument("--exact", action="store_true", help="Use exact brute-force search instead of the approximate HNSW index")
    search_parser.set_defaults(func=cmd_search)

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare approximate (HNSW) vs exact search speed and recall")
    benchmark_parser.add_argument("directory", nargs="?", default=None, help="Directory whose index to benchmark (default: auto-detect)")
    benchmark_parser.add_argument("--top-k", type=int, default=5, help="Number of results to compare per query (default: 5)")
    benchmark_parser.add_argument("--samples", type=int, default=20, help="Number of sample queries to run (default: 20)")
    benchmark_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
