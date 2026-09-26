"""vecsearch — HNSW vector search engine (C++ core, pybind11 bindings)."""
import numpy as np

from ._vecsearch import BM25Index, BruteForceIndex, HNSWIndex

__all__ = ["HNSWIndex", "BruteForceIndex", "BM25Index", "normalize", "recall_at_k", "rrf"]
__version__ = "0.1.0"


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Fraction of the true top-k that appear in the returned top-k, averaged over queries."""
    hits = sum(len(set(f[:k]) & set(t[:k])) for f, t in zip(found, truth))
    return hits / (len(found) * k)


def rrf(rankings, k: int = 10, c: int = 60):
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum 1 / (c + rank).

    rankings: ranked id lists (best first); -1 entries are ignored.
    Returns (ids, scores) for the top k, ties broken by id.
    """
    scores = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            d = int(d)
            if d >= 0:
                scores[d] = scores.get(d, 0.0) + 1.0 / (c + rank)
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [d for d, _ in top], [s for _, s in top]
