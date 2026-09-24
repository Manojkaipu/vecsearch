"""vecsearch — HNSW vector search engine (C++ core, pybind11 bindings)."""
import numpy as np

from ._vecsearch import BruteForceIndex, HNSWIndex

__all__ = ["HNSWIndex", "BruteForceIndex", "normalize", "recall_at_k"]
__version__ = "0.1.0"


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Fraction of the true top-k that appear in the returned top-k, averaged over queries."""
    hits = sum(len(set(f[:k]) & set(t[:k])) for f, t in zip(found, truth))
    return hits / (len(found) * k)
