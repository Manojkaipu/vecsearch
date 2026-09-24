"""Exact top-k for every query -> gt.npy.

Uses FAISS IndexFlatIP since BruteForceIndex is single-threaded, then checks the
two agree on a sample.
"""
import argparse
import os
import time

import faiss
import numpy as np

import vecsearch as vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tweets")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    xb = np.load(os.path.join(args.data, "base.npy"))
    xq = np.load(os.path.join(args.data, "queries.npy"))
    t = time.time()
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    _, gt = index.search(xq, args.k)
    print(f"exact search {len(xq):,} x {len(xb):,} in {time.time() - t:.1f}s")
    np.save(os.path.join(args.data, "gt.npy"), gt.astype(np.int64))

    # compare as sets: exact ties can come back in either order
    sample = min(len(xb), 200_000)
    bf = vs.BruteForceIndex(xb.shape[1], "ip")
    bf.add(xb[:sample])
    sub = faiss.IndexFlatIP(xb.shape[1])
    sub.add(xb[:sample])
    _, ref = sub.search(xq[:20], 10)
    mine, _ = bf.search(xq[:20], 10)
    agree = vs.recall_at_k(mine, ref, 10)
    print(f"vecsearch.BruteForce vs FAISS Flat agreement: {agree:.4f}")
    assert agree > 0.99, "BruteForceIndex and IndexFlatIP disagree"


if __name__ == "__main__":
    main()
