"""Split base.npy into N contiguous ranges and build an HNSW index per range.
Workers add their offset to local ids, so merged results use global ids.
"""
import argparse
import json
import os
import time

import numpy as np

import vecsearch as vs


def build(data_dir, shards, out_dir, M=16, efc=200, threads=os.cpu_count()):
    xb = np.load(os.path.join(data_dir, "base.npy"), mmap_mode="r")
    n = len(xb)
    os.makedirs(out_dir, exist_ok=True)
    bounds = np.linspace(0, n, shards + 1, dtype=np.int64)
    manifest = {"dim": int(xb.shape[1]), "n": int(n), "shards": []}
    for i in range(shards):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        path = os.path.join(out_dir, f"shard_{i}.bin")
        if not os.path.exists(path):
            t = time.time()
            idx = vs.HNSWIndex(xb.shape[1], hi - lo, "ip", M=M, ef_construction=efc)
            idx.add(np.ascontiguousarray(xb[lo:hi]), num_threads=threads)
            idx.save(path)
            print(f"shard {i}: {hi - lo:,} vectors built in {time.time() - t:.1f}s")
        manifest["shards"].append({"file": os.path.basename(path), "offset": lo, "n": hi - lo})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def shard_path(manifest_path, shard):
    return os.path.join(os.path.dirname(os.path.abspath(manifest_path)), shard["file"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tweets")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    a = ap.parse_args()
    build(a.data, a.shards, a.out or f"indexes/{a.shards}shards", a.M, a.efc)
