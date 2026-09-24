"""Recall@k vs QPS for vecsearch, hnswlib, FAISS (HNSW, IVF) and ScaNN.

ann-benchmarks rules: build on all cores, query on one thread, best of --runs.
The HNSW libraries share M and ef_construction. Writes results/<name>/results.csv.
"""
import argparse
import csv
import os
import shutil
import tempfile
import time

import numpy as np

import vecsearch as vs


def dir_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 2**20


class VecsearchHNSW:
    name = "vecsearch-hnsw"

    def __init__(self, M=16, efc=200):
        self.M, self.efc = M, efc
        self.params = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        self.label = f"M={M},efC={efc}"

    def build(self, xb, threads):
        self.idx = vs.HNSWIndex(xb.shape[1], len(xb), "ip", M=self.M, ef_construction=self.efc)
        self.idx.add(xb, num_threads=threads)

    def search(self, xq, k, ef):
        return self.idx.search(xq, k=k, ef=ef, num_threads=1)[0]

    def size_mb(self, tmp):
        p = os.path.join(tmp, "v.bin")
        self.idx.save(p)
        return dir_size_mb(p)


class HnswlibHNSW(VecsearchHNSW):
    name = "hnswlib"

    def build(self, xb, threads):
        import hnswlib

        self.idx = hnswlib.Index(space="ip", dim=xb.shape[1])
        self.idx.init_index(max_elements=len(xb), M=self.M, ef_construction=self.efc, random_seed=42)
        self.idx.add_items(xb, num_threads=threads)
        self.idx.set_num_threads(1)

    def search(self, xq, k, ef):
        self.idx.set_ef(max(ef, k))
        return self.idx.knn_query(xq, k=k, num_threads=1)[0]

    def size_mb(self, tmp):
        p = os.path.join(tmp, "h.bin")
        self.idx.save_index(p)
        return dir_size_mb(p)


class FaissHNSW(VecsearchHNSW):
    name = "faiss-hnsw"

    def build(self, xb, threads):
        import faiss

        faiss.omp_set_num_threads(threads)
        self.idx = faiss.IndexHNSWFlat(xb.shape[1], self.M, faiss.METRIC_INNER_PRODUCT)
        self.idx.hnsw.efConstruction = self.efc
        self.idx.add(xb)
        faiss.omp_set_num_threads(1)

    def search(self, xq, k, ef):
        self.idx.hnsw.efSearch = max(ef, k)
        return self.idx.search(xq, k)[1]

    def size_mb(self, tmp):
        import faiss

        p = os.path.join(tmp, "f.bin")
        faiss.write_index(self.idx, p)
        return dir_size_mb(p)


class FaissIVF:
    name = "faiss-ivfflat"

    def __init__(self, nlist=None):
        self.nlist = nlist
        self.params = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    def build(self, xb, threads):
        import faiss

        faiss.omp_set_num_threads(threads)
        self.nlist = self.nlist or int(4 * np.sqrt(len(xb)))
        self.label = f"nlist={self.nlist}"
        q = faiss.IndexFlatIP(xb.shape[1])
        self.idx = faiss.IndexIVFFlat(q, xb.shape[1], self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.idx.train(xb[np.random.default_rng(0).choice(len(xb), min(len(xb), 50 * self.nlist), replace=False)])
        self.idx.add(xb)
        faiss.omp_set_num_threads(1)

    def search(self, xq, k, nprobe):
        self.idx.nprobe = nprobe
        return self.idx.search(xq, k)[1]

    def size_mb(self, tmp):
        return FaissHNSW.size_mb(self, tmp)


class ScaNN:
    """Tree partitioning + anisotropic quantization + exact re-rank.
    Param is (leaves_to_search, pre_reorder_num_neighbors)."""

    name = "scann"

    def __init__(self, num_leaves=None):
        self.num_leaves = num_leaves
        self.params = [(1, 50), (2, 50), (4, 100), (8, 100), (16, 100), (32, 150), (64, 200),
                       (128, 250), (256, 400), (512, 800)]

    def build(self, xb, threads):
        import scann

        n = len(xb)
        self.num_leaves = self.num_leaves or max(16, int(np.sqrt(n)))
        self.label = f"leaves={self.num_leaves},AH2,reorder"
        b = scann.scann_ops_pybind.builder(xb, 10, "dot_product")
        b = b.tree(num_leaves=self.num_leaves, num_leaves_to_search=32,
                   training_sample_size=min(n, 250_000))
        b = b.score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100)
        if hasattr(b, "set_n_training_threads"):
            b = b.set_n_training_threads(threads)
        self.idx = b.build()
        self.params = [p for p in self.params if p[0] <= self.num_leaves]

    def search(self, xq, k, p):
        leaves, reorder = p
        return self.idx.search_batched(xq, final_num_neighbors=k, leaves_to_search=leaves,
                                       pre_reorder_num_neighbors=max(reorder, k))[0]

    def size_mb(self, tmp):
        d = os.path.join(tmp, "scann")
        os.makedirs(d, exist_ok=True)
        self.idx.serialize(d)
        return dir_size_mb(d)


ALGOS = {
    "vecsearch": lambda: VecsearchHNSW(),
    "hnswlib": lambda: HnswlibHNSW(),
    "faiss-hnsw": lambda: FaissHNSW(),
    "faiss-ivf": lambda: FaissIVF(),
    "scann": lambda: ScaNN(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tweets")
    ap.add_argument("--name", default="tweets")
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    xb = np.load(os.path.join(args.data, "base.npy"))
    xq = np.load(os.path.join(args.data, "queries.npy"))[: args.n_queries]
    gt = np.load(os.path.join(args.data, "gt.npy"))[: args.n_queries]
    print(f"base {xb.shape}  queries {xq.shape}  build threads={args.threads}")

    out_dir = os.path.join("results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")
    rows = []
    for key in args.algos.split(","):
        algo = ALGOS[key]()
        t = time.perf_counter()
        algo.build(xb, args.threads)
        build_s = time.perf_counter() - t
        tmp = tempfile.mkdtemp()
        size = algo.size_mb(tmp)
        shutil.rmtree(tmp)
        print(f"\n[{algo.name}] {algo.label}  build {build_s:.1f}s  size {size:.0f} MB")
        for p in algo.params:
            algo.search(xq[:100], args.k, p)  # warm-up
            best = float("inf")
            for _ in range(args.runs):
                t = time.perf_counter()
                ids = algo.search(xq, args.k, p)
                best = min(best, time.perf_counter() - t)
            rec = vs.recall_at_k(np.asarray(ids), gt, args.k)
            qps = len(xq) / best
            print(f"  param={str(p):>10}  recall@{args.k}={rec:.4f}  QPS={qps:>9.0f}")
            rows.append(dict(algo=algo.name, config=algo.label, param=str(p), recall=rec, qps=qps,
                             build_s=build_s, index_mb=size, n=len(xb), dim=xb.shape[1]))
            with open(out_csv, "w", newline="") as f:  # rewritten per point so a crash keeps partial results
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        del algo
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
