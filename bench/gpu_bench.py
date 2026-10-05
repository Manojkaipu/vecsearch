"""Exact (brute-force) k-NN throughput: CPU vs CUDA vs FAISS, across batch sizes.

    python bench/gpu_bench.py --backends cpu,cuda,faiss-gpu --out results/gpu/colab_t4.csv
    python bench/gpu_bench.py --backends cpu,faiss-cpu --out results/gpu/laptop.csv

Brute-force cost doesn't depend on the values, so the data is random unit vectors with the
tweet benchmark's shape (1M x 384, cosine). Every timed search includes copying the queries
in and the results out; the data itself is already loaded (on the GPU, for GPU backends).
Each backend's results on the first --check queries are compared with the CPU index.
"""
import argparse
import csv
import os
import platform
import subprocess
import time

import numpy as np

import vecsearch as vs

BACKENDS = ["cpu", "cuda", "cuda-naive", "cuda-skinny", "cuda-tiled", "faiss-gpu", "faiss-cpu"]


def unit_vectors(n, d, seed):
    rng = np.random.default_rng(seed)
    x = np.empty((n, d), np.float32)
    for i in range(0, n, 100_000):  # in slices, to keep float64 temporaries small
        x[i:i + 100_000] = rng.standard_normal((min(100_000, n - i), d), dtype=np.float32)
    return vs.normalize(x)


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "gpu"


def make_backend(name, xb, metric, threads):
    """Returns (search(q, k) -> ids, device description)."""
    d = xb.shape[1]
    if name == "cpu":
        idx = vs.BruteForceIndex(d, metric)
        idx.add(xb)
        return (lambda q, k: idx.search(q, k, num_threads=threads)[0]), f"{cpu_name()}, {threads} threads"
    if name.startswith("cuda"):
        idx = vs.GpuBruteForceIndex(d, metric)
        idx.add(xb)
        method = name.split("-", 1)[1] if "-" in name else "auto"
        return (lambda q, k: idx.search(q, k, method=method)[0]), gpu_name()
    import faiss
    flat = faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)
    if name == "faiss-gpu":
        res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(res, 0, flat)
        idx.add(xb)
        return (lambda q, k, res=res: idx.search(q, k)[1]), gpu_name()  # res must outlive idx
    faiss.omp_set_num_threads(threads)
    flat.add(xb)
    return (lambda q, k: flat.search(q, k)[1]), f"{cpu_name()}, {threads} threads"


def time_batch(search, q, k, min_time):
    """Median seconds per call: at least 3 calls and min_time seconds, unless one call is longer."""
    search(q, k)  # warm-up (allocations, caches)
    times = []
    while len(times) < 3 or sum(times) < min_time:
        t = time.perf_counter()
        search(q, k)
        times.append(time.perf_counter() - t)
        if times[0] > min_time:
            break
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="cpu,cuda,faiss-gpu", help=f"comma list of {BACKENDS}")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batches", default="1,8,32,128,512,1024,4096")
    ap.add_argument("--max-batch", default="cpu=1024,faiss-cpu=1024,cuda-naive=1024",
                    help="largest batch per backend (slow ones), e.g. cpu=1024")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--min-time", type=float, default=1.0)
    ap.add_argument("--check", type=int, default=128, help="queries compared with the CPU index")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    backends = args.backends.split(",")
    unknown = set(backends) - set(BACKENDS)
    if unknown:
        ap.error(f"unknown backends {sorted(unknown)}")
    batches = [int(b) for b in args.batches.split(",")]
    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.max_batch.split(",") if kv}

    print(f"data: {args.n:,} x {args.dim} ({args.metric}), queries: {max(batches):,}", flush=True)
    xb = unit_vectors(args.n, args.dim, 0)
    xq = unit_vectors(max(max(batches), args.check), args.dim, 1)

    cpu = vs.BruteForceIndex(args.dim, args.metric)
    cpu.add(xb)
    truth = cpu.search(xq[:args.check], args.k, num_threads=args.threads)[0]
    del cpu

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["backend", "device", "n", "dim", "metric", "k", "batch", "ms_per_batch", "qps", "recall_vs_cpu"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in backends:
            search, device = make_backend(name, xb, args.metric, args.threads)
            recall = vs.recall_at_k(search(xq[:args.check], args.k), truth, args.k)
            print(f"{name} on {device}: recall vs cpu {recall:.4f}", flush=True)
            for b in batches:
                if b > caps.get(name, b):
                    continue
                sec = time_batch(search, xq[:b], args.k, args.min_time)
                row = dict(backend=name, device=device, n=args.n, dim=args.dim, metric=args.metric, k=args.k,
                           batch=b, ms_per_batch=round(sec * 1e3, 3), qps=round(b / sec, 1),
                           recall_vs_cpu=round(recall, 4))
                w.writerow(row)
                f.flush()
                print(f"  batch {b:>5}: {sec * 1e3:10.2f} ms  {b / sec:12,.0f} QPS", flush=True)
            del search
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
