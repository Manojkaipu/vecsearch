"""Where does a search spend its time? Per-kernel milliseconds from OpenCL event profiling.

    python bench/opencl_profile.py --opencl-device Intel --out results/opencl/intel_arc140v_profile.csv

Same data as gpu_bench.py (random unit vectors, 1M x 384, inner product). For each batch size it
runs the search a few times and reports the wall time per call next to the time the GPU spent in
each kernel, so the rest (queries in, results out, launch gaps) shows as "other".
"""
import argparse
import csv
import os
import statistics
import time

os.environ["VECSEARCH_OPENCL_PROFILE"] = "1"  # must be set before the index is created

import numpy as np  # noqa: E402

import vecsearch as vs  # noqa: E402
from gpu_bench import opencl_device, unit_vectors  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opencl-device", default="0")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batches", default="1,8,16,32,128,512,1024,4096")
    ap.add_argument("--method", default="auto")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xb = unit_vectors(args.n, args.dim, 0)
    batches = [int(b) for b in args.batches.split(",")]
    xq = unit_vectors(max(batches), args.dim, 1)
    idx = vs.OpenCLBruteForceIndex(args.dim, "ip", device=opencl_device(args.opencl_device))
    idx.add(xb)
    print(f"{idx.device_name}, sub-groups: {idx.uses_subgroups}, method {args.method}")

    rows = []
    for b in batches:
        q = xq[:b]
        idx.search(q, args.k, method=args.method)  # warm-up (compiles the kernels)
        idx.take_profile()
        walls, kernels = [], []
        for _ in range(args.repeats):
            t = time.perf_counter()
            idx.search(q, args.k, method=args.method)
            walls.append((time.perf_counter() - t) * 1e3)
            kernels.append(idx.take_profile())
        wall = statistics.median(walls)
        names = sorted({k for p in kernels for k in p})
        med = {k: statistics.median(p.get(k, 0.0) for p in kernels) for k in names}
        row = dict(batch=b, wall_ms=round(wall, 2), **{f"{k}_ms": round(v, 2) for k, v in med.items()},
                   other_ms=round(wall - sum(med.values()), 2))
        rows.append(row)
        print("  batch %5d: wall %8.2f ms | " % (b, wall) + ", ".join(f"{k} {v:.2f}" for k, v in med.items())
              + f" | other {row['other_ms']:.2f}", flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "batch", k != "wall_ms", k == "other_ms", k))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=0)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
