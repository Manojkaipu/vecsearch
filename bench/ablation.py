"""vecsearch ablations on a --subset of the base vectors:
neighbor-selection heuristic on/off, M in {8, 16, 32}, ef_construction in {50, 100, 200, 400}.
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vecsearch as vs

EFS = [10, 16, 32, 64, 128, 256]


def curve(xb, xq, gt, threads, **kw):
    t = time.perf_counter()
    idx = vs.HNSWIndex(xb.shape[1], len(xb), "ip", **kw)
    idx.add(xb, num_threads=threads)
    build = time.perf_counter() - t
    pts = []
    for ef in EFS:
        t = time.perf_counter()
        ids, _ = idx.search(xq, 10, ef)
        pts.append((vs.recall_at_k(ids, gt), len(xq) / (time.perf_counter() - t)))
    return build, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tweets")
    ap.add_argument("--subset", type=int, default=200_000)
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--name", default="tweets")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()

    import faiss

    xb = np.ascontiguousarray(np.load(os.path.join(a.data, "base.npy"), mmap_mode="r")[: a.subset])
    xq = np.load(os.path.join(a.data, "queries.npy"))[: a.n_queries]
    flat = faiss.IndexFlatIP(xb.shape[1])
    flat.add(xb)
    gt = flat.search(xq, 10)[1]

    studies = {
        "neighbor selection": [("heuristic", dict(M=16, use_heuristic=True)),
                               ("naive (M closest)", dict(M=16, use_heuristic=False))],
        "M (degree)": [(f"M={m}", dict(M=m)) for m in (8, 16, 32)],
        "ef_construction": [(f"efC={e}", dict(ef_construction=e)) for e in (50, 100, 200, 400)],
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (title, variants) in zip(axes, studies.items()):
        for label, kw in variants:
            build, pts = curve(xb, xq, gt, a.threads, **kw)
            r, q = zip(*pts)
            ax.plot(r, q, "o-", label=f"{label} (build {build:.0f}s)")
            print(f"{title:20s} {label:20s} build={build:6.1f}s  " +
                  "  ".join(f"ef{e}:{rr:.3f}/{qq:.0f}" for e, (rr, qq) in zip(EFS, pts)), flush=True)
        ax.set_yscale("log")
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS (1 thread)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"vecsearch HNSW ablations ({len(xb):,} vectors)")
    fig.tight_layout()
    out = os.path.join("results", a.name, "ablations.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
