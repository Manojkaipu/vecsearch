"""Recall vs QPS chart (Pareto frontier per library) and summary.md."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    "vecsearch-hnsw": dict(color="#d62728", marker="o", lw=2.5),
    "hnswlib": dict(color="#9467bd", marker="v"),
    "faiss-hnsw": dict(color="#1f77b4", marker="s"),
    "faiss-ivfflat": dict(color="#17becf", marker="D"),
    "scann": dict(color="#2ca02c", marker="^"),
}


def pareto(df):
    """Drop points beaten on both recall and QPS by another point."""
    df = df.sort_values("recall", ascending=False)
    keep, best_qps = [], -1
    for _, r in df.iterrows():
        if r.qps > best_qps:
            keep.append(r)
            best_qps = r.qps
    return pd.DataFrame(keep).sort_values("recall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tweets")
    ap.add_argument("--min-recall", type=float, default=0.5)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join("results", args.name, "results.csv"))
    n, dim = int(df.n.iloc[0]), int(df.dim.iloc[0])

    fig, ax = plt.subplots(figsize=(9, 6))
    for algo, g in df.groupby("algo"):
        f = pareto(g[g.recall >= args.min_recall])
        ax.plot(f.recall, f.qps, label=f"{algo} ({g.config.iloc[0]})", **STYLE.get(algo, {}))
    ax.set_yscale("log")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (1 thread, log scale)")
    ax.set_title(f"Recall–throughput trade-off: {n:,} x {dim}-d vectors")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = os.path.join("results", args.name, "recall_vs_qps.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    # best QPS at fixed recall targets
    lines = ["| library | build (s) | index (MB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |", "|---|---|---|---|---|---|"]
    for algo, g in df.groupby("algo"):
        cells = []
        for t in (0.90, 0.95, 0.99):
            ok = g[g.recall >= t]
            cells.append(f"{ok.qps.max():,.0f}" if len(ok) else "—")
        lines.append(f"| {algo} | {g.build_s.iloc[0]:.0f} | {g.index_mb.iloc[0]:.0f} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    open(os.path.join("results", args.name, "summary.md"), "w").write(table + "\n")
    print(table)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
