"""Plot p50/p99 latency vs number of shards from latency_bench.py output."""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="tweets")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join("results", a.name, "distributed_c*.csv")))
fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 4.5), squeeze=False)
for ax, f in zip(axes[0], files):
    df = pd.read_csv(f)
    g = df[df["mode"] == "grpc"]
    base = df[df["mode"] == "in-process"].iloc[0]
    ax.plot(g.shards, g.p50_ms, "o-", label="p50")
    ax.plot(g.shards, g.p99_ms, "s-", label="p99")
    ax.axhline(base.p50_ms, ls="--", c="gray", label="in-process p50 (no RPC)")
    for _, r in g.iterrows():
        ax.annotate(f"R={r.recall:.3f}", (r.shards, r.p99_ms), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.set_xticks(g.shards)
    ax.set_xlabel("shards (worker processes)")
    ax.set_ylabel("latency (ms)")
    ax.set_title(os.path.basename(f).replace(".csv", "").replace("distributed_c", "concurrency="))
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
out = os.path.join("results", a.name, "latency_vs_shards.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
