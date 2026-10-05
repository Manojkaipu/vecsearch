"""Plot QPS vs batch size from one or more gpu_bench.py CSVs, and print a markdown table.

    python bench/plot_gpu.py results/gpu/colab_t4.csv results/gpu/laptop.csv --out results/gpu/qps_vs_batch.png
"""
import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    series = defaultdict(dict)  # (backend, device) -> {batch: qps}
    for path in args.csvs:
        for r in csv.DictReader(open(path)):
            series[(r["backend"], r["device"])][int(r["batch"])] = float(r["qps"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (backend, device), pts in series.items():
        b = sorted(pts)
        ax.plot(b, [pts[x] for x in b], marker="o", label=f"{backend} ({device})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("queries per call (batch size)")
    ax.set_ylabel("queries / second")
    ax.set_title("Exact k-NN throughput")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)

    batches = sorted({b for pts in series.values() for b in pts})
    print("| backend | device | " + " | ".join(f"B={b}" for b in batches) + " |")
    print("|---|---|" + "---|" * len(batches))
    for (backend, device), pts in series.items():
        cells = [f"{pts[b]:,.0f}" if b in pts else "—" for b in batches]
        print(f"| {backend} | {device} | " + " | ".join(cells) + " |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
