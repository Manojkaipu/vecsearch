"""Charts for the OpenCL results: throughput vs batch size, kernel ablation, and where the time goes.

    python bench/plot_opencl.py --median results/opencl/intel_arc140v_median.csv \
        --methods results/opencl/intel_arc140v_methods.csv --profile results/opencl/intel_arc140v_profile.csv \
        --cuda results/gpu/colab.csv --out-dir results/opencl

Colours are the first three slots of the validated categorical palette (blue, orange, aqua; worst
all-pairs CVD delta-E 9.2). Aqua is under 3:1 against the light surface, so every series is also
direct-labelled and has its own marker shape.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def read(path):
    return list(csv.DictReader(open(path)))


def series(rows, backend, device_contains=""):
    pts = {}
    for r in rows:
        if r["backend"] == backend and device_contains in r["device"]:
            pts[int(r["batch"])] = float(r["qps"])
    return dict(sorted(pts.items()))


def style(ax, xlabel, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=3)


def line_chart(ax, lines):
    """lines: (label, points, colour, marker, linestyle). Direct label at the right end of each."""
    for label, pts, colour, marker, ls in lines:
        b = list(pts)
        ax.plot(b, [pts[x] for x in b], color=colour, marker=marker, linestyle=ls, linewidth=2, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.annotate(f"{pts[b[-1]]:,.0f}", (b[-1], pts[b[-1]]), xytext=(7, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted({x for _, pts, *_ in lines for x in pts}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(right=max(x for _, pts, *_ in lines for x in pts) * 1.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--median", required=True, help="median_runs.py output with opencl and cpu rows")
    ap.add_argument("--methods", help="gpu_bench.py CSV with opencl-naive/skinny/tiled rows")
    ap.add_argument("--profile", help="opencl_profile.py CSV")
    ap.add_argument("--cuda", help="results/gpu/colab.csv, drawn for context (a different GPU, run on another day)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    rows = read(args.median)
    lines = [("OpenCL, Intel Arc 140V (this laptop)", series(rows, "opencl"), BLUE, "o", "-"),
             ("AVX2 CPU index, same laptop, 8 threads", series(rows, "cpu"), ORANGE, "s", "-")]
    if args.cuda:
        lines.append(("CUDA, Colab T4 (for context)", series(read(args.cuda), "cuda"), AQUA, "^", "--"))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    line_chart(ax, lines)
    style(ax, "queries per call (batch size)", "queries / second (log scale)")
    ax.set_title("Exact top-10 over 1M x 384 vectors (cosine)", loc="left", color=INK, fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(out / "qps_vs_batch.png", dpi=140)

    if args.methods:
        m = read(args.methods)
        lines = [("naive: one work-item per pair (6-14 QPS across three runs)", series(m, "opencl-naive"), BLUE, "o", "-"),
                 ("skinny: one sub-group per vector", series(m, "opencl-skinny"), ORANGE, "s", "-"),
                 ("tiled: 128x128 local-memory tiles", series(m, "opencl-tiled"), AQUA, "^", "-")]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        line_chart(ax, lines)
        style(ax, "queries per call (batch size)", "queries / second (log scale)")
        ax.set_title("The three distance kernels on the Arc 140V", loc="left", color=INK, fontsize=10, pad=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "kernels.png", dpi=140)

    if args.profile:
        p = read(args.profile)
        batches = [int(r["batch"]) for r in p]
        dist = [sum(float(r.get(k) or 0) for k in ("naive_ms", "skinny_ms", "tiled_ms")) for r in p]
        topk = [float(r.get("select_ms") or 0) for r in p]
        other = [max(float(r["wall_ms"]) - d - t, 0) for r, d, t in zip(p, dist, topk)]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        x = range(len(batches))
        totals = [d + t + o for d, t, o in zip(dist, topk, other)]
        share = lambda v: [100 * a / tot for a, tot in zip(v, totals)]  # noqa: E731
        gap = dict(edgecolor=SURFACE, linewidth=2)  # a 2px surface gap between stacked fills
        ax.bar(x, share(dist), 0.6, color=BLUE, label="distance kernel (skinny up to 16 queries, tiled above)", **gap)
        ax.bar(x, share(topk), 0.6, bottom=share(dist), color=ORANGE, label="top-k select", **gap)
        ax.bar(x, share(other), 0.6, bottom=[a + b for a, b in zip(share(dist), share(topk))], color=AXIS,
               label="copies in and out, launch gaps", **gap)
        for i, (t, tot) in enumerate(zip(topk, totals)):
            ax.annotate(f"{tot:,.0f} ms", (i, 100), xytext=(0, 4), textcoords="offset points", ha="center",
                        color=INK2, fontsize=8)
            if t / tot >= 0.06:
                ax.text(i, 100 * (dist[i] + t / 2) / tot, f"{t / tot:.0%}", ha="center", va="center", color=INK, fontsize=8)
        ax.set_xticks(list(x), [str(b) for b in batches])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        style(ax, "queries per call (batch size)", "share of the call's time (%)")
        ax.set_title("Where one search spends its time, Arc 140V (total ms above each bar)", loc="left", color=INK,
                     fontsize=10, pad=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=1)
        fig.tight_layout()
        fig.savefig(out / "time_split.png", dpi=140, bbox_inches="tight")
    print("wrote charts to", out)


if __name__ == "__main__":
    main()
