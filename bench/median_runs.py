"""Median of repeated gpu_bench.py runs, as one CSV in the same format.

    python bench/median_runs.py results/opencl/intel_arc140v.csv results/opencl/intel_arc140v_run2.csv \
        results/opencl/intel_arc140v_run3.csv --out results/opencl/intel_arc140v_median.csv

For each (backend, batch) that appears in at least two runs it takes the median ms_per_batch and
recomputes qps; the spread (min and max across runs) goes into extra columns. The recall is the lowest seen.
"""
import argparse
import csv
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups, first = defaultdict(list), {}
    for path in args.csvs:
        for r in csv.DictReader(open(path)):
            key = (r["backend"], int(r["batch"]))
            groups[key].append(r)
            first.setdefault(key, r)
    fields = ["backend", "device", "n", "dim", "metric", "k", "batch", "ms_per_batch", "qps", "recall_vs_cpu",
              "runs", "ms_min", "ms_max"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(groups, key=lambda k: (list(first).index(k), k[1])):
            rs = groups[key]
            ms = [float(r["ms_per_batch"]) for r in rs]
            med = statistics.median(ms)
            w.writerow(dict({k: first[key][k] for k in ("backend", "device", "n", "dim", "metric", "k", "batch")},
                            ms_per_batch=round(med, 3), qps=round(key[1] / (med / 1e3), 1),
                            recall_vs_cpu=min(float(r["recall_vs_cpu"]) for r in rs), runs=len(rs),
                            ms_min=round(min(ms), 3), ms_max=round(max(ms), 3)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
