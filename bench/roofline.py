"""Is each exact-search run limited by memory bandwidth or by compute? A roofline table.

    python bench/roofline.py results/opencl/intel_arc140v.csv --ceilings results/opencl/ceilings_intel_arc140v.json

For every row of a gpu_bench.py CSV:
  * flops = 2 * n * dim * rows, where rows is the batch for skinny and naive, and the batch rounded up
    to a multiple of 128 for tiled (it always computes whole 128-query tiles, so a batch of 32 does
    the work of 128). One multiply and one add per dimension of every query-vector pair.
  * the kernel streams the data from memory `passes` times: ceil(batch / 8) for skinny (8 queries per
    pass), ceil(batch / 128) for tiled (one pass per 128-query tile row), batch for naive. Auto is
    skinny up to 16 queries and tiled above. Caches can absorb some re-reads, so this is an upper bound
    on traffic and a lower bound on arithmetic intensity.
  * arithmetic intensity = flops / bytes streamed. The ridge point is fma_gflops / read_gb_per_s: below
    it a kernel can't reach the compute ceiling however good it is, so it is memory-bound.
  * the achieved fraction is against the ceiling that binds: effective GB/s over the read-bandwidth
    ceiling when memory-bound, GFLOP/s over the FMA ceiling when compute-bound.

The ceilings are what simple kernels reach on the same device (bench/opencl_roofline.py), not datasheet
peaks. Times are end to end: queries copied in, distances, top-k, results copied out.

With --profile (bench/opencl_profile.py's CSV) the table also splits each OpenCL run into the distance
kernel and the top-k select, and gives the distance kernel's own % of its ceiling: a run can be
compute-bound and still spend a quarter of its time somewhere else.
"""
import argparse
import csv
import json
import math
from pathlib import Path

AUTO_SKINNY_MAX = 16
SKINNY_QUERIES = 8
TILE = 128


def method_of(backend, batch):
    name = backend.split("-", 1)[1] if "-" in backend else "auto"
    if name in ("auto", "nosg"):
        return "skinny" if batch <= AUTO_SKINNY_MAX else "tiled"
    return name


def passes_of(method, batch):
    if method == "skinny":
        return math.ceil(batch / SKINNY_QUERIES)
    if method == "tiled":
        return math.ceil(batch / TILE)
    return batch  # naive: each (query, vector) pair reads the vector


def analyse(row, ceilings, profile=None):
    n, dim, batch = int(row["n"]), int(row["dim"]), int(row["batch"])
    sec = float(row["ms_per_batch"]) / 1e3
    method = method_of(row["backend"], batch)
    passes = passes_of(method, batch)
    rows_computed = passes * TILE if method == "tiled" else batch
    flops = 2.0 * n * dim * rows_computed
    streamed = passes * n * dim * 4.0
    ai = flops / streamed
    ridge = ceilings["fma_gflops"] / ceilings["read_gb_per_s"]
    gflops, gbs = flops / sec / 1e9, streamed / sec / 1e9
    memory_bound = ai < ridge
    frac = gbs / ceilings["read_gb_per_s"] if memory_bound else gflops / ceilings["fma_gflops"]
    out = dict(backend=row["backend"], batch=batch, method=method, ms=float(row["ms_per_batch"]),
               qps=float(row["qps"]), passes=passes, ai=ai, gflops=gflops, gbs=gbs,
               bound="memory" if memory_bound else "compute", frac=frac, dist_ms=None, topk_ms=None, kernel_frac=None)
    if profile and batch in profile and row["backend"] == "opencl":
        p = profile[batch]
        dist_ms = sum(float(p.get(f"{m}_ms", 0) or 0) for m in ("naive", "skinny", "tiled"))
        out.update(dist_ms=dist_ms, topk_ms=float(p.get("select_ms", 0) or 0))
        k_gbs, k_gflops = streamed / (dist_ms / 1e3) / 1e9, flops / (dist_ms / 1e3) / 1e9
        out["kernel_frac"] = k_gbs / ceilings["read_gb_per_s"] if memory_bound else k_gflops / ceilings["fma_gflops"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--ceilings", required=True, help="JSON from bench/opencl_roofline.py")
    ap.add_argument("--profile", help="CSV from bench/opencl_profile.py, for the kernel-time split")
    ap.add_argument("--backends", help="comma list; default every backend in the CSV")
    ap.add_argument("--out", help="also write the table here")
    args = ap.parse_args()

    ceilings = json.loads(Path(args.ceilings).read_text())
    rows = list(csv.DictReader(open(args.csv)))
    profile = {int(r["batch"]): r for r in csv.DictReader(open(args.profile))} if args.profile else None
    keep = set(args.backends.split(",")) if args.backends else {r["backend"] for r in rows}
    lines = [f"Ceilings ({ceilings['device']}): {ceilings['read_gb_per_s']:.0f} GB/s read, "
             f"{ceilings['fma_gflops']:,.0f} GFLOP/s FP32 FMA; ridge point "
             f"{ceilings['fma_gflops'] / ceilings['read_gb_per_s']:.0f} FLOP/byte.", "",
             "| backend | batch | kernel | ms | queries/s | data passes | FLOP/byte | GB/s streamed | GFLOP/s computed | bound by | % of that ceiling "
             + ("| distance kernel (ms) | top-k (ms) | distance kernel alone: % of ceiling |" if profile else "|"),
             "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|---|" if profile else "")]
    for r in rows:
        if r["backend"] not in keep:
            continue
        a = analyse(r, ceilings, profile)
        extra = ""
        if profile:
            extra = (f" {a['dist_ms']:.1f} | {a['topk_ms']:.1f} | {a['kernel_frac']:.0%} |" if a["dist_ms"] is not None
                     else " | | |")
        lines.append(f"| {a['backend']} | {a['batch']} | {a['method']} | {a['ms']:.1f} | {a['qps']:,.0f} | {a['passes']} "
                     f"| {a['ai']:.1f} | {a['gbs']:.0f} | {a['gflops']:,.0f} | {a['bound']} | {a['frac']:.0%} |" + extra)
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
