"""Measure an OpenCL device's streaming read bandwidth and FP32 FMA throughput, the two ceilings
bench/roofline.py compares the brute-force kernels against.

    pip install pyopencl
    python bench/opencl_roofline.py --device Intel --out results/opencl/ceilings_intel_arc140v.json

Both numbers are what a simple kernel achieves, not the datasheet figures: a datasheet bandwidth
is rarely reachable, and a kernel that is a fraction of a datasheet ceiling can still be the best
the device does.
"""
import argparse
import json
import statistics
import time

import numpy as np
import pyopencl as cl

READ_SRC = """
__kernel void read_sum(__global const float4* x, __global float* out, ulong n4) {
  float4 acc = (float4)(0.f);
  for (ulong i = get_global_id(0); i < n4; i += get_global_size(0)) acc += x[i];
  out[get_global_id(0)] = acc.x + acc.y + acc.z + acc.w;
}
"""

# 24 independent scalar FMA chains per work-item: enough parallelism to cover FMA latency, no memory
# traffic, and every chain feeds the result so the compiler can't delete any. Scalar on purpose: the
# same loop written with float2/float4 chains compiled to about a third of this speed on Intel's
# compiler (1.3 vs 3.6 TFLOP/s on the Arc 140V), which would understate the ceiling.
FMA_CHAINS = 24


def fma_source(chains):
    init = "\n".join(f"  float x{i} = (float)(get_global_id(0) + {i});" for i in range(chains))
    step = "\n".join(f"    x{i} = fma(x{i}, a, b);" for i in range(chains))
    total = " + ".join(f"x{i}" for i in range(chains))
    return (f"__kernel void fma_loop(__global float* out, int iters, float a, float b) {{\n{init}\n"
            f"  for (int i = 0; i < iters; ++i) {{\n{step}\n  }}\n  out[get_global_id(0)] = {total};\n}}\n")


FMA_SRC = fma_source(FMA_CHAINS)


def pick(spec):
    devs = [(p, d) for p in cl.get_platforms() for d in p.get_devices()]
    devs.sort(key=lambda pd: not (pd[1].type & cl.device_type.GPU))  # GPUs first, like vecsearch
    if spec.isdigit():
        return devs[int(spec)]
    for p, d in devs:
        if spec.lower() in f"{p.name} / {d.name}".lower():
            return p, d
    raise SystemExit(f"no device matches {spec!r}; have: " + "; ".join(f"{p.name} / {d.name}" for p, d in devs))


def timed(fn, repeats):
    fn()  # warm-up (compile, caches)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), min(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="0")
    ap.add_argument("--read-mb", type=int, default=512, help="size of the buffer streamed by the read test")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plat, dev = pick(args.device)
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    cus = dev.max_compute_units
    print(f"{plat.name} / {dev.name}: {cus} compute units @ {dev.max_clock_frequency} MHz")

    # Streaming read: every work-item sums a strided slice of one big buffer.
    nbytes = min(args.read_mb << 20, dev.max_mem_alloc_size)
    n4 = nbytes // 16
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, n4 * 16)
    cl.enqueue_fill_buffer(q, buf, np.float32(1.0), 0, n4 * 16)
    q.finish()
    groups = cus * 32
    glob, loc = groups * 256, 256
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, glob * 4)
    k_read = cl.Program(ctx, READ_SRC).build().read_sum

    def run_read():
        k_read(q, (glob,), (loc,), buf, out, np.uint64(n4))
        q.finish()

    med, best = timed(run_read, args.repeats)
    read_gbs = nbytes / med / 1e9
    print(f"read   {nbytes >> 20} MiB: median {med * 1e3:.2f} ms = {read_gbs:.1f} GB/s (best {nbytes / best / 1e9:.1f})")

    # FP32 FMA throughput.
    iters = 2048
    work = cus * 8 * 16 * 128  # plenty of work-items to fill every thread slot (about 1M on 64 CUs)
    out2 = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, work * 4)
    k_fma = cl.Program(ctx, FMA_SRC).build().fma_loop

    def run_fma():
        k_fma(q, (work,), None, out2, np.int32(iters), np.float32(0.999), np.float32(0.001))
        q.finish()

    med, best = timed(run_fma, args.repeats)
    flop = work * iters * FMA_CHAINS * 2  # items * iterations * chains * 2 (multiply + add)
    gflops = flop / med / 1e9
    print(f"fma    {work} work-items x {iters} iters: median {med * 1e3:.2f} ms = {gflops:.0f} GFLOP/s (best {flop / best / 1e9:.0f})")

    result = dict(platform=plat.name, device=dev.name, compute_units=cus, clock_mhz=dev.max_clock_frequency,
                  read_gb_per_s=round(read_gbs, 1), fma_gflops=round(gflops, 0),
                  read_bytes=nbytes, note="median of %d runs; simple kernels, not datasheet peaks" % args.repeats)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
