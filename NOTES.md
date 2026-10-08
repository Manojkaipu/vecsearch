# OpenCL + OpenVINO notes

A running log of every command and result for the `opencl` branch. Dates are 2026. Numbers here are
the raw record; the README section is the cleaned-up version.

## Step 0: hardware

Laptop: Intel Core Ultra 7 256V (Lunar Lake), integrated **Intel Arc 140V** GPU. Windows 11 host,
WSL2 Ubuntu 26.04 (kernel 6.18, `/dev/dxg` present, 11 GB RAM for WSL). No NVIDIA GPU.

```
$ wsl -d Ubuntu -u root apt-get install -y clinfo intel-opencl-icd ocl-icd-opencl-dev opencl-headers pocl-opencl-icd
$ clinfo -l
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Graphics [0x64a0]
Platform #1: Portable Computing Language
 `-- Device #0: cpu-haswell-Intel(R) Core(TM) Ultra 7 256V
```

So the Arc 140V works as an OpenCL device inside WSL2 through Intel's compute runtime (NEO), and POCL
gives a second, CPU-backed device for the CI path. Selected `clinfo` fields:

| | Intel Arc 140V (NEO 26.05.037020) | POCL 6.0 (CPU) |
|---|---|---|
| Version | OpenCL 3.0, OpenCL C 1.2 + 3.0 features | OpenCL 3.0, OpenCL C 1.2 |
| Compute units | 64 | 8 |
| Max clock | 1950 MHz | 3302 MHz |
| Max work-group size | 1024 | 4096 |
| Sub-group sizes | 16, 32 | (not reported) |
| Sub-group extensions | cl_khr_subgroups, shuffle, ballot, arithmetic, `cl_intel_subgroups` | cl_khr_subgroups |
| Local memory | 128 KiB | 2.5 MiB |
| Global memory | 7.9 GiB (shared with the CPU) | 8.7 GiB |
| **Max single allocation** | **1 GiB** | 4 GiB |

The 1 GiB allocation limit matters: 1M x 384 float32 vectors are 1.5 GB, so a single `cl_mem` can't
hold the data. The OpenCL index stores its data in pages of at most `CL_DEVICE_MAX_MEM_ALLOC_SIZE`.

Power: the laptop was on battery when I started. Benchmark runs wait for AC (see the benchmark
sections for which runs were on AC).

`NVIDIA T4` (Colab) comparison for Step 3 is not something I can run from here; see Step 3.

## Step 1: the port

Branch `opencl` (from `main` at a7c2fe6). New files:

| file | what |
|---|---|
| `include/vecsearch/gpu_method.h` | `GpuMethod` enum, moved out of the CUDA header so both indexes share it |
| `include/vecsearch/opencl_brute_force.h` | `OpenCLBruteForceIndex`, `opencl_devices()`, `opencl_available()` |
| `src/opencl/kernels.cl` | the five kernels in OpenCL C (norms, naive, skinny, tiled, select) |
| `src/opencl/kernels_embed.cpp.in` | CMake embeds `kernels.cl` as a string at configure time |
| `src/opencl_brute_force.cpp` | host code: device pick, paged buffers, lazy kernel compile, chunked search |
| `tests/cpp/test_opencl.cpp`, `tests/python/test_opencl.py` | GoogleTest and pytest suites |

CMake: `-DVECSEARCH_BUILD_OPENCL=ON` (next to `VECSEARCH_BUILD_CUDA`); needs `find_package(OpenCL)`.
Python: `vs.OpenCLBruteForceIndex(dim, metric="l2", device=0, subgroups=None)` with the same
`add(x)` / `search(q, k, method)` as `GpuBruteForceIndex`, plus `vs.opencl_available()` and
`vs.opencl_devices()`. `device` indexes `opencl_devices()`, which lists GPUs first.

How the CUDA ideas translated (also in the header of `kernels.cl`):

| CUDA | OpenCL C | note |
|---|---|---|
| `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x` | `get_local_id(0)`, `get_group_id(0)`, `get_local_size(0)`, `get_num_groups(0)` | a 2-D NDRange's global size is groups x local size |
| `__shared__`, `__syncthreads()` | `__local`, `barrier(CLK_LOCAL_MEM_FENCE)` | tiled kernel: 2 x 4 KB; select kernel: 32 KB |
| warp of 32, `__shfl_xor_sync` | sub-group, `sub_group_reduce_add` | sub-group size is 16 or 32 on Intel, chosen by the compiler; loops use `get_sub_group_size()` and grid-stride so nothing assumes 32 |
| `__launch_bounds__(n)` | `__attribute__((reqd_work_group_size(n,1,1)))` | |
| `__ldg`, `__restrict__` | `const` + `restrict` | |
| `bool` kernel argument | `int` | OpenCL C has no bool arguments |
| `x.data() + c0 * dp` | an offset argument | a `cl_mem` is a handle, not a pointer |
| `cudaMalloc` of the whole dataset | pages of at most `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | Intel caps one allocation at 1 GiB; 1M x 384 floats are 1.5 GB |
| `cudaMemcpy2D` | `clEnqueueWriteBufferRect` / `ReadBufferRect` | plain `WriteBuffer` when dim is already a multiple of 8 |
| `cudaMemset` | `clEnqueueFillBuffer` | |
| `<<<grid, block>>>` with template `K`, `QB` | `-DVS_K=n`, `-DVS_QB=n` build options | each variant is a separate run-time compile, built on first use |

Not in CUDA, added for portability: a **local-memory fallback** for devices without sub-groups
(`subgroups=False`, or automatic): the sub-group reduction becomes a 5-step tree reduction in local
memory over 32-lane slices of a 256-item work-group. The tests run both paths.

## Step 2: validation

Comparison: the one `tests/python/test_gpu.py` uses for CUDA (distances within rtol 1e-4 / atol 1e-3,
at least 99% of ids equal, rows sorted). It is not bit-for-bit: the GPU sums in a different order and
tiled L2 uses |q|^2+|x|^2-2q.x, so near-ties can swap ids. I added a stricter check on top: every
returned id's distance is recomputed on the CPU and must equal the reported one, ids can't repeat in
a row, and padded slots must be `kNone`/inf.

First run, one failure: tile-edge cases with dim 1-9 got 98.8% id agreement against the 99% bar.
Distances all matched. With dim 1 and 1000 random values there are many near-ties, so ids swap at
equal distance. The CUDA tests only use dim 37 and 128, which hides this. I did not loosen the
99% bar for the normal cases; the tiny-dim cases use 90% agreement plus the true-distance check.

Commands (WSL, repo copy at `/root/vs-opencl`, `OCL_ICD_VENDORS` pins the platform):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVECSEARCH_BUILD_OPENCL=ON && cmake --build build -j8
./build/vecsearch_tests --gtest_filter='*OpenCL*'                    # on Intel and on POCL
CMAKE_ARGS="-DVECSEARCH_BUILD_OPENCL=ON" pip install . && pytest tests/python/test_opencl.py
```

| suite | Intel Arc 140V | POCL 6.0 (CPU) |
|---|---|---|
| GoogleTest, 8 edge-case tests | 8/8 pass, 17 s | 8/8 pass, 68 s |
| GoogleTest, 96 kernel x metric x dim x batch x sub-group cases | 96/96 pass, 8 s | 96/96 pass, 28 s |
| pytest `test_opencl.py` | 181 passed, 13 s | 181 passed, 63 s |

Edge cases covered: k larger than the dataset (3 vectors, k=5; 1 vector, k=128; empty index),
batch size 1 (1-D query too), 5,000-query batch (two rounds of 4,096, with the data split into chunks),
n in {1, 9, 127, 128, 129, 255, 257, 1000} x nq in {1, 127, 128, 129, 257} x dim in {1, 7, 9, 130}
(the tile edge is 128), every k bucket (1, 16, 17, 32, 33, 64, 65, 100, 128; 129 raises),
paged storage (`VECSEARCH_OPENCL_MAX_ALLOC_MB=4` forces 5 pages of 8,192 rows on 40,000 vectors),
two-step `add` (buffer growth keeps the first part), and the sub-group and fallback paths.

CI: new `opencl` job in `.github/workflows/ci.yml` installs `pocl-opencl-icd`, builds with
`-DVECSEARCH_BUILD_OPENCL=ON`, runs the GoogleTest OpenCL tests and the full pytest suite.

## Step 4: OpenVINO setup

Environment: a separate WSL venv `/root/ov-venv` (Python 3.12): openvino 2026.4.1, openvino-genai
2026.4.1, optimum-intel 1.27.0, optimum 2.1.0, nncf 3.4.0, transformers 4.57.6, torch 2.14.1+cpu,
sentence-transformers 5.7.0. OpenVINO sees the Arc 140V inside WSL:

```
core.available_devices -> ['CPU', 'GPU']   GPU = Intel(R) Graphics [0x64a0] (iGPU), 64 EUs, arch v20.4.4
GPU_DEVICE_TOTAL_MEM_SIZE 8469446656       (the same 7.9 GiB the OpenCL runtime reports)
```

Exports (`bench/openvino/export_models.sh`, which calls `optimum-cli export openvino --weight-format ...`):

| model | fp16 | int8 | int4 |
|---|---|---|---|
| all-MiniLM-L6-v2 (22.7M params) | 45.1 MB, 34 s | 22.8 MB, 19 s | 17.9 MB, 19 s |

MiniLM recipes, from NNCF's own statistics line: int8 = `int8_asym, per-channel` on all 39 weight
tensors; int4 = `int4_asym, group size 128` on 35 of 39 tensors and int8 on the other 4.

| Qwen2.5-1.5B-Instruct (1.54B params) | 3,087 MB, 4 min 15 s | 1,546 MB, 1 min 29 s | 978 MB, 12 min 33 s |

Qwen recipes (NNCF statistics): int8 = `int8_asym, per-channel`. int4 = optimum-intel's default for this
model, which is **data-aware**: `int4_asym, group size 128` on 173 of 198 weight tensors (90% of the
ratio-defining parameters), the other 25 kept as int8 per-channel, with AWQ then scale estimation,
calibrated on wikitext2 (128 samples; 98 s statistics, 130 s AWQ, 200 s scale estimation). So "INT4"
here means that recipe, not plain round-to-nearest. Peak memory of the INT4 export was 6.1 GB.

Two things that went wrong while exporting Qwen2.5-1.5B-Instruct:
1. The INT4 export failed with `ImportError: ... get_wikitext2`. optimum-intel's default INT4 recipe for
   this model is data-aware (it calibrates on wikitext2), and that needs the `datasets` package.
2. Installing `datasets` pulled in `huggingface-hub` 2.2.0, which breaks transformers 4.57.6
   (`huggingface-hub>=0.34.0,<1.0 is required`). Pinned `huggingface-hub<1.0` (0.36.2).
   WSL has 10.7 GB and the support-rag API pod holds 3.6 GB of it, so the INT4 export (calibration pass)
   ran with under 1 GB free at its worst. It survived.

Evaluation data comes from support-rag: `bench/openvino/export_support_rag.py` writes one JSON bundle
(96 approved questions, 512 corpus chunks, and for each of the 86 stored answers that the verifier
actually checked: the exact prompt Agent.verify built and the original verdict). 7 of those 86 were
final verdicts of "unsupported". The other 10 of the 96 have no verdict to compare with: 8 were
"the history doesn't answer this" submissions with no citations (support-rag accepts those without
calling the verifier), and 2 never produced an answer (they ran out of turns).
The bundle stays in `data/` (gitignored): the conversations are tweets from the Kaggle dataset.

## Step 3: benchmark on the Intel Arc 140V

All on AC power, WSL2, Intel NEO 26.05 OpenCL driver, 1M x 384 random unit vectors, inner product,
k=10, every timed call includes copying queries in and results out. Same sweep and same
`bench/gpu_bench.py` as the CUDA results. Recall vs the CPU index is 1.0000 in every row.

```
python bench/gpu_bench.py --backends opencl,opencl-nosg,cpu --opencl-device Intel --out results/opencl/intel_arc140v.csv
python bench/gpu_bench.py --backends opencl,cpu --opencl-device Intel --max-batch cpu=1024 --out results/opencl/intel_arc140v_run{2,3}.csv
python bench/median_runs.py ... --out results/opencl/intel_arc140v_median.csv
```

Queries per second, median of 3 full sweeps (`results/opencl/intel_arc140v_median.csv`):

| | 1 | 8 | 32 | 128 | 512 | 1024 | 4096 |
|---|---|---|---|---|---|---|---|
| **OpenCL, Arc 140V** | 59 | 434 | 750 | 1,991 | 1,893 | 1,806 | 1,536 |
| AVX2 CPU index, same laptop, 8 threads | 52 | 242 | 403 | 478 | 466 | 471 | — |
| OpenCL speedup over that CPU | 1.1x | 1.8x | 1.9x | **4.2x** | 4.1x | 3.8x | — |
| OpenCL without sub-groups (1 run) | 60 | 336 | 750 | 2,030 | 1,929 | 1,854 | 1,568 |
| *for context, from the CUDA README: CUDA, Colab T4* | *159* | *824* | *1,253* | *3,691* | *3,530* | *3,358* | *2,897* |

Run-to-run spread of the OpenCL rows across the three sweeps (max-min over the median) is 7% at batch 1
and under 3.5% for batch 8-4096. The CPU baseline on the same days is 6-26% above the one in the CUDA
README table (52 vs 49 QPS at batch 1, 478 vs 397 at batch 128), so these speedups use the faster,
same-day baseline.

**Cost of portability on the same T4: not measured.** The T4 is a Colab GPU and I can't run Colab
from here. `bench/opencl_colab.ipynb` builds both backends on a T4, runs the OpenCL tests there, and
times `cuda`, `opencl`, `opencl-nosg` and the three kernels on the same GPU. It needs someone to run it
and send back `opencl_t4_results.zip`. The T4 row above is the CUDA README's, on different hardware,
and says nothing about portability cost.

### Memory-bound or compute-bound

Ceilings on this GPU, measured with plain OpenCL kernels (`bench/opencl_roofline.py`,
`results/opencl/ceilings_intel_arc140v.json`): **103.5 GB/s** streaming read (76% of the 136.5 GB/s that
LPDDR5X-8533 on a 128-bit bus gives on paper) and **3,648 GFLOP/s** FP32 FMA (91% of the 3,994
that 64 EUs x 16 lanes x 2 x 1.95 GHz gives). Ridge point 35 FLOP/byte.

One mistake worth recording: my first FMA kernel used float4 chains and measured 1,316 GFLOP/s. The
tiled kernel then measured 1,400 GFLOP/s, above the "ceiling", which is how I knew the ceiling was
wrong. Intel's compiler turns that float4 loop into code about 2.8x slower than the same loop on scalar
floats (1.3 vs 3.6 TFLOP/s). A variant I tried first also returned only lane 0, so the compiler deleted
the other lanes and reported 10-27 TFLOP/s, which is impossible. Only the scalar, every-chain-feeds-the-result
kernel is used now.

With OpenCL event profiling (`VECSEARCH_OPENCL_PROFILE=1`, `bench/opencl_profile.py`, measured after a
3 minute idle cool-down) and `bench/roofline.py`:

| batch | kernel | ms (wall) | distance kernel | top-k | bound by | distance kernel at |
|---|---|---|---|---|---|---|
| 1 | skinny | 17.0 | 15.4 ms | 0.6 ms | memory | 96% of read bandwidth (100 GB/s) |
| 8 | skinny | 18.4 | 15.8 ms | 1.7 ms | memory | 94% (97 GB/s) |
| 32 | tiled | 42.7 | 38.9 ms | 3.7 ms | compute | 69% of FP32 peak, on 128-row padded tiles |
| 128 | tiled | 64.3 | 53.9 ms | 11.0 ms | compute | 50% (1.8 TFLOP/s) |
| 512 | tiled | 270.4 | 225.0 ms | 54.6 ms | compute | 48% |
| 1024 | tiled | 567.1 | 444.0 ms | 134.8 ms | compute | 49% |
| 4096 | tiled | 2,666.7 | 1,751.2 ms | 983.9 ms | compute | 49% |

* Batches up to 16 are **memory-bound**: the skinny kernel streams the 1.5 GB of vectors at 94-96% of
  the measured read ceiling, so there is little left to gain from the kernel. The other 9-14% of the wall
  time is the top-k select and the copies.
* Batches of 32 and up are **compute-bound** (64 FLOP per byte streamed, far past the ridge point), but
  the tiled kernel reaches only about half of the FP32 peak. For scale, the CUDA README reports 2.8 TFLOP/s
  end to end for its tiled kernel at batch 128 on a T4, whose FP32 peak on paper is 8.1 (about 35%; an
  end-to-end figure against a datasheet peak, so not directly comparable with the 50% above).
* **The top-k select is the weak spot at large batches**: 17% of the time at batch 128, 36% at batch
  4096, where it reads the distance block at about 17 GB/s against a 104 GB/s ceiling. A select fused into
  the tiled kernel, or a better select, is where the next speed-up is, not the matrix multiply.
* Sub-groups vs the local-memory fallback: the sub-group version is 29% faster at batch 8 (434 vs 336
  QPS) and the fallback is 1-3% ahead everywhere else (one run each, so that is inside the noise).

Kernel ablation (cool run, `results/opencl/intel_arc140v_methods.csv`), queries per second:

| kernel | 1 | 4 | 8 | 16 | 32 | 128 | 1024 |
|---|---|---|---|---|---|---|---|
| naive | 6 | 6 | 6 | 6 | 6 | 6 | — |
| skinny | 59 | 224 | 437 | 466 | 482 | 490 | — |
| tiled | 27 | 104 | 204 | 393 | 732 | 1,938 | 1,764 |

Skinny wins up to 16 queries and tiled from 32, so `auto`'s switch at 16 (inherited from the T4) is right
on this GPU too; the crossover is between 16 (skinny 466 vs tiled 393) and 32 (482 vs 732).

**The naive kernel is not repeatable here.** Three runs of the same code gave 14, 9 and 6 QPS (71, 107
and 172 ms per query), each slower than the one before, while skinny and tiled moved by under 10%.
Its reads are uncoalesced (neighbouring work-items are 1.5 KB apart), so it is the kernel most exposed to
whatever the memory system is doing (page-table or TLB behaviour on shared memory is my guess; I did not
isolate it). Treat it as "6-14 QPS, 4-10x slower than skinny at batch 1", not as a number.

Variance and thermals: the iGPU and the CPU share one power and thermal budget. The profile and the
ablation above were re-run after a 3 minute idle cool-down, with no CPU benchmark before them. For skinny
and tiled the earlier runs, which came right after 8-thread CPU baselines, were 3-8% slower than the cool
ones; the naive kernel did not follow that pattern (its cool run was its slowest). The main-sweep OpenCL
rows are the median of three.
Earlier in the session (on battery) a 100k-vector smoke test gave a 1.24 TFLOP/s tiled rate, which is
why the sweep waited for AC.

## Step 3, second comparison: OpenCL vs CUDA on the same T4

Run by the user in Colab with `bench/opencl_colab.ipynb` (branch `opencl`); files in
`results/opencl/colab_t4*.csv`, `ceilings_t4.json`, `colab_t4_roofline.md`. clinfo on the T4: NVIDIA CUDA
platform, OpenCL 3.0, OpenCL C 1.2, 40 compute units, max 3.64 GiB allocation, 48 KiB local memory, **"Max
sub-groups per work group 0"**, so the OpenCL index took the local-memory reduction path in `opencl` as
well as in `opencl-nosg` (both rows are labelled "local-memory reduction"). Nothing here compares
sub-groups with CUDA warp shuffles.

Ceilings measured on the T4 with the same microbenchmark: 273.3 GB/s read (85% of 320), 7,649 GFLOP/s
FP32 FMA (94% of 8.1 TFLOP/s).

| queries/s | 1 | 8 | 32 | 128 | 512 | 1024 | 4096 |
|---|---|---|---|---|---|---|---|
| CUDA | 157.8 | 690.9 | 1,083 | 2,977.7 | 2,858.3 | 2,703.6 | 2,385.2 |
| OpenCL | 158.3 | 532.0 | 1,008.3 | 2,738.1 | 2,617.5 | 2,497.6 | 2,202.2 |
| ratio | 1.00 | 0.77 | 0.93 | 0.92 | 0.92 | 0.92 | 0.92 |

Per kernel, OpenCL as a share of CUDA (`colab_t4_methods.csv`): naive 0.97-1.03 (same loop), skinny 1.00
at batch 1 then 0.77-0.83 at batches 4-128, tiled 0.95-0.98. Batch 1 is memory-bound in both (89% of the
273 GB/s read ceiling, 243 vs 242 GB/s). The skinny loss is the one kernel whose warp reduction became a
local-memory reduction, and the Arc 140V shows the same direction (its sub-group version is 29% faster than the
fallback at batch 8), but the T4 has no sub-group path to confirm it. The two OpenCL rows on the T4 are the
same code and differ by about 4%, which is the noise floor there.

Caution: this session's CUDA numbers are 14-20% below the CUDA README table at batch 8 and up (2,978 vs 3,691
at batch 128), equal at batch 1. Different day, different Colab T4 (driver 580.82.07). Only same-session
numbers are compared.

The files were uploaded into the support-rag folder (`support-rag/opencl_t4_results/`, untracked there);
the four T4 files were copied here. That folder can be deleted from support-rag.
