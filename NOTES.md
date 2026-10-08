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
