#include "vecsearch/gpu_brute_force.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                              \
  do {                                                                                \
    cudaError_t err_ = (expr);                                                        \
    if (err_ != cudaSuccess)                                                          \
      throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err_)); \
  } while (0)

namespace vecsearch {

namespace {

constexpr uint32_t kNoId = GpuBruteForceIndex::kNone;
constexpr int kSkinnyQueries = 8;          // queries per skinny pass over the data
constexpr size_t kSkinnyMaxBatch = 8;      // Auto picks Skinny up to this many queries
constexpr size_t kMaxQueryTile = 4096;     // queries per round (bounds the candidate buffers)
constexpr size_t kDistBudget = size_t(1) << 27;  // floats in the distance block (512 MB)
constexpr int kSelectSlots = 4096;         // candidates one select block sorts in shared memory
constexpr size_t kMinSegment = 16384;      // a select block scans at least this many distances
constexpr int kTile = 128, kTileK = 8;     // tiled kernel: 128x128 outputs, 8 dims per step

struct CudaFree {
  void operator()(void* p) const { cudaFree(p); }
};

// Grow-only device buffer, zero-filled when (re)allocated.
template <class T>
class DeviceArray {
 public:
  T* data() const { return p_.get(); }
  void reserve(size_t n, size_t keep = 0) {
    if (n <= n_) return;
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, n * sizeof(T)));
    std::unique_ptr<T, CudaFree> p(static_cast<T*>(raw));
    CUDA_CHECK(cudaMemset(p.get(), 0, n * sizeof(T)));
    if (keep) CUDA_CHECK(cudaMemcpy(p.get(), p_.get(), keep * sizeof(T), cudaMemcpyDeviceToDevice));
    p_ = std::move(p);
    n_ = n;
  }

 private:
  std::unique_ptr<T, CudaFree> p_;
  size_t n_ = 0;
};

__device__ inline float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Squared norm of each row, one warp per row.
__global__ void norms_kernel(const float* __restrict__ x, size_t n, int dp, float* __restrict__ out) {
  const size_t row = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (row >= n) return;  // the whole warp leaves together
  const float* r = x + row * dp;
  float s = 0.f;
  for (int i = lane; i < dp; i += 32) s = fmaf(r[i], r[i], s);
  s = warp_sum(s);
  if (lane == 0) out[row] = s;
}

// The CPU loop as it is: one thread per (query, vector) pair, both read from global memory.
// Neighbouring threads read rows dp floats apart, so loads don't coalesce.
__global__ void naive_kernel(const float* __restrict__ q, const float* __restrict__ x, int nq, int n,
                             int dp, bool l2, float* __restrict__ out, size_t ld) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  if (j >= n || i >= nq) return;
  const float* a = q + size_t(i) * dp;
  const float* b = x + size_t(j) * dp;
  float r = 0.f;
  if (l2) {
    for (int t = 0; t < dp; ++t) {
      const float e = a[t] - b[t];
      r = fmaf(e, e, r);
    }
  } else {
    for (int t = 0; t < dp; ++t) r = fmaf(a[t], b[t], r);
    r = 1.f - r;
  }
  out[size_t(i) * ld + j] = r;
}

// One warp per vector: lanes read consecutive float4s (coalesced), each vector is read once for
// QB queries, and a warp reduction finishes each distance. Small batches are bound by reading the
// data, so this aims to stream it at full bandwidth.
template <int QB>
__global__ void skinny_kernel(const float* __restrict__ q, const float* __restrict__ x, int n, int dp,
                              bool l2, float* __restrict__ out, size_t ld) {
  const int lane = threadIdx.x % 32;
  const size_t nwarps = size_t(gridDim.x) * blockDim.x / 32;
  const int d4 = dp / 4;
  const float4* q4 = reinterpret_cast<const float4*>(q);
  for (size_t j = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; j < size_t(n); j += nwarps) {
    const float4* x4 = reinterpret_cast<const float4*>(x + j * dp);
    float acc[QB];
#pragma unroll
    for (int b = 0; b < QB; ++b) acc[b] = 0.f;
    for (int t = lane; t < d4; t += 32) {
      const float4 v = x4[t];
#pragma unroll
      for (int b = 0; b < QB; ++b) {
        const float4 u = __ldg(q4 + b * d4 + t);
        if (l2) {
          float e = u.x - v.x;
          acc[b] = fmaf(e, e, acc[b]);
          e = u.y - v.y;
          acc[b] = fmaf(e, e, acc[b]);
          e = u.z - v.z;
          acc[b] = fmaf(e, e, acc[b]);
          e = u.w - v.w;
          acc[b] = fmaf(e, e, acc[b]);
        } else {
          acc[b] = fmaf(u.x, v.x, acc[b]);
          acc[b] = fmaf(u.y, v.y, acc[b]);
          acc[b] = fmaf(u.z, v.z, acc[b]);
          acc[b] = fmaf(u.w, v.w, acc[b]);
        }
      }
    }
#pragma unroll
    for (int b = 0; b < QB; ++b) {
      const float s = warp_sum(acc[b]);
      if (lane == 0) out[b * ld + j] = l2 ? s : 1.f - s;
    }
  }
}

// q.x for a 128x128 tile of (query, vector) pairs: the classic shared-memory SGEMM. Each step
// stages 8 dims of 128 queries and 128 vectors in shared memory, and each of the 256 threads
// accumulates an 8x8 block in registers, so every value loaded from shared memory feeds 8 FMAs.
// The next step's global loads are issued before this step's math to hide their latency.
// L2 uses |q|^2 + |x|^2 - 2 q.x, so the data norms are computed once at add().
__global__ void __launch_bounds__(256)
    tiled_kernel(const float* __restrict__ q, const float* __restrict__ x, int nq, int n, int dp,
                 bool l2, const float* __restrict__ qn, const float* __restrict__ xn,
                 float* __restrict__ out, size_t ld) {
  __shared__ __align__(16) float qs[kTileK][kTile];
  __shared__ __align__(16) float xs[kTileK][kTile];
  const int tid = threadIdx.x;
  const int row0 = blockIdx.y * kTile, col0 = blockIdx.x * kTile;

  // Loading: each thread fetches 4 consecutive dims of one query row and one data row.
  const int lr = tid / 2, lk = (tid % 2) * 4;
  const bool q_ok = row0 + lr < nq, x_ok = col0 + lr < n;
  const float4* qp = reinterpret_cast<const float4*>(q + size_t(row0 + lr) * dp + lk);
  const float4* xp = reinterpret_cast<const float4*>(x + size_t(col0 + lr) * dp + lk);
  const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 qv = q_ok ? __ldg(qp) : zero;
  float4 xv = x_ok ? __ldg(xp) : zero;

  // Computing: rows ty*4+{0..3} and 64+ty*4+{0..3}, columns likewise with tx. Splitting each
  // thread's block in two halves keeps the float4 reads from shared memory free of bank conflicts.
  const int ty = tid / 16, tx = tid % 16;
  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < dp; k0 += kTileK) {
    qs[lk + 0][lr] = qv.x;
    qs[lk + 1][lr] = qv.y;
    qs[lk + 2][lr] = qv.z;
    qs[lk + 3][lr] = qv.w;
    xs[lk + 0][lr] = xv.x;
    xs[lk + 1][lr] = xv.y;
    xs[lk + 2][lr] = xv.z;
    xs[lk + 3][lr] = xv.w;
    __syncthreads();
    if (k0 + kTileK < dp) {
      qv = q_ok ? __ldg(qp + (k0 + kTileK) / 4) : zero;
      xv = x_ok ? __ldg(xp + (k0 + kTileK) / 4) : zero;
    }
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      const float4 a0 = *reinterpret_cast<const float4*>(&qs[kk][ty * 4]);
      const float4 a1 = *reinterpret_cast<const float4*>(&qs[kk][64 + ty * 4]);
      const float4 b0 = *reinterpret_cast<const float4*>(&xs[kk][tx * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&xs[kk][64 + tx * 4]);
      const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (r >= nq) continue;
    const float qr = l2 ? qn[r] : 0.f;
    float* o = out + size_t(r) * ld;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = col0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
      if (c >= n) continue;
      const float s = acc[i][j];
      o[c] = l2 ? fmaxf(qr + xn[c] - 2.f * s, 0.f) : 1.f - s;
    }
  }
}

// Top K of one segment of one row. Each thread keeps a sorted list of the K best distances it
// has seen; once the list is full most distances lose to its last entry, so inserts are rare.
// The lists (4096 entries in all) are then bitonic-sorted in shared memory and the first K kept.
// ids == nullptr means the id is id_base + column.
template <int K>
__global__ void __launch_bounds__(kSelectSlots / K)
    select_kernel(const float* __restrict__ dist, const uint32_t* __restrict__ ids, size_t ld_in,
                  uint32_t id_base, int len, int seg_len, float* __restrict__ out_d,
                  uint32_t* __restrict__ out_i, size_t ld_out, int out_col) {
  constexpr int kThreads = kSelectSlots / K;
  __shared__ float sd[kSelectSlots];
  __shared__ uint32_t si[kSelectSlots];
  const int tid = threadIdx.x;
  const size_t row = blockIdx.y;
  const float* d_row = dist + row * ld_in;
  const uint32_t* i_row = ids ? ids + row * ld_in : nullptr;
  const int begin = blockIdx.x * seg_len, end = min(len, begin + seg_len);

  float td[K];
  uint32_t ti[K];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    td[j] = INFINITY;
    ti[j] = kNoId;
  }
  for (int c = begin + tid; c < end; c += kThreads) {
    float d = d_row[c];
    if (d < td[K - 1]) {
      uint32_t id = i_row ? i_row[c] : id_base + uint32_t(c);
#pragma unroll
      for (int j = 0; j < K; ++j) {
        if (d < td[j]) {
          const float t = td[j];
          td[j] = d;
          d = t;
          const uint32_t u = ti[j];
          ti[j] = id;
          id = u;
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < K; ++j) {  // order doesn't matter before the sort; this one avoids bank conflicts
    sd[j * kThreads + tid] = td[j];
    si[j * kThreads + tid] = ti[j];
  }
  __syncthreads();

  for (int size = 2; size <= kSelectSlots; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      for (int t = tid; t < kSelectSlots / 2; t += kThreads) {
        const int lo = 2 * t - (t & (stride - 1)), hi = lo + stride;
        const bool ascending = (lo & size) == 0;
        const float a = sd[lo], b = sd[hi];
        const uint32_t ia = si[lo], ib = si[hi];
        const bool a_after_b = a > b || (a == b && ia > ib);
        if (a_after_b == ascending) {
          sd[lo] = b;
          sd[hi] = a;
          si[lo] = ib;
          si[hi] = ia;
        }
      }
      __syncthreads();
    }
  }
  const size_t o = row * ld_out + out_col + size_t(blockIdx.x) * K;
  for (int j = tid; j < K; j += kThreads) {
    out_d[o + j] = sd[j];
    out_i[o + j] = si[j];
  }
}

int select_bucket(size_t k) {
  for (int K : {16, 32, 64, 128})
    if (k <= size_t(K)) return K;
  throw std::invalid_argument("k must be <= " + std::to_string(GpuBruteForceIndex::kMaxK));
}

void launch_select(int K, dim3 grid, const float* dist, const uint32_t* ids, size_t ld_in,
                   uint32_t id_base, size_t len, size_t seg_len, float* out_d, uint32_t* out_i,
                   size_t ld_out, size_t out_col) {
#define VECSEARCH_SELECT(KK)                                                                    \
  select_kernel<KK><<<grid, kSelectSlots / KK>>>(dist, ids, ld_in, id_base, int(len), int(seg_len), \
                                                  out_d, out_i, ld_out, int(out_col))
  switch (K) {
    case 16: VECSEARCH_SELECT(16); break;
    case 32: VECSEARCH_SELECT(32); break;
    case 64: VECSEARCH_SELECT(64); break;
    default: VECSEARCH_SELECT(128); break;
  }
#undef VECSEARCH_SELECT
  CUDA_CHECK(cudaGetLastError());
}

void launch_skinny(int nb, int blocks, const float* q, const float* x, int n, int dp, bool l2,
                   float* out, size_t ld) {
  switch (nb) {
    case 1: skinny_kernel<1><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    case 2: skinny_kernel<2><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    case 3: skinny_kernel<3><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    case 4: skinny_kernel<4><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    case 5: skinny_kernel<5><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    case 6: skinny_kernel<6><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    case 7: skinny_kernel<7><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
    default: skinny_kernel<8><<<blocks, 256>>>(q, x, n, dp, l2, out, ld); break;
  }
}

size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

}  // namespace

bool cuda_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error, e.g. no driver
    return false;
  }
  return count > 0;
}

struct GpuBruteForceIndex::Impl {
  size_t dim, dp;
  Metric metric;
  int device, sms = 1;
  size_t n = 0, cap = 0;
  DeviceArray<float> x, xn;  // data (cap x dp) and squared norms
  DeviceArray<float> q, qn, dist, cand_d[2];
  DeviceArray<uint32_t> cand_i[2];
  std::mutex mu;

  // Segment length for a select pass: enough segments to fill the GPU, but each long enough
  // that scanning it outweighs the fixed cost of the sort.
  size_t seg_len(size_t rows, size_t len) const {
    const size_t want = std::max<size_t>(1, size_t(2 * sms) / rows);
    return std::max(kMinSegment, ceil_div(len, want));
  }

  void distances(GpuMethod m, size_t qt, size_t c0, size_t len, size_t ld) {
    const bool l2 = metric == Metric::L2;
    const float* xc = x.data() + c0 * dp;
    switch (m) {
      case GpuMethod::Naive:
        naive_kernel<<<dim3(unsigned(ceil_div(len, 256)), unsigned(qt)), 256>>>(
            q.data(), xc, int(qt), int(len), int(dp), l2, dist.data(), ld);
        break;
      case GpuMethod::Skinny: {
        const int blocks = int(std::min<size_t>(ceil_div(len, 8), size_t(sms) * 16));
        for (size_t b0 = 0; b0 < qt; b0 += kSkinnyQueries) {
          const int nb = int(std::min<size_t>(kSkinnyQueries, qt - b0));
          launch_skinny(nb, blocks, q.data() + b0 * dp, xc, int(len), int(dp), l2,
                        dist.data() + b0 * ld, ld);
        }
        break;
      }
      default:
        tiled_kernel<<<dim3(unsigned(ceil_div(len, kTile)), unsigned(ceil_div(qt, kTile))), 256>>>(
            q.data(), xc, int(qt), int(len), int(dp), l2, qn.data(), xn.data() + c0, dist.data(), ld);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  }
};

GpuBruteForceIndex::GpuBruteForceIndex(size_t dim, Metric metric, int device) : impl_(new Impl) {
  if (dim == 0) throw std::invalid_argument("dim must be > 0");
  impl_->dim = dim;
  impl_->dp = (dim + 7) / 8 * 8;
  impl_->metric = metric;
  impl_->device = device;
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  impl_->sms = prop.multiProcessorCount;
}

GpuBruteForceIndex::~GpuBruteForceIndex() = default;

size_t GpuBruteForceIndex::size() const { return impl_->n; }
size_t GpuBruteForceIndex::dim() const { return impl_->dim; }
Metric GpuBruteForceIndex::metric() const { return impl_->metric; }

void GpuBruteForceIndex::add(const float* data, size_t m) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> g(s.mu);
  if (m == 0) return;
  if (s.n + m >= kNoId) throw std::length_error("GpuBruteForceIndex holds at most 2^32 - 2 vectors");
  CUDA_CHECK(cudaSetDevice(s.device));
  if (s.n + m > s.cap) {
    const size_t cap = std::max({s.n + m, s.cap * 2, size_t(1024)});
    s.x.reserve(cap * s.dp, s.n * s.dp);
    s.xn.reserve(cap, s.n);
    s.cap = cap;
  }
  float* dst = s.x.data() + s.n * s.dp;
  CUDA_CHECK(cudaMemcpy2D(dst, s.dp * sizeof(float), data, s.dim * sizeof(float), s.dim * sizeof(float),
                          m, cudaMemcpyHostToDevice));
  norms_kernel<<<unsigned(ceil_div(m * 32, 256)), 256>>>(dst, m, int(s.dp), s.xn.data() + s.n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  s.n += m;
}

void GpuBruteForceIndex::search(const float* queries, size_t nq, size_t k, uint32_t* out_ids,
                                float* out_dists, GpuMethod method) {
  if (nq == 0 || k == 0) return;
  const int K = select_bucket(k);
  Impl& s = *impl_;
  std::lock_guard<std::mutex> g(s.mu);
  if (s.n == 0) {
    std::fill(out_ids, out_ids + nq * k, kNoId);
    std::fill(out_dists, out_dists + nq * k, INFINITY);
    return;
  }
  CUDA_CHECK(cudaSetDevice(s.device));
  const GpuMethod m =
      method != GpuMethod::Auto ? method : nq <= kSkinnyMaxBatch ? GpuMethod::Skinny : GpuMethod::Tiled;

  // Queries go in rounds of up to qt_max; the data in chunks sized so a round's distance
  // block fits the budget. Each chunk leaves nseg * K candidates per query.
  const size_t qt_max = std::min(nq, kMaxQueryTile);
  const size_t chunk = std::min(s.n, std::max<size_t>(kTile, kDistBudget / qt_max / kTile * kTile));
  const size_t seg = s.seg_len(qt_max, chunk);
  size_t width = 0;
  for (size_t c0 = 0; c0 < s.n; c0 += chunk) width += ceil_div(std::min(chunk, s.n - c0), seg) * K;

  s.q.reserve(qt_max * s.dp);
  s.qn.reserve(qt_max);
  s.dist.reserve(qt_max * chunk);
  for (int b = 0; b < 2; ++b) {
    s.cand_d[b].reserve(qt_max * width);
    s.cand_i[b].reserve(qt_max * width);
  }

  for (size_t q0 = 0; q0 < nq; q0 += qt_max) {
    const size_t qt = std::min(qt_max, nq - q0);
    // Padding columns of s.q were zeroed when it was allocated and are never written.
    CUDA_CHECK(cudaMemcpy2D(s.q.data(), s.dp * sizeof(float), queries + q0 * s.dim, s.dim * sizeof(float),
                            s.dim * sizeof(float), qt, cudaMemcpyHostToDevice));
    if (m == GpuMethod::Tiled && s.metric == Metric::L2) {
      norms_kernel<<<unsigned(ceil_div(qt * 32, 256)), 256>>>(s.q.data(), qt, int(s.dp), s.qn.data());
      CUDA_CHECK(cudaGetLastError());
    }

    size_t cols = 0;
    for (size_t c0 = 0; c0 < s.n; c0 += chunk) {
      const size_t len = std::min(chunk, s.n - c0);
      s.distances(m, qt, c0, len, chunk);
      const size_t nseg = ceil_div(len, seg);
      launch_select(K, dim3(unsigned(nseg), unsigned(qt)), s.dist.data(), nullptr, chunk, uint32_t(c0),
                    len, seg, s.cand_d[0].data(), s.cand_i[0].data(), width, cols);
      cols += nseg * K;
    }
    int src = 0;  // merge the candidates until one sorted list of K is left per query
    while (cols > size_t(K)) {
      const size_t sl = s.seg_len(qt, cols), nseg = ceil_div(cols, sl);
      launch_select(K, dim3(unsigned(nseg), unsigned(qt)), s.cand_d[src].data(), s.cand_i[src].data(),
                    width, 0, cols, sl, s.cand_d[1 - src].data(), s.cand_i[1 - src].data(), width, 0);
      src = 1 - src;
      cols = nseg * K;
    }
    CUDA_CHECK(cudaMemcpy2D(out_ids + q0 * k, k * sizeof(uint32_t), s.cand_i[src].data(),
                            width * sizeof(uint32_t), k * sizeof(uint32_t), qt, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(out_dists + q0 * k, k * sizeof(float), s.cand_d[src].data(),
                            width * sizeof(float), k * sizeof(float), qt, cudaMemcpyDeviceToHost));
  }
}

}  // namespace vecsearch
