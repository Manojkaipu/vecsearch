// Exact k-NN on an NVIDIA GPU. Built only with -DVECSEARCH_BUILD_CUDA=ON.
//
// The data lives on the device, padded to a multiple of 8 dims with zeros. A search computes a
// block of distances with one of the kernels below, then picks the top k of each row with a
// block-wide select. Big batches go through the data in chunks, so the distance block stays
// under a fixed memory budget, and the per-chunk winners are merged at the end.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include "vecsearch/distance.h"

namespace vecsearch {

enum class GpuMethod : uint32_t {
  Auto = 0,    // Skinny for small batches, Tiled otherwise
  Naive = 1,   // one thread per (query, vector) pair: the AVX2 loop, ported as is
  Skinny = 2,  // one warp per vector, up to 8 queries at a time; for memory-bound small batches
  Tiled = 3,   // 128x128 register-blocked tiles of q.x (a hand-written SGEMM); for big batches
};

// True if the CUDA runtime finds at least one device.
bool cuda_available();

class GpuBruteForceIndex {
 public:
  static constexpr uint32_t kNone = 0xFFFFFFFFu;  // id of a missing result (k > size)
  static constexpr size_t kMaxK = 128;

  GpuBruteForceIndex(size_t dim, Metric metric = Metric::L2, int device = 0);
  ~GpuBruteForceIndex();
  GpuBruteForceIndex(const GpuBruteForceIndex&) = delete;
  GpuBruteForceIndex& operator=(const GpuBruteForceIndex&) = delete;

  void add(const float* data, size_t n);

  // Top-k for nq host queries into host arrays (nq x k, nearest first; missing = kNone, inf).
  // Includes copying the queries in and the results out. k <= kMaxK.
  void search(const float* queries, size_t nq, size_t k, uint32_t* out_ids, float* out_dists,
              GpuMethod method = GpuMethod::Auto);

  size_t size() const;
  size_t dim() const;
  Metric metric() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vecsearch
