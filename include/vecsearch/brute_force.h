// Exact k-NN by linear scan. Reference for recall tests, and the CPU baseline for the GPU index.
#pragma once
#include <vector>
#include "vecsearch/distance.h"

namespace vecsearch {

class BruteForceIndex {
 public:
  static constexpr uint32_t kNone = 0xFFFFFFFFu;  // id of a missing result (k > size)

  BruteForceIndex(size_t dim, Metric metric = Metric::L2);

  void add(const float* data, size_t n);
  std::vector<Neighbor> search(const float* query, size_t k) const;

  // Top-k for nq queries into out_ids/out_dists (nq x k, nearest first; missing = kNone, inf).
  // Scans the data in cache-sized tiles, four queries at a time, split across threads by
  // data range. Same arithmetic as search(), so the results match it.
  void search_batch(const float* queries, size_t nq, size_t k, uint32_t* out_ids, float* out_dists,
                    int num_threads = 1) const;

  size_t size() const { return n_; }
  size_t dim() const { return dim_; }
  Metric metric() const { return metric_; }

 private:
  size_t dim_;
  Metric metric_;
  size_t n_ = 0;
  std::vector<float> data_;
};

}  // namespace vecsearch
