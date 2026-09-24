// Exact k-NN by linear scan. Reference for recall tests.
#pragma once
#include <vector>
#include "vecsearch/distance.h"

namespace vecsearch {

class BruteForceIndex {
 public:
  BruteForceIndex(size_t dim, Metric metric = Metric::L2);

  void add(const float* data, size_t n);
  std::vector<Neighbor> search(const float* query, size_t k) const;

  size_t size() const { return n_; }
  size_t dim() const { return dim_; }

 private:
  size_t dim_;
  Metric metric_;
  size_t n_ = 0;
  std::vector<float> data_;
};

}  // namespace vecsearch
