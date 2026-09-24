#include "vecsearch/brute_force.h"

#include <queue>

namespace vecsearch {

BruteForceIndex::BruteForceIndex(size_t dim, Metric metric) : dim_(dim), metric_(metric) {}

void BruteForceIndex::add(const float* data, size_t n) {
  data_.insert(data_.end(), data, data + n * dim_);
  n_ += n;
}

std::vector<Neighbor> BruteForceIndex::search(const float* q, size_t k) const {
  std::priority_queue<Neighbor> heap;
  for (size_t i = 0; i < n_; ++i) {
    float d = distance(metric_, q, data_.data() + i * dim_, dim_);
    if (heap.size() < k) {
      heap.push({d, static_cast<uint32_t>(i)});
    } else if (d < heap.top().dist) {
      heap.pop();
      heap.push({d, static_cast<uint32_t>(i)});
    }
  }
  std::vector<Neighbor> out(heap.size());
  for (size_t i = out.size(); i-- > 0;) {
    out[i] = heap.top();
    heap.pop();
  }
  return out;
}

}  // namespace vecsearch
