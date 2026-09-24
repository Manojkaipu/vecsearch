#pragma once
#include <random>
#include <unordered_set>
#include <vector>

#include "vecsearch/brute_force.h"
#include "vecsearch/hnsw.h"

namespace vstest {

inline std::vector<float> random_data(size_t n, size_t d, uint64_t seed, bool normalize = false) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> v(n * d);
  for (auto& x : v) x = g(rng);
  if (normalize) {
    for (size_t i = 0; i < n; ++i) {
      float s = 0;
      for (size_t j = 0; j < d; ++j) s += v[i * d + j] * v[i * d + j];
      s = std::sqrt(s);
      for (size_t j = 0; j < d; ++j) v[i * d + j] /= s;
    }
  }
  return v;
}

// Mean recall@k of HNSW vs exact brute force.
inline double recall_at_k(const vecsearch::HNSWIndex& h, const vecsearch::BruteForceIndex& bf,
                          const std::vector<float>& queries, size_t nq, size_t k, size_t ef) {
  const size_t d = h.dim();
  double hit = 0;
  for (size_t i = 0; i < nq; ++i) {
    auto truth = bf.search(queries.data() + i * d, k);
    auto got = h.search(queries.data() + i * d, k, ef);
    std::unordered_set<uint32_t> t;
    for (auto& n : truth) t.insert(n.id);
    for (auto& n : got) hit += t.count(n.id);
  }
  return hit / double(nq * k);
}

}  // namespace vstest
