// HNSW graph index (Malkov & Yashunin, 2016).
//
// Layer 0 holds every vector with up to 2*M links; a node reaches level l with
// probability M^-l. Search descends greedily through the upper layers, then
// runs a beam search of width ef on layer 0.
#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "vecsearch/distance.h"

namespace vecsearch {

struct HNSWParams {
  size_t M = 16;
  size_t ef_construction = 200;
  uint64_t seed = 42;
  bool use_heuristic = true;  // false: keep the M closest (ablation only)
};

class VisitedPool;  // defined in hnsw.cpp

class HNSWIndex {
 public:
  static constexpr uint32_t kNone = 0xFFFFFFFFu;

  HNSWIndex(size_t dim, size_t max_elements, Metric metric = Metric::L2,
            HNSWParams params = {});
  ~HNSWIndex();
  HNSWIndex(const HNSWIndex&) = delete;
  HNSWIndex& operator=(const HNSWIndex&) = delete;

  // Appends n row-major vectors; ids continue from size(). A parallel build is
  // not deterministic. Must not run concurrently with search().
  void add(const float* data, size_t n, int num_threads = 1);

  // Sorted by ascending distance.
  std::vector<Neighbor> search(const float* query, size_t k, size_t ef) const;

  // Outputs are nq x k; unfilled slots get id=kNone, dist=+inf.
  void search_batch(const float* queries, size_t nq, size_t k, size_t ef,
                    uint32_t* out_ids, float* out_dists, int num_threads = 1) const;

  void save(const std::string& path) const;
  static std::unique_ptr<HNSWIndex> load(const std::string& path, size_t extra_capacity = 0);

  size_t size() const { return count_.load(); }
  size_t dim() const { return dim_; }
  size_t capacity() const { return max_elements_; }
  Metric metric() const { return metric_; }
  size_t M() const { return M_; }
  int max_level() const { return max_level_; }
  uint32_t entry_point() const { return entry_point_; }

  int level_of(uint32_t id) const { return levels_[id]; }
  std::vector<uint32_t> neighbors(uint32_t id, int level) const;
  const float* vector(uint32_t id) const { return data_.data() + size_t(id) * dim_; }

 private:
  float dist(const float* a, const float* b) const { return distance(metric_, a, b, dim_); }
  uint32_t* links(uint32_t id, int level);
  const uint32_t* links(uint32_t id, int level) const;
  size_t copy_links(uint32_t id, int level, uint32_t* buf, bool lock) const;

  int random_level();
  void insert(uint32_t id);
  uint32_t greedy_descend(const float* q, uint32_t ep, int from_level, int to_level,
                          bool lock) const;
  std::vector<Neighbor> search_layer(const float* q, uint32_t ep, size_t ef, int level,
                                     bool lock) const;
  std::vector<Neighbor> select_neighbors(const std::vector<Neighbor>& sorted_cands,
                                         size_t M) const;

  size_t dim_, max_elements_;
  Metric metric_;
  size_t M_, M0_, ef_construction_;
  bool use_heuristic_;
  double level_mult_;

  std::vector<float> data_;
  std::vector<uint32_t> level0_;              // per node: [count, ids...], stride M0+1
  std::vector<std::vector<uint32_t>> upper_;  // per node: levels 1..L, stride M+1
  std::vector<int> levels_;

  std::atomic<size_t> count_{0};
  int max_level_ = -1;
  uint32_t entry_point_ = kNone;

  std::mutex global_mtx_;                     // entry_point_, max_level_
  std::unique_ptr<std::mutex[]> node_locks_;  // per-node link lists
  std::mutex rng_mtx_;
  std::mt19937_64 rng_;
  std::unique_ptr<VisitedPool> visited_;
};

}  // namespace vecsearch
