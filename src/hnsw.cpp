#include "vecsearch/hnsw.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>

namespace vecsearch {

// Epoch-tagged visited set: node i is visited iff mark[i] == cur, so reset is
// ++cur and the array is only cleared on wraparound. Pooled across searches.
struct VisitedList {
  std::vector<uint16_t> mark;
  uint16_t cur = 0;
  explicit VisitedList(size_t n) : mark(n, 0) {}
  void reset() {
    if (++cur == 0) {
      std::fill(mark.begin(), mark.end(), 0);
      cur = 1;
    }
  }
  bool check_and_mark(uint32_t id) {
    if (mark[id] == cur) return true;
    mark[id] = cur;
    return false;
  }
};

class VisitedPool {
 public:
  explicit VisitedPool(size_t n) : n_(n) {}
  std::unique_ptr<VisitedList> get() {
    std::unique_ptr<VisitedList> v;
    {
      std::lock_guard<std::mutex> g(m_);
      if (!pool_.empty()) {
        v = std::move(pool_.back());
        pool_.pop_back();
      }
    }
    if (!v) v = std::make_unique<VisitedList>(n_);
    v->reset();
    return v;
  }
  void put(std::unique_ptr<VisitedList> v) {
    std::lock_guard<std::mutex> g(m_);
    pool_.push_back(std::move(v));
  }

 private:
  size_t n_;
  std::mutex m_;
  std::vector<std::unique_ptr<VisitedList>> pool_;
};

struct VisitedGuard {
  VisitedPool& pool;
  std::unique_ptr<VisitedList> list;
  explicit VisitedGuard(VisitedPool& p) : pool(p), list(p.get()) {}
  ~VisitedGuard() { pool.put(std::move(list)); }
};

static void parallel_for(size_t begin, size_t end, int num_threads,
                         const std::function<void(size_t)>& fn) {
  if (num_threads <= 1 || end - begin < 2) {
    for (size_t i = begin; i < end; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{begin};
  std::exception_ptr err;
  std::mutex err_mtx;
  std::vector<std::thread> ts;
  for (int t = 0; t < num_threads; ++t) {
    ts.emplace_back([&] {
      while (true) {
        size_t i = next.fetch_add(1);
        if (i >= end) break;
        try {
          fn(i);
        } catch (...) {
          std::lock_guard<std::mutex> g(err_mtx);
          if (!err) err = std::current_exception();
          next = end;
        }
      }
    });
  }
  for (auto& t : ts) t.join();
  if (err) std::rethrow_exception(err);
}

HNSWIndex::HNSWIndex(size_t dim, size_t max_elements, Metric metric, HNSWParams p)
    : dim_(dim),
      max_elements_(max_elements),
      metric_(metric),
      M_(std::max<size_t>(p.M, 2)),
      M0_(2 * std::max<size_t>(p.M, 2)),
      ef_construction_(std::max(p.ef_construction, std::max<size_t>(p.M, 2))),
      use_heuristic_(p.use_heuristic),
      level_mult_(1.0 / std::log(double(std::max<size_t>(p.M, 2)))),
      data_(max_elements * dim),
      level0_(max_elements * (2 * std::max<size_t>(p.M, 2) + 1), 0),
      upper_(max_elements),
      levels_(max_elements, 0),
      node_locks_(new std::mutex[max_elements]),
      rng_(p.seed),
      visited_(std::make_unique<VisitedPool>(max_elements)) {
  if (dim == 0) throw std::invalid_argument("dim must be > 0");
}

HNSWIndex::~HNSWIndex() = default;

uint32_t* HNSWIndex::links(uint32_t id, int level) {
  if (level == 0) return level0_.data() + size_t(id) * (M0_ + 1);
  return upper_[id].data() + size_t(level - 1) * (M_ + 1);
}
const uint32_t* HNSWIndex::links(uint32_t id, int level) const {
  return const_cast<HNSWIndex*>(this)->links(id, level);
}

// During a parallel build other threads rewrite link lists, so copy under the node lock.
size_t HNSWIndex::copy_links(uint32_t id, int level, uint32_t* buf, bool lock) const {
  std::unique_lock<std::mutex> g;
  if (lock) g = std::unique_lock<std::mutex>(node_locks_[id]);
  const uint32_t* ll = links(id, level);
  size_t cnt = ll[0];
  std::memcpy(buf, ll + 1, cnt * sizeof(uint32_t));
  return cnt;
}

std::vector<uint32_t> HNSWIndex::neighbors(uint32_t id, int level) const {
  if (id >= size() || level > levels_[id]) return {};
  std::vector<uint32_t> out(M0_);
  out.resize(copy_links(id, level, out.data(), false));
  return out;
}

// floor(-ln(U) / ln(M)): P(level >= l) = M^-l.
int HNSWIndex::random_level() {
  std::lock_guard<std::mutex> g(rng_mtx_);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = 1.0 - u(rng_);  // (0, 1], avoids log(0)
  return static_cast<int>(-std::log(r) * level_mult_);
}

// Greedy walk (beam width 1) from from_level down to to_level + 1.
uint32_t HNSWIndex::greedy_descend(const float* q, uint32_t ep, int from_level, int to_level,
                                   bool lock) const {
  float cur = dist(q, vector(ep));
  std::vector<uint32_t> buf(M0_);
  for (int lc = from_level; lc > to_level; --lc) {
    bool changed = true;
    while (changed) {
      changed = false;
      size_t cnt = copy_links(ep, lc, buf.data(), lock);
      for (size_t i = 0; i < cnt; ++i) {
        float d = dist(q, vector(buf[i]));
        if (d < cur) {
          cur = d;
          ep = buf[i];
          changed = true;
        }
      }
    }
  }
  return ep;
}

// Algorithm 2. Stops once the nearest unexpanded candidate is farther than the
// worst of the ef results.
std::vector<Neighbor> HNSWIndex::search_layer(const float* q, uint32_t ep, size_t ef, int level,
                                              bool lock) const {
  VisitedGuard vg(*visited_);
  VisitedList& vis = *vg.list;

  std::vector<Neighbor> cand, res;  // min-heap, max-heap
  cand.reserve(ef * 2);
  res.reserve(ef + 1);
  auto minh = std::greater<Neighbor>();

  float d0 = dist(q, vector(ep));
  vis.check_and_mark(ep);
  cand.push_back({d0, ep});
  res.push_back({d0, ep});

  std::vector<uint32_t> buf(M0_);
  while (!cand.empty()) {
    Neighbor c = cand.front();
    if (c.dist > res.front().dist && res.size() >= ef) break;
    std::pop_heap(cand.begin(), cand.end(), minh);
    cand.pop_back();

    size_t cnt = copy_links(c.id, level, buf.data(), lock);
#if defined(__GNUC__)
    for (size_t i = 0; i < cnt; ++i) __builtin_prefetch(vector(buf[i]));
#endif
    for (size_t i = 0; i < cnt; ++i) {
      uint32_t nb = buf[i];
      if (vis.check_and_mark(nb)) continue;
      float d = dist(q, vector(nb));
      if (res.size() < ef || d < res.front().dist) {
        cand.push_back({d, nb});
        std::push_heap(cand.begin(), cand.end(), minh);
        res.push_back({d, nb});
        std::push_heap(res.begin(), res.end());
        if (res.size() > ef) {
          std::pop_heap(res.begin(), res.end());
          res.pop_back();
        }
      }
    }
  }
  std::sort_heap(res.begin(), res.end());
  return res;
}

// Algorithm 4: drop a candidate that is closer to an already selected neighbor
// than to the base node, so links spread out instead of piling into one cluster.
std::vector<Neighbor> HNSWIndex::select_neighbors(const std::vector<Neighbor>& cands,
                                                  size_t M) const {
  if (cands.size() <= M) return cands;
  if (!use_heuristic_) return {cands.begin(), cands.begin() + M};
  std::vector<Neighbor> out;
  out.reserve(M);
  for (const auto& c : cands) {
    if (out.size() >= M) break;
    bool keep = true;
    for (const auto& r : out) {
      if (dist(vector(c.id), vector(r.id)) < c.dist) {
        keep = false;
        break;
      }
    }
    if (keep) out.push_back(c);
  }
  return out;
}

// Algorithm 1.
void HNSWIndex::insert(uint32_t id) {
  const int level = random_level();
  {
    std::lock_guard<std::mutex> g(node_locks_[id]);
    levels_[id] = level;
    if (level > 0) upper_[id].assign(size_t(level) * (M_ + 1), 0);
  }

  // Keep the global lock for the whole insert only if this node becomes the new entry point.
  std::unique_lock<std::mutex> glock(global_mtx_);
  if (entry_point_ == kNone) {
    entry_point_ = id;
    max_level_ = level;
    return;
  }
  const int cur_max = max_level_;
  uint32_t ep = entry_point_;
  if (level <= cur_max) glock.unlock();

  const float* q = vector(id);

  if (level < cur_max) ep = greedy_descend(q, ep, cur_max, level, true);

  for (int lc = std::min(level, cur_max); lc >= 0; --lc) {
    std::vector<Neighbor> cands = search_layer(q, ep, ef_construction_, lc, true);
    ep = cands.front().id;
    std::vector<Neighbor> selected = select_neighbors(cands, M_);

    {
      std::lock_guard<std::mutex> g(node_locks_[id]);
      uint32_t* ll = links(id, lc);
      ll[0] = static_cast<uint32_t>(selected.size());
      for (size_t i = 0; i < selected.size(); ++i) ll[1 + i] = selected[i].id;
    }

    const size_t Mmax = (lc == 0) ? M0_ : M_;
    for (const auto& s : selected) {
      std::lock_guard<std::mutex> g(node_locks_[s.id]);
      uint32_t* ll = links(s.id, lc);
      uint32_t cnt = ll[0];
      if (cnt < Mmax) {
        ll[1 + cnt] = id;
        ll[0] = cnt + 1;
        continue;
      }
      // Full: re-prune old links + the new node with the same heuristic.
      const float* sv = vector(s.id);
      std::vector<Neighbor> all;
      all.reserve(cnt + 1);
      all.push_back({dist(sv, q), id});
      for (uint32_t j = 0; j < cnt; ++j) all.push_back({dist(sv, vector(ll[1 + j])), ll[1 + j]});
      std::sort(all.begin(), all.end());
      std::vector<Neighbor> keep = select_neighbors(all, Mmax);
      ll[0] = static_cast<uint32_t>(keep.size());
      for (size_t j = 0; j < keep.size(); ++j) ll[1 + j] = keep[j].id;
    }
  }

  if (level > cur_max) {  // glock still held in this branch
    entry_point_ = id;
    max_level_ = level;
  }
}

void HNSWIndex::add(const float* data, size_t n, int num_threads) {
  const size_t start = count_.load();
  if (start + n > max_elements_)
    throw std::length_error("HNSWIndex capacity exceeded (" + std::to_string(start + n) + " > " +
                            std::to_string(max_elements_) + ")");
  std::memcpy(data_.data() + start * dim_, data, n * dim_ * sizeof(float));
  count_ = start + n;  // safe: new ids aren't reachable until linked
  parallel_for(start, start + n, num_threads, [this](size_t i) { insert(uint32_t(i)); });
}

// Algorithm 5.
std::vector<Neighbor> HNSWIndex::search(const float* q, size_t k, size_t ef) const {
  if (entry_point_ == kNone || k == 0) return {};
  ef = std::max(ef, k);
  uint32_t ep = greedy_descend(q, entry_point_, max_level_, 0, false);
  std::vector<Neighbor> res = search_layer(q, ep, ef, 0, false);
  if (res.size() > k) res.resize(k);
  return res;
}

void HNSWIndex::search_batch(const float* queries, size_t nq, size_t k, size_t ef,
                             uint32_t* out_ids, float* out_dists, int num_threads) const {
  parallel_for(0, nq, num_threads, [&](size_t i) {
    std::vector<Neighbor> r = search(queries + i * dim_, k, ef);
    for (size_t j = 0; j < k; ++j) {
      out_ids[i * k + j] = j < r.size() ? r[j].id : kNone;
      out_dists[i * k + j] = j < r.size() ? r[j].dist : std::numeric_limits<float>::infinity();
    }
  });
}

// File layout: magic, header, vectors, levels, layer-0 links, upper links.
static const char kMagic[8] = {'V', 'S', 'H', 'N', 'S', 'W', '0', '1'};

template <class T>
static void wr(std::ofstream& f, const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T>
static void rd(std::ifstream& f, T& v) { f.read(reinterpret_cast<char*>(&v), sizeof(T)); }

void HNSWIndex::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  const uint64_t n = size();
  f.write(kMagic, 8);
  wr(f, uint64_t(dim_)); wr(f, n); wr(f, uint32_t(metric_)); wr(f, uint64_t(M_));
  wr(f, uint64_t(ef_construction_)); wr(f, int32_t(max_level_)); wr(f, entry_point_);
  f.write(reinterpret_cast<const char*>(data_.data()), n * dim_ * sizeof(float));
  f.write(reinterpret_cast<const char*>(levels_.data()), n * sizeof(int));
  f.write(reinterpret_cast<const char*>(level0_.data()), n * (M0_ + 1) * sizeof(uint32_t));
  for (uint64_t i = 0; i < n; ++i)
    if (levels_[i] > 0)
      f.write(reinterpret_cast<const char*>(upper_[i].data()), upper_[i].size() * sizeof(uint32_t));
  if (!f) throw std::runtime_error("write failed: " + path);
}

std::unique_ptr<HNSWIndex> HNSWIndex::load(const std::string& path, size_t extra_capacity) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("not a vecsearch HNSW file");
  uint64_t dim, n, M, efc;
  uint32_t metric, ep;
  int32_t maxl;
  rd(f, dim); rd(f, n); rd(f, metric); rd(f, M); rd(f, efc); rd(f, maxl); rd(f, ep);
  HNSWParams p;
  p.M = M;
  p.ef_construction = efc;
  auto idx = std::make_unique<HNSWIndex>(dim, n + extra_capacity, Metric(metric), p);
  f.read(reinterpret_cast<char*>(idx->data_.data()), n * dim * sizeof(float));
  f.read(reinterpret_cast<char*>(idx->levels_.data()), n * sizeof(int));
  f.read(reinterpret_cast<char*>(idx->level0_.data()), n * (idx->M0_ + 1) * sizeof(uint32_t));
  for (uint64_t i = 0; i < n; ++i) {
    if (idx->levels_[i] > 0) {
      idx->upper_[i].resize(size_t(idx->levels_[i]) * (idx->M_ + 1));
      f.read(reinterpret_cast<char*>(idx->upper_[i].data()), idx->upper_[i].size() * sizeof(uint32_t));
    }
  }
  if (!f) throw std::runtime_error("truncated index file: " + path);
  idx->count_ = n;
  idx->max_level_ = maxl;
  idx->entry_point_ = ep;
  return idx;
}

}  // namespace vecsearch
