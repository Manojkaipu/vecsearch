#include "vecsearch/brute_force.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <thread>

namespace vecsearch {

namespace {

constexpr size_t kQueryBlock = 4;

// Distances from 4 queries (consecutive rows of q) to one vector x, loading x once. Same
// accumulators and order as l2_sq/dot, so each result equals distance(m, q_j, x, d).
template <bool L2>
void distances4(const float* q, const float* x, size_t d, float* out) {
  size_t i = 0;
#ifdef VECSEARCH_AVX2
  __m256 a0[kQueryBlock], a1[kQueryBlock];
  for (size_t j = 0; j < kQueryBlock; ++j) a0[j] = a1[j] = _mm256_setzero_ps();
  for (; i + 16 <= d; i += 16) {
    const __m256 x0 = _mm256_loadu_ps(x + i), x1 = _mm256_loadu_ps(x + i + 8);
    for (size_t j = 0; j < kQueryBlock; ++j) {
      __m256 q0 = _mm256_loadu_ps(q + j * d + i), q1 = _mm256_loadu_ps(q + j * d + i + 8);
      if (L2) {
        q0 = _mm256_sub_ps(q0, x0);
        q1 = _mm256_sub_ps(q1, x1);
        a0[j] = _mm256_fmadd_ps(q0, q0, a0[j]);
        a1[j] = _mm256_fmadd_ps(q1, q1, a1[j]);
      } else {
        a0[j] = _mm256_fmadd_ps(q0, x0, a0[j]);
        a1[j] = _mm256_fmadd_ps(q1, x1, a1[j]);
      }
    }
  }
  for (; i + 8 <= d; i += 8) {
    const __m256 x0 = _mm256_loadu_ps(x + i);
    for (size_t j = 0; j < kQueryBlock; ++j) {
      __m256 q0 = _mm256_loadu_ps(q + j * d + i);
      if (L2) {
        q0 = _mm256_sub_ps(q0, x0);
        a0[j] = _mm256_fmadd_ps(q0, q0, a0[j]);
      } else {
        a0[j] = _mm256_fmadd_ps(q0, x0, a0[j]);
      }
    }
  }
#endif
  for (size_t j = 0; j < kQueryBlock; ++j) {
    const float* qj = q + j * d;
    float r = 0.f;
#ifdef VECSEARCH_AVX2
    r = hsum256(_mm256_add_ps(a0[j], a1[j]));
#endif
    for (size_t t = i; t < d; ++t) {
      if (L2) {
        float e = qj[t] - x[t];
        r += e * e;
      } else {
        r += qj[t] * x[t];
      }
    }
    out[j] = L2 ? r : 1.0f - r;
  }
}

void distances(Metric m, const float* q, size_t nb, const float* x, size_t d, float* out) {
  if (nb == kQueryBlock) {
    m == Metric::L2 ? distances4<true>(q, x, d, out) : distances4<false>(q, x, d, out);
  } else {
    for (size_t j = 0; j < nb; ++j) out[j] = distance(m, q + j * d, x, d);
  }
}

// Bounded max-heap in h[0..n): keeps the k smallest distances seen.
inline void push(Neighbor* h, size_t& n, size_t k, float d, uint32_t id) {
  if (n < k) {
    h[n++] = {d, id};
    std::push_heap(h, h + n);
  } else if (d < h[0].dist) {
    std::pop_heap(h, h + k);
    h[k - 1] = {d, id};
    std::push_heap(h, h + k);
  }
}

}  // namespace

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

void BruteForceIndex::search_batch(const float* queries, size_t nq, size_t k, uint32_t* out_ids,
                                   float* out_dists, int num_threads) const {
  if (nq == 0 || k == 0) return;
  const size_t d = dim_;
  // Each thread scans its own slice of the data, so a single query is still parallel.
  const size_t nt = std::max<size_t>(1, std::min<size_t>(std::max(num_threads, 1), n_));
  // A tile of ~256 KB stays in L2 while every query block passes over it.
  const size_t tile = std::max<size_t>(16, (size_t(256) << 10) / (d * sizeof(float) + 1));
  std::vector<Neighbor> heaps(nt * nq * k);
  std::vector<size_t> sizes(nt * nq, 0);

  auto scan = [&](size_t t) {
    const size_t lo = n_ * t / nt, hi = n_ * (t + 1) / nt;
    Neighbor* h = heaps.data() + t * nq * k;
    size_t* sz = sizes.data() + t * nq;
    float dist[kQueryBlock];
    for (size_t b0 = lo; b0 < hi; b0 += tile) {
      const size_t b1 = std::min(hi, b0 + tile);
      for (size_t q0 = 0; q0 < nq; q0 += kQueryBlock) {
        const size_t nb = std::min(kQueryBlock, nq - q0);
        for (size_t i = b0; i < b1; ++i) {
          distances(metric_, queries + q0 * d, nb, data_.data() + i * d, d, dist);
          for (size_t j = 0; j < nb; ++j)
            push(h + (q0 + j) * k, sz[q0 + j], k, dist[j], static_cast<uint32_t>(i));
        }
      }
    }
  };
  if (nt == 1) {
    scan(0);
  } else {
    std::vector<std::thread> ts;
    for (size_t t = 0; t < nt; ++t) ts.emplace_back(scan, t);
    for (auto& t : ts) t.join();
  }

  // Merge the per-thread top-k lists; ties go to the lower id, whatever the thread count.
  std::vector<Neighbor> all;
  for (size_t q = 0; q < nq; ++q) {
    all.clear();
    for (size_t t = 0; t < nt; ++t) {
      const Neighbor* h = heaps.data() + (t * nq + q) * k;
      all.insert(all.end(), h, h + sizes[t * nq + q]);
    }
    const size_t m = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + m, all.end(), [](const Neighbor& a, const Neighbor& b) {
      return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
    });
    for (size_t j = 0; j < k; ++j) {
      out_ids[q * k + j] = j < m ? all[j].id : kNone;
      out_dists[q * k + j] = j < m ? all[j].dist : INFINITY;
    }
  }
}

}  // namespace vecsearch
