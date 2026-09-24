// Distance kernels. Smaller is always closer:
//   L2           squared Euclidean (no sqrt)
//   InnerProduct 1 - <a, b>, i.e. cosine distance on normalized vectors
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define VECSEARCH_AVX2 1
#endif

namespace vecsearch {

enum class Metric : uint32_t { L2 = 0, InnerProduct = 1 };

#ifdef VECSEARCH_AVX2
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
#endif

inline float l2_sq(const float* a, const float* b, size_t d) {
  size_t i = 0;
  float r = 0.f;
#ifdef VECSEARCH_AVX2
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (; i + 16 <= d; i += 16) {  // two accumulators hide FMA latency
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  for (; i + 8 <= d; i += 8) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
  }
  r = hsum256(_mm256_add_ps(acc0, acc1));
#endif
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    r += t * t;
  }
  return r;
}

inline float dot(const float* a, const float* b, size_t d) {
  size_t i = 0;
  float r = 0.f;
#ifdef VECSEARCH_AVX2
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (; i + 16 <= d; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 8 <= d; i += 8)
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  r = hsum256(_mm256_add_ps(acc0, acc1));
#endif
  for (; i < d; ++i) r += a[i] * b[i];
  return r;
}

inline float distance(Metric m, const float* a, const float* b, size_t d) {
  return m == Metric::L2 ? l2_sq(a, b, d) : 1.0f - dot(a, b, d);
}

struct Neighbor {
  float dist;
  uint32_t id;
  bool operator<(const Neighbor& o) const { return dist < o.dist; }
  bool operator>(const Neighbor& o) const { return dist > o.dist; }
};

}  // namespace vecsearch
