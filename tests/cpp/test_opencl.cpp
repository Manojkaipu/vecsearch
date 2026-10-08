// OpenCLBruteForceIndex against the CPU BruteForceIndex. Every test skips if no OpenCL device is
// visible; CI runs them on POCL, which executes the same kernels on a CPU.
//
// The comparison is the one tests/python/test_gpu.py uses for CUDA: distances within rtol 1e-4 /
// atol 1e-3 of the CPU's (the GPU adds in a different order, and tiled L2 uses |q|^2+|x|^2-2q.x),
// at least 99% of ids equal (near-ties may swap places), and every row sorted. On top of that,
// each returned id's distance is recomputed on the CPU and must match the one the device reported.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "test_util.h"
#include "vecsearch/brute_force.h"
#include "vecsearch/opencl_brute_force.h"

using namespace vecsearch;

namespace {

constexpr int kAuto = -1, kFallback = 0;  // OpenCLBruteForceIndex's `subgroups` argument

struct Result {
  std::vector<uint32_t> ids;
  std::vector<float> dists;
};

Result cpu_search(const std::vector<float>& x, size_t n, const std::vector<float>& q, size_t nq,
                  size_t dim, Metric metric, size_t k) {
  BruteForceIndex cpu(dim, metric);
  cpu.add(x.data(), n);
  Result r{std::vector<uint32_t>(nq * k), std::vector<float>(nq * k)};
  cpu.search_batch(q.data(), nq, k, r.ids.data(), r.dists.data(), 2);
  return r;
}

Result gpu_search(OpenCLBruteForceIndex& gpu, const std::vector<float>& q, size_t nq, size_t k,
                  GpuMethod method) {
  Result r{std::vector<uint32_t>(nq * k), std::vector<float>(nq * k)};
  gpu.search(q.data(), nq, k, r.ids.data(), r.dists.data(), method);
  return r;
}

// min_agreement: share of ids that must equal the CPU's. Tiny dimensions have many near-ties
// whose order is arbitrary, so those cases lower it; the true-distance check below doesn't depend on it.
void expect_matches(const Result& g, const Result& c, const std::vector<float>& x, size_t n,
                    const std::vector<float>& q, size_t nq, size_t dim, Metric metric, size_t k,
                    double min_agreement) {
  size_t same = 0;
  for (size_t i = 0; i < nq; ++i) {
    for (size_t j = 0; j < k; ++j) {
      const size_t p = i * k + j;
      if (c.ids[p] == BruteForceIndex::kNone) {  // fewer than k vectors: padded with kNone / inf
        ASSERT_EQ(g.ids[p], OpenCLBruteForceIndex::kNone) << "query " << i << " rank " << j;
        ASSERT_TRUE(std::isinf(g.dists[p]));
        ++same;
        continue;
      }
      ASSERT_NEAR(g.dists[p], c.dists[p], 1e-3 + 1e-4 * std::fabs(c.dists[p])) << "query " << i << " rank " << j;
      if (j > 0) ASSERT_LE(g.dists[p - 1], g.dists[p]) << "query " << i << " not sorted at rank " << j;
      // Whatever id came back, it must be a real vector whose CPU distance is the one reported,
      // and it must not repeat within the row.
      ASSERT_LT(g.ids[p], n) << "query " << i << " rank " << j;
      const float truth = distance(metric, q.data() + i * dim, x.data() + size_t(g.ids[p]) * dim, dim);
      ASSERT_NEAR(g.dists[p], truth, 1e-3 + 1e-4 * std::fabs(truth)) << "query " << i << " rank " << j;
      for (size_t jj = 0; jj < j; ++jj) ASSERT_NE(g.ids[i * k + jj], g.ids[p]) << "query " << i << " repeats an id";
      same += g.ids[p] == c.ids[p];
    }
  }
  EXPECT_GT(double(same) / double(nq * k), min_agreement);
}

// Builds a CPU index and an OpenCL index over the same data, searches both, compares.
void check(size_t n, size_t dim, size_t nq, size_t k, Metric metric, GpuMethod method,
           int subgroups = kAuto, size_t add_split = 0, double min_agreement = 0.99) {
  const bool unit = metric == Metric::InnerProduct;
  const auto x = vstest::random_data(n, dim, 1, unit), q = vstest::random_data(nq, dim, 2, unit);
  OpenCLBruteForceIndex gpu(dim, metric, 0, subgroups);
  if (add_split && add_split < n) {  // two adds: the device buffer grows and keeps the first part
    gpu.add(x.data(), add_split);
    gpu.add(x.data() + add_split * dim, n - add_split);
  } else {
    gpu.add(x.data(), n);
  }
  ASSERT_EQ(gpu.size(), n);
  expect_matches(gpu_search(gpu, q, nq, k, method), cpu_search(x, n, q, nq, dim, metric, k), x, n, q, nq,
                 dim, metric, k, min_agreement);
}

#define REQUIRE_OPENCL()                                          \
  do {                                                            \
    if (!opencl_available()) GTEST_SKIP() << "no OpenCL device";  \
  } while (0)

class OpenCLMatchesCpu
    : public testing::TestWithParam<std::tuple<Metric, GpuMethod, size_t, size_t, int>> {};

TEST_P(OpenCLMatchesCpu, TopK) {
  REQUIRE_OPENCL();
  const auto [metric, method, dim, nq, subgroups] = GetParam();
  check(3000, dim, nq, 10, metric, method, subgroups, 1000);
}

INSTANTIATE_TEST_SUITE_P(
    Kernels, OpenCLMatchesCpu,
    testing::Combine(testing::Values(Metric::L2, Metric::InnerProduct),
                     testing::Values(GpuMethod::Naive, GpuMethod::Skinny, GpuMethod::Tiled, GpuMethod::Auto),
                     testing::Values(size_t(37), size_t(128)),  // 37 is padded to 40 on the device
                     testing::Values(size_t(1), size_t(5), size_t(300)),
                     testing::Values(kAuto, kFallback)));

}  // namespace

TEST(OpenCL, ListsDevices) {
  REQUIRE_OPENCL();
  const auto devs = opencl_devices();
  ASSERT_FALSE(devs.empty());
  for (const auto& d : devs) {
    EXPECT_FALSE(d.name.empty());
    EXPECT_GT(d.max_alloc, size_t(0));
  }
  OpenCLBruteForceIndex idx(8, Metric::L2, 0);
  EXPECT_EQ(idx.device_info().name, devs[0].name);
  EXPECT_EQ(idx.uses_subgroups(), devs[0].subgroups);  // the first index probes the compiler too
  OpenCLBruteForceIndex fallback(8, Metric::L2, 0, kFallback);
  EXPECT_FALSE(fallback.uses_subgroups());
  EXPECT_THROW(OpenCLBruteForceIndex(8, Metric::L2, int(devs.size())), std::invalid_argument);
}

TEST(OpenCL, KLargerThanDataset) {
  REQUIRE_OPENCL();
  for (GpuMethod m : {GpuMethod::Naive, GpuMethod::Skinny, GpuMethod::Tiled}) {
    check(3, 8, 2, 5, Metric::L2, m);   // 3 vectors, k = 5: two results are missing
    check(1, 8, 3, 10, Metric::L2, m);  // a single vector
    check(20, 8, 4, 128, Metric::InnerProduct, m);
  }
  OpenCLBruteForceIndex empty(8);
  const std::vector<float> q(2 * 8, 0.f);
  const Result r = gpu_search(empty, q, 2, 5, GpuMethod::Auto);
  for (uint32_t id : r.ids) EXPECT_EQ(id, OpenCLBruteForceIndex::kNone);
  for (float d : r.dists) EXPECT_TRUE(std::isinf(d));
}

TEST(OpenCL, BatchSizeOne) {
  REQUIRE_OPENCL();
  for (GpuMethod m : {GpuMethod::Naive, GpuMethod::Skinny, GpuMethod::Tiled, GpuMethod::Auto})
    for (Metric metric : {Metric::L2, Metric::InnerProduct}) check(5000, 64, 1, 10, metric, m);
}

TEST(OpenCL, SizesThatDontDivideEvenlyIntoTiles) {
  REQUIRE_OPENCL();
  // 128 is the tile edge: sizes just under, at and over it, in both the query and data directions.
  for (size_t n : {size_t(127), size_t(128), size_t(129), size_t(255), size_t(257), size_t(1000)})
    for (size_t nq : {size_t(1), size_t(127), size_t(128), size_t(129), size_t(257)})
      for (size_t dim : {size_t(1), size_t(7), size_t(9), size_t(130)})
        check(n, dim, nq, 7, Metric::L2, GpuMethod::Tiled, kAuto, 0, 0.9);  // tiny dims: many near-ties
  for (size_t n : {size_t(1), size_t(9), size_t(257)})  // the other kernels, at a few of the same sizes
    for (GpuMethod m : {GpuMethod::Naive, GpuMethod::Skinny}) check(n, 13, 9, 4, Metric::L2, m, kAuto, 0, 0.9);
}

TEST(OpenCL, EveryKBucket) {
  REQUIRE_OPENCL();
  // 16/32/64/128 are the select kernel's list sizes; 17, 33 and 65 are the first k past each.
  for (size_t k : {size_t(1), size_t(16), size_t(17), size_t(32), size_t(33), size_t(64), size_t(65), size_t(100), size_t(128)})
    check(20000, 64, 40, k, Metric::L2, GpuMethod::Auto);
  OpenCLBruteForceIndex gpu(8);
  const std::vector<float> q(8, 0.f);
  std::vector<uint32_t> ids(129);
  std::vector<float> d(129);
  gpu.add(q.data(), 1);
  EXPECT_THROW(gpu.search(q.data(), 1, 129, ids.data(), d.data()), std::invalid_argument);
}

TEST(OpenCL, VeryLargeBatchRunsInRoundsAndChunks) {
  REQUIRE_OPENCL();
  // 5000 queries are two rounds of up to 4096, and 70,000 vectors are split into chunks inside each.
  check(70000, 16, 5000, 10, Metric::L2, GpuMethod::Auto);
  check(70000, 16, 5000, 10, Metric::InnerProduct, GpuMethod::Tiled);
}

TEST(OpenCL, BatchSizeSweepAgreesWithEachOther) {
  REQUIRE_OPENCL();
  // The same queries in different batch sizes (so different kernels and chunkings) give one answer.
  const size_t n = 30000, dim = 48, k = 10;
  const auto x = vstest::random_data(n, dim, 11), q = vstest::random_data(300, dim, 12);
  OpenCLBruteForceIndex gpu(dim);
  gpu.add(x.data(), n);
  const Result whole = gpu_search(gpu, q, 300, k, GpuMethod::Auto);
  for (size_t b : {size_t(1), size_t(7), size_t(8), size_t(16), size_t(17), size_t(100)}) {
    const std::vector<float> part(q.begin(), q.begin() + b * dim);
    const Result r = gpu_search(gpu, part, b, k, GpuMethod::Auto);
    for (size_t i = 0; i < b * k; ++i) ASSERT_NEAR(r.dists[i], whole.dists[i], 1e-3) << "batch " << b << " slot " << i;
  }
}

#ifndef _WIN32
TEST(OpenCL, PagedStorageWhenOneAllocationIsLimited) {
  REQUIRE_OPENCL();
  // 40,000 x 128 floats are 20 MB; a 4 MB allocation limit makes that 5 pages of 8,192 rows,
  // with chunks that stop at page ends and ids that continue across pages.
  setenv("VECSEARCH_OPENCL_MAX_ALLOC_MB", "4", 1);
  for (GpuMethod m : {GpuMethod::Skinny, GpuMethod::Tiled, GpuMethod::Naive}) {
    check(40000, 128, 20, 10, Metric::L2, m, kAuto, 15000);
    check(40000, 128, 300, 10, Metric::InnerProduct, m, kAuto, 100);
  }
  unsetenv("VECSEARCH_OPENCL_MAX_ALLOC_MB");
}
#endif
