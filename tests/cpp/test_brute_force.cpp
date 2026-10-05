#include <gtest/gtest.h>

#include <cmath>

#include "test_util.h"

using namespace vecsearch;

TEST(BruteForce, FindsExactNeighborsOnALine) {
  std::vector<float> pts;  // (0,0), (1,0), ..., (9,0)
  for (int i = 0; i < 10; ++i) { pts.push_back(float(i)); pts.push_back(0.f); }
  BruteForceIndex bf(2);
  bf.add(pts.data(), 10);
  float q[2] = {3.2f, 0.f};
  auto r = bf.search(q, 3);
  ASSERT_EQ(r.size(), 3u);
  EXPECT_EQ(r[0].id, 3u);
  EXPECT_EQ(r[1].id, 4u);
  EXPECT_EQ(r[2].id, 2u);
}

TEST(BruteForce, BatchMatchesSingleQuery) {
  // 37 dims exercises the 16-wide, 8-wide and scalar paths; 13 queries leave a partial block.
  const size_t n = 500, d = 37, nq = 13, k = 10;
  auto data = vstest::random_data(n, d, 3);
  auto queries = vstest::random_data(nq, d, 4);
  for (Metric m : {Metric::L2, Metric::InnerProduct}) {
    BruteForceIndex bf(d, m);
    bf.add(data.data(), n);
    for (int threads : {1, 3}) {
      std::vector<uint32_t> ids(nq * k);
      std::vector<float> ds(nq * k);
      bf.search_batch(queries.data(), nq, k, ids.data(), ds.data(), threads);
      for (size_t q = 0; q < nq; ++q) {
        auto r = bf.search(queries.data() + q * d, k);
        for (size_t j = 0; j < k; ++j) {
          EXPECT_EQ(ids[q * k + j], r[j].id) << "query " << q << " rank " << j;
          EXPECT_FLOAT_EQ(ds[q * k + j], r[j].dist);
        }
      }
    }
  }
}

TEST(BruteForce, BatchPadsMissingResults) {
  auto data = vstest::random_data(5, 8, 9);
  BruteForceIndex bf(8);
  bf.add(data.data(), 5);
  std::vector<uint32_t> ids(7);
  std::vector<float> ds(7);
  bf.search_batch(data.data(), 1, 7, ids.data(), ds.data(), 4);
  EXPECT_EQ(ids[0], 0u);
  EXPECT_EQ(ids[5], BruteForceIndex::kNone);
  EXPECT_TRUE(std::isinf(ds[6]));
}

TEST(BruteForce, ResultsSortedAndKClamped) {
  auto data = vstest::random_data(50, 16, 7);
  BruteForceIndex bf(16);
  bf.add(data.data(), 50);
  auto r = bf.search(data.data(), 100);
  ASSERT_EQ(r.size(), 50u);
  EXPECT_EQ(r[0].id, 0u);
  for (size_t i = 1; i < r.size(); ++i) EXPECT_LE(r[i - 1].dist, r[i].dist);
}
