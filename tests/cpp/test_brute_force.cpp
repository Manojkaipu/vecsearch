#include <gtest/gtest.h>

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

TEST(BruteForce, ResultsSortedAndKClamped) {
  auto data = vstest::random_data(50, 16, 7);
  BruteForceIndex bf(16);
  bf.add(data.data(), 50);
  auto r = bf.search(data.data(), 100);
  ASSERT_EQ(r.size(), 50u);
  EXPECT_EQ(r[0].id, 0u);
  for (size_t i = 1; i < r.size(); ++i) EXPECT_LE(r[i - 1].dist, r[i].dist);
}
