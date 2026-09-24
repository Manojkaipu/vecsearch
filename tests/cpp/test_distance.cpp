#include <gtest/gtest.h>

#include "test_util.h"
#include "vecsearch/distance.h"

using namespace vecsearch;

// Covers the 16-wide, 8-wide and scalar tail paths.
class DistanceTest : public ::testing::TestWithParam<size_t> {};

TEST_P(DistanceTest, MatchesNaive) {
  size_t d = GetParam();
  auto a = vstest::random_data(1, d, 1), b = vstest::random_data(1, d, 2);
  double l2 = 0, ip = 0;
  for (size_t i = 0; i < d; ++i) {
    l2 += double(a[i] - b[i]) * (a[i] - b[i]);
    ip += double(a[i]) * b[i];
  }
  EXPECT_NEAR(l2_sq(a.data(), b.data(), d), l2, 1e-3 * (1 + l2));
  EXPECT_NEAR(dot(a.data(), b.data(), d), ip, 1e-3 * (1 + std::abs(ip)));
  EXPECT_NEAR(distance(Metric::InnerProduct, a.data(), b.data(), d), 1.0 - ip, 1e-3 * (1 + std::abs(ip)));
}

INSTANTIATE_TEST_SUITE_P(Dims, DistanceTest, ::testing::Values(1, 3, 7, 8, 9, 15, 16, 17, 31, 128, 384, 385));

TEST(Distance, SelfDistanceIsZero) {
  auto a = vstest::random_data(1, 384, 3);
  EXPECT_FLOAT_EQ(l2_sq(a.data(), a.data(), 384), 0.f);
}
