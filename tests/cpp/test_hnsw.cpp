#include <gtest/gtest.h>

#include <cstdio>
#include <set>

#include "test_util.h"

using namespace vecsearch;

namespace {
constexpr size_t kD = 32, kN = 5000, kQ = 200, kK = 10;
}

TEST(HNSW, EmptyIndexReturnsNothing) {
  HNSWIndex h(kD, 10);
  auto q = vstest::random_data(1, kD, 1);
  EXPECT_TRUE(h.search(q.data(), 5, 10).empty());
}

TEST(HNSW, SingleElement) {
  HNSWIndex h(kD, 10);
  auto x = vstest::random_data(1, kD, 1);
  h.add(x.data(), 1);
  auto r = h.search(x.data(), 5, 10);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].id, 0u);
  EXPECT_NEAR(r[0].dist, 0.f, 1e-5);
}

TEST(HNSW, KLargerThanNReturnsAll) {
  HNSWIndex h(kD, 100);
  auto x = vstest::random_data(20, kD, 2);
  h.add(x.data(), 20);
  EXPECT_EQ(h.search(x.data(), 50, 50).size(), 20u);
}

TEST(HNSW, CapacityExceededThrows) {
  HNSWIndex h(kD, 5);
  auto x = vstest::random_data(6, kD, 3);
  EXPECT_THROW(h.add(x.data(), 6), std::length_error);
}

TEST(HNSW, HighRecallL2) {
  auto data = vstest::random_data(kN, kD, 10);
  auto qs = vstest::random_data(kQ, kD, 11);
  HNSWIndex h(kD, kN, Metric::L2, {16, 200, 42});
  h.add(data.data(), kN);
  BruteForceIndex bf(kD);
  bf.add(data.data(), kN);
  double r = vstest::recall_at_k(h, bf, qs, kQ, kK, 100);
  EXPECT_GE(r, 0.95) << "recall@10 = " << r;
}

TEST(HNSW, HighRecallInnerProduct) {
  auto data = vstest::random_data(kN, kD, 12, true);
  auto qs = vstest::random_data(kQ, kD, 13, true);
  HNSWIndex h(kD, kN, Metric::InnerProduct);
  h.add(data.data(), kN);
  BruteForceIndex bf(kD, Metric::InnerProduct);
  bf.add(data.data(), kN);
  EXPECT_GE(vstest::recall_at_k(h, bf, qs, kQ, kK, 100), 0.95);
}

TEST(HNSW, RecallIncreasesWithEf) {
  auto data = vstest::random_data(kN, kD, 14);
  auto qs = vstest::random_data(kQ, kD, 15);
  HNSWIndex h(kD, kN, Metric::L2, {8, 100, 1});
  h.add(data.data(), kN);
  BruteForceIndex bf(kD);
  bf.add(data.data(), kN);
  double lo = vstest::recall_at_k(h, bf, qs, kQ, kK, 10);
  double hi = vstest::recall_at_k(h, bf, qs, kQ, kK, 200);
  EXPECT_LT(lo, hi);
}

TEST(HNSW, GraphInvariants) {
  auto data = vstest::random_data(kN, kD, 16);
  HNSWIndex h(kD, kN, Metric::L2, {12, 100, 7});
  h.add(data.data(), kN);
  EXPECT_EQ(h.level_of(h.entry_point()), h.max_level());
  size_t isolated = 0;
  for (uint32_t i = 0; i < kN; ++i) {
    EXPECT_LE(h.level_of(i), h.max_level());
    for (int l = 0; l <= h.level_of(i); ++l) {
      auto nb = h.neighbors(i, l);
      EXPECT_LE(nb.size(), l == 0 ? 2 * h.M() : h.M());
      std::set<uint32_t> uniq(nb.begin(), nb.end());
      EXPECT_EQ(uniq.size(), nb.size()) << "duplicate link at node " << i;
      EXPECT_EQ(uniq.count(i), 0u) << "self loop at node " << i;
      for (uint32_t j : nb) EXPECT_GE(h.level_of(j), l) << "link to node absent from layer";
    }
    if (h.neighbors(i, 0).empty()) ++isolated;
  }
  EXPECT_EQ(isolated, 0u);
}

TEST(HNSW, LevelDistributionIsGeometric) {
  auto data = vstest::random_data(kN, 8, 17);
  HNSWIndex h(8, kN, Metric::L2, {16, 50, 3});
  h.add(data.data(), kN);
  size_t upper = 0;
  for (uint32_t i = 0; i < kN; ++i) upper += h.level_of(i) > 0;
  double frac = double(upper) / kN;  // expected 1/M = 0.0625
  EXPECT_GT(frac, 0.04);
  EXPECT_LT(frac, 0.09);
}

TEST(HNSW, SaveLoadRoundTrip) {
  auto data = vstest::random_data(2000, kD, 18);
  HNSWIndex h(kD, 2000);
  h.add(data.data(), 2000);
  const std::string path = ::testing::TempDir() + "hnsw_roundtrip.bin";
  h.save(path);
  auto h2 = HNSWIndex::load(path);
  ASSERT_EQ(h2->size(), h.size());
  EXPECT_EQ(h2->entry_point(), h.entry_point());
  for (size_t i = 0; i < 50; ++i) {
    auto a = h.search(data.data() + i * kD, 10, 50);
    auto b = h2->search(data.data() + i * kD, 10, 50);
    ASSERT_EQ(a.size(), b.size());
    for (size_t j = 0; j < a.size(); ++j) EXPECT_EQ(a[j].id, b[j].id);
  }
  std::remove(path.c_str());
}

TEST(HNSW, MultithreadedBuildKeepsRecall) {
  auto data = vstest::random_data(kN, kD, 19);
  auto qs = vstest::random_data(kQ, kD, 20);
  HNSWIndex h(kD, kN);
  h.add(data.data(), kN, /*num_threads=*/4);
  BruteForceIndex bf(kD);
  bf.add(data.data(), kN);
  EXPECT_GE(vstest::recall_at_k(h, bf, qs, kQ, kK, 100), 0.95);
}

TEST(HNSW, IncrementalAddsMatchCapacity) {
  auto data = vstest::random_data(1000, kD, 21);
  HNSWIndex h(kD, 1000);
  h.add(data.data(), 500);
  h.add(data.data() + 500 * kD, 500);
  EXPECT_EQ(h.size(), 1000u);
  auto r = h.search(data.data() + 999 * kD, 1, 50);
  EXPECT_EQ(r[0].id, 999u);
}

TEST(HNSW, BatchSearchMatchesSingle) {
  auto data = vstest::random_data(2000, kD, 22);
  HNSWIndex h(kD, 2000);
  h.add(data.data(), 2000);
  std::vector<uint32_t> ids(20 * 10);
  std::vector<float> ds(20 * 10);
  h.search_batch(data.data(), 20, 10, 50, ids.data(), ds.data(), 2);
  for (size_t i = 0; i < 20; ++i) {
    auto r = h.search(data.data() + i * kD, 10, 50);
    for (size_t j = 0; j < 10; ++j) EXPECT_EQ(ids[i * 10 + j], r[j].id);
  }
}

TEST(HNSW, HeuristicBeatsNaiveSelection) {
  auto data = vstest::random_data(kN, kD, 23);
  auto qs = vstest::random_data(kQ, kD, 24);
  BruteForceIndex bf(kD);
  bf.add(data.data(), kN);
  HNSWIndex smart(kD, kN, Metric::L2, {8, 100, 1, true});
  HNSWIndex naive(kD, kN, Metric::L2, {8, 100, 1, false});
  smart.add(data.data(), kN);
  naive.add(data.data(), kN);
  EXPECT_GE(vstest::recall_at_k(smart, bf, qs, kQ, kK, 20) + 0.01,
            vstest::recall_at_k(naive, bf, qs, kQ, kK, 20));
}
