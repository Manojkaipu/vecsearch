#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <unordered_set>

#include "test_util.h"

using namespace vecsearch;

namespace {

constexpr size_t kD = 32, kN = 5000, kQ = 100, kK = 10;

std::vector<uint8_t> random_mask(size_t n, double frac, uint64_t seed) {
  std::mt19937 rng(seed);
  std::bernoulli_distribution keep(frac);
  std::vector<uint8_t> m(n);
  for (auto& v : m) v = keep(rng);
  return m;
}

SearchFilter make_filter(const std::vector<uint8_t>& m, size_t exact_below = 0) {
  size_t cnt = std::count(m.begin(), m.end(), uint8_t(1));
  return {m.data(), cnt, exact_below};
}

// Exact top-k among allowed ids.
std::vector<uint32_t> exact_filtered(const HNSWIndex& h, const float* q, size_t k,
                                     const std::vector<uint8_t>& m) {
  std::vector<Neighbor> all;
  for (uint32_t i = 0; i < h.size(); ++i)
    if (m[i]) all.push_back({l2_sq(q, h.vector(i), h.dim()), i});
  std::sort(all.begin(), all.end());
  std::vector<uint32_t> out;
  for (size_t i = 0; i < std::min(k, all.size()); ++i) out.push_back(all[i].id);
  return out;
}

double filtered_recall(const HNSWIndex& h, const std::vector<float>& qs,
                       const std::vector<uint8_t>& m, const SearchFilter& f, size_t ef) {
  double hit = 0, total = 0;
  for (size_t i = 0; i < kQ; ++i) {
    const float* q = qs.data() + i * kD;
    auto truth = exact_filtered(h, q, kK, m);
    std::unordered_set<uint32_t> t(truth.begin(), truth.end());
    for (auto& n : h.search(q, kK, ef, f)) hit += t.count(n.id);
    total += truth.size();
  }
  return hit / total;
}

class FilteredSearch : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    data_ = new std::vector<float>(vstest::random_data(kN, kD, 30));
    qs_ = new std::vector<float>(vstest::random_data(kQ, kD, 31));
    h_ = new HNSWIndex(kD, kN);
    h_->add(data_->data(), kN, 4);
  }
  static void TearDownTestSuite() { delete h_; delete data_; delete qs_; }
  static std::vector<float>* data_;
  static std::vector<float>* qs_;
  static HNSWIndex* h_;
};
std::vector<float>* FilteredSearch::data_ = nullptr;
std::vector<float>* FilteredSearch::qs_ = nullptr;
HNSWIndex* FilteredSearch::h_ = nullptr;

}  // namespace

TEST_F(FilteredSearch, OnlyAllowedIdsReturned) {
  for (double frac : {0.5, 0.1, 0.01}) {
    auto m = random_mask(kN, frac, 40);
    auto f = make_filter(m);
    for (size_t i = 0; i < kQ; ++i)
      for (auto& n : h_->search(qs_->data() + i * kD, kK, 64, f))
        ASSERT_TRUE(m[n.id]) << "frac=" << frac << " id=" << n.id;
  }
}

TEST_F(FilteredSearch, HighRecallAtModerateSelectivity) {
  auto m = random_mask(kN, 0.3, 41);
  EXPECT_GE(filtered_recall(*h_, *qs_, m, make_filter(m), 100), 0.95);
}

TEST_F(FilteredSearch, ExactPathMatchesBruteForce) {
  auto m = random_mask(kN, 0.01, 42);  // ~50 allowed
  auto f = make_filter(m, /*exact_below=*/1000);
  for (size_t i = 0; i < kQ; ++i) {
    const float* q = qs_->data() + i * kD;
    auto truth = exact_filtered(*h_, q, kK, m);
    auto got = h_->search(q, kK, 10, f);
    ASSERT_EQ(got.size(), truth.size());
    for (size_t j = 0; j < got.size(); ++j) EXPECT_EQ(got[j].id, truth[j]);
  }
}

TEST_F(FilteredSearch, AllowAllMatchesUnfiltered) {
  std::vector<uint8_t> m(kN, 1);
  auto f = make_filter(m);
  for (size_t i = 0; i < 20; ++i) {
    const float* q = qs_->data() + i * kD;
    auto a = h_->search(q, kK, 64), b = h_->search(q, kK, 64, f);
    ASSERT_EQ(a.size(), b.size());
    for (size_t j = 0; j < a.size(); ++j) EXPECT_EQ(a[j].id, b[j].id);
  }
}

TEST_F(FilteredSearch, EmptyFilterReturnsNothing) {
  std::vector<uint8_t> m(kN, 0);
  EXPECT_TRUE(h_->search(qs_->data(), kK, 64, make_filter(m)).empty());
}

TEST_F(FilteredSearch, FewerAllowedThanK) {
  std::vector<uint8_t> m(kN, 0);
  m[7] = m[4000] = 1;
  auto r = h_->search(qs_->data(), kK, 64, make_filter(m));  // graph path
  ASSERT_EQ(r.size(), 2u);
  EXPECT_TRUE(m[r[0].id] && m[r[1].id]);
}
