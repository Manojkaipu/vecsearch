#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <thread>

#include "vecsearch/bm25.h"

using namespace vecsearch;

namespace {

const std::vector<std::string> kDocs = {
    "My iPhone battery drains fast after the iOS 11 update",
    "Battery drain on iPhone 7, tried restarting, still draining",
    "Refund for my cancelled flight has not arrived",
    "How do I get a refund for a cancelled order?",
    "The app crashes when I open my playlist",
    "iOS 11 update broke my wifi and battery",
};

// Straight transcription of the BM25 formula, computed in double.
std::map<uint32_t, double> reference_scores(const std::vector<std::string>& docs,
                                            const std::string& query, double k1, double b) {
  std::vector<std::vector<std::string>> toks;
  double total = 0;
  for (const auto& d : docs) {
    toks.push_back(BM25Index::tokenize(d));
    total += toks.back().size();
  }
  const double n = docs.size(), avg = total / n;
  auto q = BM25Index::tokenize(query);
  std::sort(q.begin(), q.end());
  q.erase(std::unique(q.begin(), q.end()), q.end());
  std::map<uint32_t, double> out;
  for (const auto& t : q) {
    double df = 0;
    for (const auto& d : toks) df += std::count(d.begin(), d.end(), t) > 0;
    if (df == 0) continue;
    const double idf = std::log(1 + (n - df + 0.5) / (df + 0.5));
    for (uint32_t i = 0; i < toks.size(); ++i) {
      const double tf = double(std::count(toks[i].begin(), toks[i].end(), t));
      if (tf == 0) continue;
      out[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * toks[i].size() / avg));
    }
  }
  return out;
}

}  // namespace

TEST(BM25, Tokenizer) {
  EXPECT_EQ(BM25Index::tokenize("Hello, World! It's iOS-11 caf\xc3\xa9"),
            (std::vector<std::string>{"hello", "world", "ios", "11", "caf\xc3\xa9"}));
  EXPECT_EQ(BM25Index::tokenize("It is THE app", false),
            (std::vector<std::string>{"it", "is", "the", "app"}));
  EXPECT_TRUE(BM25Index::tokenize("a I , . !").empty());
}

TEST(BM25, ScoresMatchReference) {
  BM25Index idx;
  idx.add(kDocs);
  for (const std::string q : {"iphone battery drain", "refund cancelled", "ios 11 update wifi",
                              "playlist crashes app"}) {
    auto ref = reference_scores(kDocs, q, 1.2, 0.75);
    auto got = idx.search(q, 10);
    ASSERT_EQ(got.size(), ref.size()) << q;
    for (size_t i = 0; i < got.size(); ++i) {
      EXPECT_NEAR(-got[i].dist, ref.at(got[i].id), 1e-5) << q;
      if (i) EXPECT_LE(got[i - 1].dist, got[i].dist);
    }
  }
}

TEST(BM25, ScoreMethodMatchesSearch) {
  BM25Index idx;
  idx.add(kDocs);
  auto r = idx.search("refund", 10);
  ASSERT_FALSE(r.empty());
  for (auto& n : r) EXPECT_FLOAT_EQ(idx.score("refund", n.id), -n.dist);
  EXPECT_EQ(idx.score("refund", 0), 0.f);
  EXPECT_EQ(idx.score("nonexistentterm", 2), 0.f);
}

TEST(BM25, NoMatchesAndEdgeCases) {
  BM25Index empty;
  EXPECT_TRUE(empty.search("battery", 5).empty());
  BM25Index idx;
  idx.add(kDocs);
  EXPECT_TRUE(idx.search("zzzz qqqq", 5).empty());
  EXPECT_TRUE(idx.search("the of and", 5).empty());  // only stopwords
  EXPECT_TRUE(idx.search("battery", 0).empty());
  EXPECT_EQ(idx.search("battery", 100).size(), 3u);  // k larger than matches
}

TEST(BM25, FilterSkipsDisallowedDocs) {
  BM25Index idx;
  idx.add(kDocs);
  std::vector<uint8_t> allowed = {0, 1, 1, 1, 1, 0};  // hide docs 0 and 5
  auto r = idx.search("ios 11 battery", 10, allowed.data());
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].id, 1u);
  auto ref = reference_scores(kDocs, "ios 11 battery", 1.2, 0.75);
  EXPECT_NEAR(-r[0].dist, ref.at(1), 1e-5);  // filtering doesn't change collection stats
}

TEST(BM25, TiesBrokenById) {
  BM25Index idx;
  idx.add({"same text here", "other words", "same text here", "same text here"});
  auto r = idx.search("same text", 10);
  ASSERT_EQ(r.size(), 3u);
  EXPECT_EQ(r[0].id, 0u);
  EXPECT_EQ(r[1].id, 2u);
  EXPECT_EQ(r[2].id, 3u);
}

TEST(BM25, IncrementalAddMatchesBulk) {
  BM25Index bulk, inc;
  bulk.add(kDocs);
  inc.add({kDocs.begin(), kDocs.begin() + 2});
  inc.add({kDocs.begin() + 2, kDocs.end()});
  auto a = bulk.search("battery refund ios", 10), b = inc.search("battery refund ios", 10);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].id, b[i].id);
    EXPECT_FLOAT_EQ(a[i].dist, b[i].dist);
  }
}

TEST(BM25, SaveLoadRoundTrip) {
  BM25Index idx({0.9f, 0.4f, true});
  idx.add(kDocs);
  const std::string path = ::testing::TempDir() + "bm25_roundtrip.bin";
  idx.save(path);
  auto loaded = BM25Index::load(path);
  EXPECT_EQ(loaded->size(), idx.size());
  EXPECT_EQ(loaded->vocab_size(), idx.vocab_size());
  EXPECT_FLOAT_EQ(loaded->params().k1, 0.9f);
  auto a = idx.search("iphone battery update", 10), b = loaded->search("iphone battery update", 10);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].id, b[i].id);
    EXPECT_FLOAT_EQ(a[i].dist, b[i].dist);
  }
  std::remove(path.c_str());
}

TEST(BM25, ConcurrentSearchesAgree) {
  BM25Index idx;
  std::vector<std::string> docs;
  for (int i = 0; i < 2000; ++i)
    docs.push_back(kDocs[i % kDocs.size()] + " doc" + std::to_string(i % 37));
  idx.add(docs);
  const auto expect = idx.search("battery refund doc5", 20);
  std::vector<std::thread> ts;
  std::atomic<int> mismatches{0};
  for (int t = 0; t < 4; ++t)
    ts.emplace_back([&] {
      for (int rep = 0; rep < 50; ++rep) {
        auto r = idx.search("battery refund doc5", 20);
        if (r.size() != expect.size()) { ++mismatches; continue; }
        for (size_t i = 0; i < r.size(); ++i)
          if (r[i].id != expect[i].id) ++mismatches;
      }
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(mismatches.load(), 0);
}
