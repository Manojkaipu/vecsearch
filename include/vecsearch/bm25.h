// Okapi BM25 over an in-memory inverted index.
//
// idf = ln(1 + (N - df + 0.5) / (df + 0.5))   (Lucene's variant, never negative)
// tf part = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
//
// Results reuse Neighbor with dist = -score so that, as everywhere else in the
// library, smaller is better.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vecsearch/distance.h"

namespace vecsearch {

struct BM25Params {
  float k1 = 1.2f;
  float b = 0.75f;
  bool remove_stopwords = true;
};

class BM25Index {
 public:
  explicit BM25Index(BM25Params params = {});

  // Lowercases ASCII and splits on anything that isn't [a-z0-9] or a non-ASCII
  // byte, so UTF-8 words stay whole. Drops 1-char tokens and, optionally, stopwords.
  static std::vector<std::string> tokenize(const std::string& text, bool remove_stopwords = true);

  // Appends documents; ids continue from size(). Must not run concurrently with search().
  void add(const std::vector<std::string>& docs);

  // allowed: optional byte mask of length size(); docs with allowed[id] == 0 are skipped.
  std::vector<Neighbor> search(const std::string& query, size_t k,
                               const uint8_t* allowed = nullptr) const;

  // Term-level stats, used by tests.
  float score(const std::string& term, uint32_t doc) const;

  void save(const std::string& path) const;
  static std::unique_ptr<BM25Index> load(const std::string& path);

  size_t size() const { return doc_len_.size(); }
  size_t vocab_size() const { return postings_.size(); }
  BM25Params params() const { return p_; }

 private:
  struct Posting {
    uint32_t doc;
    uint32_t tf;
  };

  float idf(size_t df) const;
  float tf_weight(uint32_t tf, uint32_t doc_len, float avg_len) const;
  float avg_len() const { return size() ? float(double(total_len_) / size()) : 0.f; }

  BM25Params p_;
  std::unordered_map<std::string, uint32_t> vocab_;
  std::vector<std::vector<Posting>> postings_;  // by term id, doc ids ascending
  std::vector<uint32_t> doc_len_;
  uint64_t total_len_ = 0;
};

}  // namespace vecsearch
