#include "vecsearch/bm25.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace vecsearch {

namespace {

// Lucene's default English stop set.
const std::unordered_set<std::string>& stopwords() {
  static const std::unordered_set<std::string> s = {
      "a",    "an",   "and",   "are",  "as",   "at",    "be",   "but",  "by",
      "for",  "if",   "in",    "into", "is",   "it",    "no",   "not",  "of",
      "on",   "or",   "such",  "that", "the",  "their", "then", "there", "these",
      "they", "this", "to",    "was",  "will", "with"};
  return s;
}

// Orders by score descending, then doc id, so ties are deterministic.
bool better(const Neighbor& a, const Neighbor& b) {
  return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

const char kMagic[8] = {'V', 'S', 'B', 'M', '2', '5', '0', '1'};

template <class T>
void wr(std::ofstream& f, const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T>
void rd(std::ifstream& f, T& v) { f.read(reinterpret_cast<char*>(&v), sizeof(T)); }

}  // namespace

BM25Index::BM25Index(BM25Params params) : p_(params) {
  if (p_.k1 < 0 || p_.b < 0 || p_.b > 1) throw std::invalid_argument("need k1 >= 0 and 0 <= b <= 1");
}

std::vector<std::string> BM25Index::tokenize(const std::string& text, bool remove_stopwords) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&] {
    if (cur.size() >= 2 && !(remove_stopwords && stopwords().count(cur))) out.push_back(cur);
    cur.clear();
  };
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z')
      cur.push_back(char(c + ('a' - 'A')));
    else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 0x80)
      cur.push_back(char(c));
    else
      flush();
  }
  flush();
  return out;
}

void BM25Index::add(const std::vector<std::string>& docs) {
  if (size() + docs.size() > UINT32_MAX) throw std::length_error("BM25Index supports at most 2^32 docs");
  std::vector<uint32_t> terms;
  for (const auto& text : docs) {
    const uint32_t id = uint32_t(size());
    const auto tokens = tokenize(text, p_.remove_stopwords);
    terms.clear();
    for (const auto& t : tokens) {
      auto [it, inserted] = vocab_.try_emplace(t, uint32_t(postings_.size()));
      if (inserted) postings_.emplace_back();
      terms.push_back(it->second);
    }
    std::sort(terms.begin(), terms.end());
    for (size_t i = 0; i < terms.size();) {
      size_t j = i;
      while (j < terms.size() && terms[j] == terms[i]) ++j;
      postings_[terms[i]].push_back({id, uint32_t(j - i)});
      i = j;
    }
    doc_len_.push_back(uint32_t(tokens.size()));
    total_len_ += tokens.size();
  }
}

float BM25Index::idf(size_t df) const {
  const double n = double(size());
  return float(std::log(1.0 + (n - double(df) + 0.5) / (double(df) + 0.5)));
}

float BM25Index::tf_weight(uint32_t tf, uint32_t doc_len, float avg) const {
  const float norm = p_.k1 * (1.f - p_.b + p_.b * (avg > 0 ? float(doc_len) / avg : 0.f));
  return float(tf) * (p_.k1 + 1.f) / (float(tf) + norm);
}

float BM25Index::score(const std::string& term, uint32_t doc) const {
  auto it = vocab_.find(term);
  if (it == vocab_.end() || doc >= size()) return 0.f;
  const auto& pl = postings_[it->second];
  auto p = std::lower_bound(pl.begin(), pl.end(), doc,
                            [](const Posting& a, uint32_t d) { return a.doc < d; });
  if (p == pl.end() || p->doc != doc) return 0.f;
  return idf(pl.size()) * tf_weight(p->tf, doc_len_[doc], avg_len());
}

std::vector<Neighbor> BM25Index::search(const std::string& query, size_t k,
                                        const uint8_t* allowed) const {
  if (k == 0 || size() == 0) return {};
  auto terms = tokenize(query, p_.remove_stopwords);
  std::sort(terms.begin(), terms.end());
  terms.erase(std::unique(terms.begin(), terms.end()), terms.end());

  // Every contribution is > 0, so acc[d] == 0 means "not touched yet". Only the
  // touched entries are reset, which keeps the buffer all-zero between calls.
  thread_local std::vector<float> acc;
  thread_local std::vector<uint32_t> touched;
  if (acc.size() < size()) acc.resize(size(), 0.f);
  touched.clear();

  const float avg = avg_len();
  for (const auto& t : terms) {
    auto it = vocab_.find(t);
    if (it == vocab_.end()) continue;
    const auto& pl = postings_[it->second];
    const float w = idf(pl.size());
    for (const Posting& ps : pl) {
      if (allowed && !allowed[ps.doc]) continue;
      if (acc[ps.doc] == 0.f) touched.push_back(ps.doc);
      acc[ps.doc] += w * tf_weight(ps.tf, doc_len_[ps.doc], avg);
    }
  }

  std::vector<Neighbor> heap;  // worst kept result on top
  heap.reserve(std::min(k, touched.size()) + 1);
  for (uint32_t d : touched) {
    Neighbor n{-acc[d], d};
    acc[d] = 0.f;
    if (heap.size() < k) {
      heap.push_back(n);
      std::push_heap(heap.begin(), heap.end(), better);
    } else if (better(n, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), better);
      heap.back() = n;
      std::push_heap(heap.begin(), heap.end(), better);
    }
  }
  std::sort_heap(heap.begin(), heap.end(), better);
  return heap;
}

// Layout: magic, k1, b, stopwords flag, n_docs, total_len, doc lengths, n_terms,
// then per term: length, bytes, n_postings, postings.
void BM25Index::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.write(kMagic, 8);
  wr(f, p_.k1); wr(f, p_.b); wr(f, uint8_t(p_.remove_stopwords));
  wr(f, uint64_t(size())); wr(f, total_len_);
  f.write(reinterpret_cast<const char*>(doc_len_.data()), doc_len_.size() * sizeof(uint32_t));
  std::vector<const std::string*> by_id(postings_.size());
  for (const auto& [term, id] : vocab_) by_id[id] = &term;
  wr(f, uint64_t(by_id.size()));
  for (size_t i = 0; i < by_id.size(); ++i) {
    wr(f, uint32_t(by_id[i]->size()));
    f.write(by_id[i]->data(), by_id[i]->size());
    wr(f, uint64_t(postings_[i].size()));
    f.write(reinterpret_cast<const char*>(postings_[i].data()), postings_[i].size() * sizeof(Posting));
  }
  if (!f) throw std::runtime_error("write failed: " + path);
}

std::unique_ptr<BM25Index> BM25Index::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("not a vecsearch BM25 file");
  BM25Params p;
  uint8_t stop;
  uint64_t n, n_terms;
  rd(f, p.k1); rd(f, p.b); rd(f, stop);
  p.remove_stopwords = stop != 0;
  auto idx = std::make_unique<BM25Index>(p);
  rd(f, n); rd(f, idx->total_len_);
  idx->doc_len_.resize(n);
  f.read(reinterpret_cast<char*>(idx->doc_len_.data()), n * sizeof(uint32_t));
  rd(f, n_terms);
  idx->postings_.resize(n_terms);
  idx->vocab_.reserve(n_terms);
  std::string term;
  for (uint64_t i = 0; i < n_terms && f; ++i) {
    uint32_t len;
    uint64_t np;
    rd(f, len);
    term.resize(len);
    f.read(term.data(), len);
    rd(f, np);
    idx->postings_[i].resize(np);
    f.read(reinterpret_cast<char*>(idx->postings_[i].data()), np * sizeof(Posting));
    idx->vocab_.emplace(term, uint32_t(i));
  }
  if (!f) throw std::runtime_error("truncated BM25 file: " + path);
  return idx;
}

}  // namespace vecsearch
