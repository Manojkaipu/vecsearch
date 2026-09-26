// Inputs are coerced to contiguous float32. The GIL is released during add and
// search so the gRPC shard workers can serve requests in parallel.
#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vecsearch/bm25.h"
#include "vecsearch/brute_force.h"
#include "vecsearch/hnsw.h"

namespace py = pybind11;
using namespace vecsearch;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using MaskArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// None or a bool/uint8 array with one entry per indexed item. ptr is null when
// there's no filter (a default-constructed array_t is a valid empty array, not null).
struct Mask {
  MaskArray arr;
  const uint8_t* ptr = nullptr;
};

static Mask as_mask(const py::object& filter, size_t n) {
  Mask m;
  if (filter.is_none()) return m;
  m.arr = MaskArray::ensure(filter);
  if (!m.arr || m.arr.ndim() != 1 || size_t(m.arr.shape(0)) != n)
    throw std::invalid_argument("filter must be a 1-D bool array of length " + std::to_string(n));
  m.ptr = m.arr.data();
  return m;
}

static Metric parse_metric(const std::string& s) {
  if (s == "l2") return Metric::L2;
  if (s == "ip" || s == "cosine") return Metric::InnerProduct;
  throw std::invalid_argument("metric must be 'l2', 'ip' or 'cosine' (cosine expects normalized vectors)");
}

static void check_2d(const FloatArray& a, size_t dim) {
  if (a.ndim() != 2 || size_t(a.shape(1)) != dim)
    throw std::invalid_argument("expected array of shape (n, " + std::to_string(dim) + ")");
}

PYBIND11_MODULE(_vecsearch, m) {
  m.doc() = "HNSW approximate nearest-neighbor search in C++";

  py::class_<HNSWIndex>(m, "HNSWIndex")
      .def(py::init([](size_t dim, size_t max_elements, const std::string& metric, size_t M,
                       size_t ef_construction, uint64_t seed, bool use_heuristic) {
             return std::make_unique<HNSWIndex>(dim, max_elements, parse_metric(metric),
                                                HNSWParams{M, ef_construction, seed, use_heuristic});
           }),
           py::arg("dim"), py::arg("max_elements"), py::arg("metric") = "l2", py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("seed") = 42, py::arg("use_heuristic") = true)
      .def(
          "add",
          [](HNSWIndex& self, FloatArray x, int num_threads) {
            check_2d(x, self.dim());
            py::gil_scoped_release nogil;
            self.add(x.data(), x.shape(0), num_threads);
          },
          py::arg("x"), py::arg("num_threads") = 1)
      .def(
          "search",
          [](const HNSWIndex& self, FloatArray q, size_t k, size_t ef, int num_threads,
             const py::object& filter, size_t exact_below) {
            if (q.ndim() == 1) q = q.reshape({py::ssize_t(1), q.shape(0)});
            check_2d(q, self.dim());
            const size_t nq = q.shape(0);
            Mask mask = as_mask(filter, self.size());
            SearchFilter f;
            if (mask.ptr) {
              f.allowed = mask.ptr;
              f.n_allowed = size_t(std::count_if(f.allowed, f.allowed + self.size(),
                                                 [](uint8_t v) { return v != 0; }));
              f.exact_below = exact_below;
            }
            py::array_t<uint32_t> ids({nq, k});
            py::array_t<float> dists({nq, k});
            {
              py::gil_scoped_release nogil;
              self.search_batch(q.data(), nq, k, ef, ids.mutable_data(), dists.mutable_data(),
                                num_threads, f);
            }
            // kNone -> -1
            py::array_t<int64_t> ids64({nq, k});
            auto src = ids.unchecked<2>();
            auto dst = ids64.mutable_unchecked<2>();
            for (size_t i = 0; i < nq; ++i)
              for (size_t j = 0; j < k; ++j)
                dst(i, j) = src(i, j) == HNSWIndex::kNone ? -1 : int64_t(src(i, j));
            return py::make_tuple(ids64, dists);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("num_threads") = 1,
          py::arg("filter") = py::none(), py::arg("exact_below") = 0,
          "Returns (ids[int64, nq x k], distances[float32, nq x k]). filter: optional bool\n"
          "mask over ids; with <= exact_below allowed ids the search is an exact scan.")
      .def("save", &HNSWIndex::save, py::arg("path"))
      .def_static("load", &HNSWIndex::load, py::arg("path"), py::arg("extra_capacity") = 0)
      .def("neighbors", &HNSWIndex::neighbors, py::arg("id"), py::arg("level") = 0)
      .def("level_of", &HNSWIndex::level_of)
      .def_property_readonly("dim", &HNSWIndex::dim)
      .def_property_readonly("capacity", &HNSWIndex::capacity)
      .def_property_readonly("max_level", &HNSWIndex::max_level)
      .def_property_readonly("M", &HNSWIndex::M)
      .def("__len__", &HNSWIndex::size);

  py::class_<BruteForceIndex>(m, "BruteForceIndex")
      .def(py::init([](size_t dim, const std::string& metric) {
             return std::make_unique<BruteForceIndex>(dim, parse_metric(metric));
           }),
           py::arg("dim"), py::arg("metric") = "l2")
      .def("add",
           [](BruteForceIndex& self, FloatArray x) {
             check_2d(x, self.dim());
             self.add(x.data(), x.shape(0));
           })
      .def(
          "search",
          [](const BruteForceIndex& self, FloatArray q, size_t k) {
            if (q.ndim() == 1) q = q.reshape({py::ssize_t(1), q.shape(0)});
            check_2d(q, self.dim());
            const size_t nq = q.shape(0);
            py::array_t<int64_t> ids({nq, k});
            py::array_t<float> dists({nq, k});
            auto I = ids.mutable_unchecked<2>();
            auto D = dists.mutable_unchecked<2>();
            for (size_t i = 0; i < nq; ++i) {
              auto r = self.search(q.data() + i * self.dim(), k);
              for (size_t j = 0; j < k; ++j) {
                I(i, j) = j < r.size() ? int64_t(r[j].id) : -1;
                D(i, j) = j < r.size() ? r[j].dist : INFINITY;
              }
            }
            return py::make_tuple(ids, dists);
          },
          py::arg("queries"), py::arg("k") = 10)
      .def("__len__", &BruteForceIndex::size);

  py::class_<BM25Index>(m, "BM25Index")
      .def(py::init([](float k1, float b, bool remove_stopwords) {
             return std::make_unique<BM25Index>(BM25Params{k1, b, remove_stopwords});
           }),
           py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("remove_stopwords") = true)
      .def(
          "add",
          [](BM25Index& self, const std::vector<std::string>& docs) {
            py::gil_scoped_release nogil;
            self.add(docs);
          },
          py::arg("docs"))
      .def(
          "search",
          [](const BM25Index& self, const py::object& queries, size_t k, const py::object& filter) {
            std::vector<std::string> qs;
            if (py::isinstance<py::str>(queries))
              qs.push_back(queries.cast<std::string>());
            else
              qs = queries.cast<std::vector<std::string>>();
            Mask mask = as_mask(filter, self.size());
            const uint8_t* allowed = mask.ptr;
            const size_t nq = qs.size();
            py::array_t<int64_t> ids({nq, k});
            py::array_t<float> scores({nq, k});
            auto I = ids.mutable_unchecked<2>();
            auto S = scores.mutable_unchecked<2>();
            {
              py::gil_scoped_release nogil;
              for (size_t i = 0; i < nq; ++i) {
                auto r = self.search(qs[i], k, allowed);
                for (size_t j = 0; j < k; ++j) {
                  I(i, j) = j < r.size() ? int64_t(r[j].id) : -1;
                  S(i, j) = j < r.size() ? -r[j].dist : 0.f;
                }
              }
            }
            return py::make_tuple(ids, scores);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("filter") = py::none(),
          "Returns (ids[int64, nq x k], scores[float32, nq x k]), best first; missing = -1.")
      .def("score", &BM25Index::score, py::arg("term"), py::arg("doc"))
      .def_static("tokenize", &BM25Index::tokenize, py::arg("text"),
                  py::arg("remove_stopwords") = true)
      .def("save", &BM25Index::save, py::arg("path"))
      .def_static("load", &BM25Index::load, py::arg("path"))
      .def_property_readonly("vocab_size", &BM25Index::vocab_size)
      .def("__len__", &BM25Index::size);
}
