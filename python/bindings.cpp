// Inputs are coerced to contiguous float32. The GIL is released during add and
// search so the gRPC shard workers can serve requests in parallel.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vecsearch/brute_force.h"
#include "vecsearch/hnsw.h"

namespace py = pybind11;
using namespace vecsearch;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

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
          [](const HNSWIndex& self, FloatArray q, size_t k, size_t ef, int num_threads) {
            if (q.ndim() == 1) q = q.reshape({py::ssize_t(1), q.shape(0)});
            check_2d(q, self.dim());
            const size_t nq = q.shape(0);
            py::array_t<uint32_t> ids({nq, k});
            py::array_t<float> dists({nq, k});
            {
              py::gil_scoped_release nogil;
              self.search_batch(q.data(), nq, k, ef, ids.mutable_data(), dists.mutable_data(),
                                num_threads);
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
          "Returns (ids[int64, nq x k], distances[float32, nq x k]).")
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
}
