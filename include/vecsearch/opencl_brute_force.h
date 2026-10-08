// Exact k-NN on any OpenCL device (Intel, NVIDIA, AMD, or POCL on a CPU). Built only with
// -DVECSEARCH_BUILD_OPENCL=ON. This is the portable twin of GpuBruteForceIndex (CUDA): the same
// kernels, the same chunked search, the same results, written in OpenCL C.
//
// The data lives on the device, padded to a multiple of 8 dims with zeros. A device may cap a
// single allocation well below its memory (Intel's caps it at 1 GiB), so the data is stored in
// pages of at most that size. A search computes a block of distances with one of the kernels,
// then picks the top k of each row with a work-group-wide select. Big batches go through the data
// in chunks, so the distance block stays under a fixed memory budget, and the per-chunk winners
// are merged at the end.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vecsearch/distance.h"
#include "vecsearch/gpu_method.h"

namespace vecsearch {

struct OpenCLDeviceInfo {
  std::string platform, name, version;
  bool gpu = false;
  bool subgroups = false;  // cl_khr_subgroups (or OpenCL 3.0 sub-group support)
  size_t compute_units = 0, global_mem = 0, max_alloc = 0, local_mem = 0, max_work_group = 0;
};

// Every OpenCL device on the machine, GPUs first. OpenCLBruteForceIndex's `device` indexes this list.
std::vector<OpenCLDeviceInfo> opencl_devices();

// True if at least one OpenCL device is visible.
bool opencl_available();

class OpenCLBruteForceIndex {
 public:
  static constexpr uint32_t kNone = 0xFFFFFFFFu;  // id of a missing result (k > size)
  static constexpr size_t kMaxK = 128;

  // subgroups: -1 = use sub-group reductions if the device has them (otherwise a local-memory
  // fallback), 1 = require them, 0 = always use the fallback.
  OpenCLBruteForceIndex(size_t dim, Metric metric = Metric::L2, int device = 0, int subgroups = -1);
  ~OpenCLBruteForceIndex();
  OpenCLBruteForceIndex(const OpenCLBruteForceIndex&) = delete;
  OpenCLBruteForceIndex& operator=(const OpenCLBruteForceIndex&) = delete;

  void add(const float* data, size_t n);

  // Top-k for nq host queries into host arrays (nq x k, nearest first; missing = kNone, inf).
  // Includes copying the queries in and the results out. k <= kMaxK.
  void search(const float* queries, size_t nq, size_t k, uint32_t* out_ids, float* out_dists,
              GpuMethod method = GpuMethod::Auto);

  size_t size() const;
  size_t dim() const;
  Metric metric() const;
  const OpenCLDeviceInfo& device_info() const;
  bool uses_subgroups() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vecsearch
