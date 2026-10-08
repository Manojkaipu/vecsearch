#include "vecsearch/opencl_brute_force.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

// The kernel source, embedded at configure time from src/opencl/kernels.cl.
extern const char* const kVecsearchOpenCLSource;

#define CL_CHECK(expr)                                                             \
  do {                                                                             \
    const cl_int err_ = (expr);                                                    \
    if (err_ != CL_SUCCESS)                                                        \
      throw std::runtime_error(std::string(#expr) + ": " + cl_error_name(err_));  \
  } while (0)

namespace vecsearch {

namespace {

constexpr uint32_t kNoId = OpenCLBruteForceIndex::kNone;
constexpr int kSkinnyQueries = 8;          // queries per skinny pass over the data
// Auto picks Skinny up to this many queries (the CUDA crossover on a T4; see NOTES.md for Intel).
constexpr size_t kSkinnyMaxBatch = 16;
constexpr size_t kMaxQueryTile = 4096;     // queries per round (bounds the candidate buffers)
constexpr size_t kDistBudget = size_t(1) << 27;  // floats in the distance block (512 MB)
constexpr int kSelectSlots = 4096;         // candidates one select work-group sorts in local memory
constexpr size_t kMinSegment = 16384;      // a select work-group scans at least this many distances
constexpr int kTile = 128;                 // tiled kernel: 128x128 outputs
constexpr size_t kGroup = 256;             // work-group size of every kernel except select
constexpr size_t kSkinnyGroupsPerCu = 16;

const char* cl_error_name(cl_int e) {
  switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case -1001: return "no OpenCL platforms (is an ICD installed?)";
    default: return "OpenCL error";
  }
}

std::string device_string(cl_device_id d, cl_device_info what) {
  size_t n = 0;
  if (clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
  std::string s(n, '\0');
  if (clGetDeviceInfo(d, what, n, &s[0], nullptr) != CL_SUCCESS) return {};
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

template <class T>
T device_value(cl_device_id d, cl_device_info what) {
  T v{};
  clGetDeviceInfo(d, what, sizeof(T), &v, nullptr);
  return v;
}

struct Found {
  cl_platform_id platform;
  cl_device_id device;
  OpenCLDeviceInfo info;
};

// OpenCL version from "OpenCL 3.0 NEO": major * 10 + minor.
int version_number(const std::string& v) {
  const size_t p = v.find("OpenCL ");
  if (p == std::string::npos || p + 10 > v.size()) return 0;
  return (v[p + 7] - '0') * 10 + (v[p + 9] - '0');
}

std::vector<Found> enumerate() {
  std::vector<Found> out;
  cl_uint np = 0;
  if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return out;
  std::vector<cl_platform_id> platforms(np);
  if (clGetPlatformIDs(np, platforms.data(), nullptr) != CL_SUCCESS) return out;
  for (cl_platform_id p : platforms) {
    size_t n = 0;
    std::string pname;
    if (clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, nullptr, &n) == CL_SUCCESS && n) {
      pname.assign(n, '\0');
      clGetPlatformInfo(p, CL_PLATFORM_NAME, n, &pname[0], nullptr);
      while (!pname.empty() && pname.back() == '\0') pname.pop_back();
    }
    cl_uint nd = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
    std::vector<cl_device_id> devs(nd);
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr) != CL_SUCCESS) continue;
    for (cl_device_id d : devs) {
      if (device_value<cl_bool>(d, CL_DEVICE_AVAILABLE) != CL_TRUE) continue;
      OpenCLDeviceInfo i;
      i.platform = pname;
      i.name = device_string(d, CL_DEVICE_NAME);
      i.version = device_string(d, CL_DEVICE_VERSION);
      i.gpu = (device_value<cl_device_type>(d, CL_DEVICE_TYPE) & CL_DEVICE_TYPE_GPU) != 0;
      const std::string ext = device_string(d, CL_DEVICE_EXTENSIONS);
      const int v = version_number(i.version);
      i.subgroups = ext.find("cl_khr_subgroups") != std::string::npos || (v >= 21 && v < 30);
      i.compute_units = device_value<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS);
      i.global_mem = size_t(device_value<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE));
      i.max_alloc = size_t(device_value<cl_ulong>(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE));
      i.local_mem = size_t(device_value<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE));
      i.max_work_group = device_value<size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE);
      out.push_back({p, d, std::move(i)});
    }
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const Found& a, const Found& b) { return a.info.gpu && !b.info.gpu; });
  return out;
}

struct MemRelease {
  void operator()(cl_mem m) const { clReleaseMemObject(m); }
};
using MemPtr = std::unique_ptr<std::remove_pointer_t<cl_mem>, MemRelease>;

// The context and queue every buffer and kernel hangs off.
struct Queue {
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  ~Queue() {
    if (q) clReleaseCommandQueue(q);
    if (ctx) clReleaseContext(ctx);
  }
};

// Grow-only device buffer, zero-filled when (re)allocated.
template <class T>
class DeviceArray {
 public:
  cl_mem get() const { return p_.get(); }
  size_t capacity() const { return n_; }
  void reserve(const Queue& qu, size_t n, size_t keep = 0) {
    if (n <= n_) return;
    cl_int err = CL_SUCCESS;
    MemPtr p(clCreateBuffer(qu.ctx, CL_MEM_READ_WRITE, n * sizeof(T), nullptr, &err));
    if (!p) throw std::runtime_error(std::string("clCreateBuffer(") + std::to_string(n * sizeof(T)) +
                                     " bytes): " + cl_error_name(err));
    const cl_uint zero = 0;
    CL_CHECK(clEnqueueFillBuffer(qu.q, p.get(), &zero, sizeof(zero), 0, n * sizeof(T), 0, nullptr, nullptr));
    if (keep)
      CL_CHECK(clEnqueueCopyBuffer(qu.q, p_.get(), p.get(), 0, 0, keep * sizeof(T), 0, nullptr, nullptr));
    CL_CHECK(clFinish(qu.q));  // the old buffer is released next
    p_ = std::move(p);
    n_ = n;
  }

 private:
  MemPtr p_;
  size_t n_ = 0;
};

size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

int select_bucket(size_t k) {
  for (int K : {16, 32, 64, 128})
    if (k <= size_t(K)) return K;
  throw std::invalid_argument("k must be <= " + std::to_string(OpenCLBruteForceIndex::kMaxK));
}

// Kernel arguments are typed on the device side, so each C++ argument is mapped to exactly one of
// cl_mem, cl_int, cl_uint or cl_ulong.
inline void set_arg(cl_kernel k, cl_uint i, cl_mem v) { CL_CHECK(clSetKernelArg(k, i, sizeof(v), &v)); }
inline void set_arg(cl_kernel k, cl_uint i, cl_int v) { CL_CHECK(clSetKernelArg(k, i, sizeof(v), &v)); }
inline void set_arg(cl_kernel k, cl_uint i, cl_uint v) { CL_CHECK(clSetKernelArg(k, i, sizeof(v), &v)); }
inline void set_arg(cl_kernel k, cl_uint i, cl_ulong v) { CL_CHECK(clSetKernelArg(k, i, sizeof(v), &v)); }

cl_uint U(size_t v) { return cl_uint(v); }
cl_ulong UL(size_t v) { return cl_ulong(v); }

}  // namespace

std::vector<OpenCLDeviceInfo> opencl_devices() {
  std::vector<OpenCLDeviceInfo> out;
  for (const Found& f : enumerate()) out.push_back(f.info);
  return out;
}

bool opencl_available() { return !enumerate().empty(); }

struct OpenCLBruteForceIndex::Impl {
  // A page holds up to page_rows vectors: no single allocation exceeds the device's limit.
  struct Page {
    DeviceArray<float> x, xn;  // data (cap x dp) and squared norms
    size_t rows = 0;
  };

  size_t dim, dp;
  Metric metric;
  OpenCLDeviceInfo info;
  cl_device_id dev = nullptr;
  bool use_sg = false;
  size_t sms = 1;
  size_t page_rows = 0, dist_budget = kDistBudget;
  size_t n = 0;
  Queue qu;
  std::vector<std::unique_ptr<std::remove_pointer_t<cl_program>, void (*)(cl_program)>> programs;
  std::map<std::string, cl_kernel> kernels;
  std::vector<Page> pages;
  bool profiling = false;  // VECSEARCH_OPENCL_PROFILE=1: time each kernel with OpenCL events
  std::vector<std::pair<std::string, cl_event>> events;
  DeviceArray<float> q, qn, dist, cand_d[2];
  DeviceArray<uint32_t> cand_i[2];
  std::mutex mu;

  ~Impl() {
    for (auto& e : events) clReleaseEvent(e.second);
    for (auto& kv : kernels) clReleaseKernel(kv.second);
    // programs, buffers and the queue are released by their own destructors, in that order
  }

  // Builds `kernel` from the embedded source with the given defines; kept for later launches.
  cl_kernel kernel(const std::string& name, const std::string& defines) {
    const std::string key = name + "|" + defines + (use_sg ? "|sg" : "");
    auto it = kernels.find(key);
    if (it != kernels.end()) return it->second;
    cl_int err = CL_SUCCESS;
    const char* src = kVecsearchOpenCLSource;
    cl_program raw = clCreateProgramWithSource(qu.ctx, 1, &src, nullptr, &err);
    if (!raw) throw std::runtime_error(std::string("clCreateProgramWithSource: ") + cl_error_name(err));
    programs.emplace_back(raw, [](cl_program p) { clReleaseProgram(p); });
    std::string options = defines + (use_sg ? " -DVS_SUBGROUPS" : "");
    err = clBuildProgram(raw, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t n = 0;
      clGetProgramBuildInfo(raw, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
      std::string log(n, '\0');
      clGetProgramBuildInfo(raw, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
      programs.pop_back();
      throw std::runtime_error("building " + name + " [" + options + "] for " + info.name + ": " +
                               cl_error_name(err) + "\n" + log);
    }
    cl_kernel k = clCreateKernel(raw, name.c_str(), &err);
    if (!k) throw std::runtime_error("clCreateKernel(" + name + "): " + cl_error_name(err));
    kernels[key] = k;
    return k;
  }

  template <class... Args>
  void launch(const char* name, cl_kernel k, size_t dims, const size_t* global, const size_t* local, Args... args) {
    cl_uint i = 0;
    int unused[] = {0, (set_arg(k, i++, args), 0)...};
    (void)unused;
    cl_event ev = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(qu.q, k, cl_uint(dims), nullptr, global, local, 0, nullptr,
                                    profiling ? &ev : nullptr));
    if (ev) events.emplace_back(name, ev);
  }

  // Milliseconds spent in each kernel since the last call (needs profiling), summed by name.
  std::map<std::string, double> take_profile() {
    std::map<std::string, double> ms;
    CL_CHECK(clFinish(qu.q));
    for (auto& e : events) {
      cl_ulong t0 = 0, t1 = 0;
      clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
      clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
      ms[e.first] += double(t1 - t0) * 1e-6;
      clReleaseEvent(e.second);
    }
    events.clear();
    return ms;
  }

  // Squared norm of `rows` rows of buf starting at row0, into out[out0 ...).
  void norms(cl_mem buf, size_t row0, size_t rows, cl_mem out, size_t out0) {
    const size_t groups = std::min(ceil_div(rows, 8), sms * kSkinnyGroupsPerCu);
    const size_t global = groups * kGroup, local = kGroup;
    launch("norms", kernel("norms_kernel", "-DVS_NORMS"), 1, &global, &local, buf, UL(row0), U(rows), U(dp), out, UL(out0));
  }

  // Segment length for a select pass: enough segments to fill the GPU, but each long enough
  // that scanning it outweighs the fixed cost of the sort.
  size_t seg_len(size_t rows, size_t len) const {
    const size_t want = std::max<size_t>(1, size_t(2 * sms) / rows);
    return std::max(kMinSegment, ceil_div(len, want));
  }

  void select(int K, size_t nseg, size_t rows, cl_mem d, cl_mem ids, bool use_ids, size_t ld_in,
              size_t id_base, size_t len, size_t seg, cl_mem out_d, cl_mem out_i, size_t ld_out,
              size_t out_col) {
    const size_t threads = kSelectSlots / K;
    const size_t global[2] = {nseg * threads, rows}, local[2] = {threads, 1};
    launch("select", kernel("select_kernel", "-DVS_K=" + std::to_string(K)), 2, global, local, d, ids,
           cl_int(use_ids), UL(ld_in), U(id_base), U(len), U(seg), out_d, out_i, UL(ld_out), U(out_col));
  }

  // Distances from the first qt queries to len vectors of page p starting at row c0, into dist.
  void distances(GpuMethod m, size_t qt, const Page& p, size_t c0, size_t len, size_t ld) {
    const cl_int l2 = metric == Metric::L2;
    switch (m) {
      case GpuMethod::Naive: {
        const size_t global[2] = {ceil_div(len, kGroup) * kGroup, qt}, local[2] = {kGroup, 1};
        launch("naive", kernel("naive_kernel", "-DVS_NAIVE"), 2, global, local, q.get(), p.x.get(), UL(c0), U(qt),
               U(len), U(dp), l2, dist.get(), UL(ld));
        break;
      }
      case GpuMethod::Skinny: {
        const size_t groups = std::min(ceil_div(len, 8), sms * kSkinnyGroupsPerCu);
        const size_t global = groups * kGroup, local = kGroup;
        for (size_t b0 = 0; b0 < qt; b0 += kSkinnyQueries) {
          const size_t nb = std::min<size_t>(kSkinnyQueries, qt - b0);
          launch("skinny", kernel("skinny_kernel", "-DVS_QB=" + std::to_string(nb)), 1, &global, &local, q.get(),
                 UL(b0 * dp / 4), p.x.get(), UL(c0), U(len), U(dp), l2, dist.get(), UL(b0 * ld), UL(ld));
        }
        break;
      }
      default: {
        const size_t global[2] = {ceil_div(len, kTile) * kGroup, ceil_div(qt, kTile)}, local[2] = {kGroup, 1};
        launch("tiled", kernel("tiled_kernel", "-DVS_TILED"), 2, global, local, q.get(), p.x.get(), UL(c0), U(qt),
               U(len), U(dp), l2, qn.get(), p.xn.get(), dist.get(), UL(ld));
        break;
      }
    }
  }

  // Host (n x dim, row pitch dim) -> buffer (n x dp, zero padding stays), starting at row `row`.
  void upload(cl_mem buf, size_t row, const float* host, size_t rows) {
    if (dim == dp) {
      CL_CHECK(clEnqueueWriteBuffer(qu.q, buf, CL_TRUE, row * dp * sizeof(float),
                                    rows * dim * sizeof(float), host, 0, nullptr, nullptr));
      return;
    }
    const size_t buf_origin[3] = {0, row, 0}, host_origin[3] = {0, 0, 0};
    const size_t region[3] = {dim * sizeof(float), rows, 1};
    CL_CHECK(clEnqueueWriteBufferRect(qu.q, buf, CL_TRUE, buf_origin, host_origin, region,
                                      dp * sizeof(float), 0, dim * sizeof(float), 0, host, 0, nullptr,
                                      nullptr));
  }

  // Device (rows x width) -> host (rows x cols), the first cols columns of each row.
  void download(cl_mem buf, size_t width, size_t elem, void* host, size_t rows, size_t cols) {
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {cols * elem, rows, 1};
    CL_CHECK(clEnqueueReadBufferRect(qu.q, buf, CL_TRUE, origin, origin, region, width * elem, 0,
                                     cols * elem, 0, host, 0, nullptr, nullptr));
  }
};

OpenCLBruteForceIndex::OpenCLBruteForceIndex(size_t dim, Metric metric, int device, int subgroups)
    : impl_(new Impl) {
  if (dim == 0) throw std::invalid_argument("dim must be > 0");
  Impl& s = *impl_;
  s.dim = dim;
  s.dp = (dim + 7) / 8 * 8;
  s.metric = metric;
  std::vector<Found> found = enumerate();
  if (found.empty()) throw std::runtime_error("no OpenCL devices found (is an OpenCL ICD installed?)");
  if (device < 0 || size_t(device) >= found.size())
    throw std::invalid_argument("device must be in [0, " + std::to_string(found.size()) + ")");
  s.info = found[device].info;
  s.dev = found[device].device;
  s.sms = std::max<size_t>(1, s.info.compute_units);
  if (s.info.max_work_group < kGroup || s.info.local_mem < size_t(kSelectSlots) * 8)
    throw std::runtime_error(s.info.name + " is too small: needs work-groups of " + std::to_string(kGroup) +
                             " and " + std::to_string(kSelectSlots * 8) + " bytes of local memory");

  cl_int err = CL_SUCCESS;
  const cl_context_properties props[] = {CL_CONTEXT_PLATFORM, cl_context_properties(found[device].platform), 0};
  s.qu.ctx = clCreateContext(props, 1, &s.dev, nullptr, nullptr, &err);
  if (!s.qu.ctx) throw std::runtime_error(std::string("clCreateContext: ") + cl_error_name(err));
  const char* prof = std::getenv("VECSEARCH_OPENCL_PROFILE");
  s.profiling = prof && prof[0] == '1';
  s.qu.q = clCreateCommandQueue(s.qu.ctx, s.dev, s.profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
  if (!s.qu.q) throw std::runtime_error(std::string("clCreateCommandQueue: ") + cl_error_name(err));

  // Sub-groups: use them if asked or available, and if the compiler accepts them.
  if (subgroups == 1 && !s.info.subgroups)
    throw std::runtime_error(s.info.name + " has no sub-group support");
  s.use_sg = subgroups != 0 && s.info.subgroups;
  if (s.use_sg) {
    try {
      s.kernel("norms_kernel", "-DVS_NORMS");
    } catch (const std::exception&) {
      if (subgroups == 1) throw;
      s.use_sg = false;
    }
  }

  // A page is as many rows as fit in one allocation (a whole number of tiles), and the distance
  // block is the same budget as CUDA's unless the device's allocation limit is smaller.
  // VECSEARCH_OPENCL_MAX_ALLOC_MB lowers the limit, so tests can exercise paging on small data.
  size_t max_alloc = s.info.max_alloc;
  if (const char* cap = std::getenv("VECSEARCH_OPENCL_MAX_ALLOC_MB"))
    max_alloc = std::min(max_alloc, size_t(std::atoll(cap)) << 20);
  s.page_rows = std::max<size_t>(kTile, max_alloc / (s.dp * sizeof(float)) / kTile * kTile);
  s.dist_budget = std::max(std::min(kDistBudget, max_alloc / sizeof(float) / 2), kTile * kMaxQueryTile);
}

OpenCLBruteForceIndex::~OpenCLBruteForceIndex() = default;

size_t OpenCLBruteForceIndex::size() const { return impl_->n; }
size_t OpenCLBruteForceIndex::dim() const { return impl_->dim; }
Metric OpenCLBruteForceIndex::metric() const { return impl_->metric; }
const OpenCLDeviceInfo& OpenCLBruteForceIndex::device_info() const { return impl_->info; }
bool OpenCLBruteForceIndex::uses_subgroups() const { return impl_->use_sg; }

std::map<std::string, double> OpenCLBruteForceIndex::take_profile() {
  std::lock_guard<std::mutex> g(impl_->mu);
  return impl_->take_profile();
}

void OpenCLBruteForceIndex::add(const float* data, size_t m) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> g(s.mu);
  if (m == 0) return;
  if (s.n + m >= kNoId) throw std::length_error("OpenCLBruteForceIndex holds at most 2^32 - 2 vectors");
  size_t left = m;
  while (left > 0) {
    if (s.pages.empty() || s.pages.back().rows == s.page_rows) s.pages.emplace_back();
    Impl::Page& p = s.pages.back();
    const size_t take = std::min(left, s.page_rows - p.rows);
    if (p.rows + take > p.x.capacity() / s.dp) {
      const size_t cap =
          std::min(s.page_rows, std::max({p.rows + take, p.x.capacity() / s.dp * 2, size_t(1024)}));
      p.x.reserve(s.qu, cap * s.dp, p.rows * s.dp);
      p.xn.reserve(s.qu, cap, p.rows);
    }
    s.upload(p.x.get(), p.rows, data + (m - left) * s.dim, take);
    s.norms(p.x.get(), p.rows, take, p.xn.get(), p.rows);
    p.rows += take;
    s.n += take;
    left -= take;
  }
  CL_CHECK(clFinish(s.qu.q));
}

void OpenCLBruteForceIndex::search(const float* queries, size_t nq, size_t k, uint32_t* out_ids,
                                   float* out_dists, GpuMethod method) {
  if (nq == 0 || k == 0) return;
  const int K = select_bucket(k);
  Impl& s = *impl_;
  std::lock_guard<std::mutex> g(s.mu);
  if (s.n == 0) {
    std::fill(out_ids, out_ids + nq * k, kNoId);
    std::fill(out_dists, out_dists + nq * k, INFINITY);
    return;
  }
  const GpuMethod m =
      method != GpuMethod::Auto ? method : nq <= kSkinnyMaxBatch ? GpuMethod::Skinny : GpuMethod::Tiled;

  // Queries go in rounds of up to qt_max; the data in chunks sized so a round's distance
  // block fits the budget, and no chunk crosses a page. Each chunk leaves nseg * K candidates
  // per query.
  const size_t qt_max = std::min(nq, kMaxQueryTile);
  const size_t chunk = std::min(s.page_rows, std::max<size_t>(kTile, s.dist_budget / qt_max / kTile * kTile));
  const size_t seg = s.seg_len(qt_max, chunk);
  size_t width = 0;
  for (const Impl::Page& p : s.pages)
    for (size_t c0 = 0; c0 < p.rows; c0 += chunk) width += ceil_div(std::min(chunk, p.rows - c0), seg) * K;

  s.q.reserve(s.qu, qt_max * s.dp);
  s.qn.reserve(s.qu, qt_max);
  s.dist.reserve(s.qu, qt_max * chunk);
  for (int b = 0; b < 2; ++b) {
    s.cand_d[b].reserve(s.qu, qt_max * width);
    s.cand_i[b].reserve(s.qu, qt_max * width);
  }

  for (size_t q0 = 0; q0 < nq; q0 += qt_max) {
    const size_t qt = std::min(qt_max, nq - q0);
    // Padding columns of s.q were zeroed when it was allocated and are never written.
    s.upload(s.q.get(), 0, queries + q0 * s.dim, qt);
    if (m == GpuMethod::Tiled && s.metric == Metric::L2) s.norms(s.q.get(), 0, qt, s.qn.get(), 0);

    size_t cols = 0, base = 0;  // base = global id of the current page's first row
    for (const Impl::Page& p : s.pages) {
      for (size_t c0 = 0; c0 < p.rows; c0 += chunk) {
        const size_t len = std::min(chunk, p.rows - c0);
        s.distances(m, qt, p, c0, len, chunk);
        const size_t nseg = ceil_div(len, seg);
        // ids is unused here (the id is base + column); cand_i[1] is just a valid buffer to bind.
        s.select(K, nseg, qt, s.dist.get(), s.cand_i[1].get(), false, chunk, base + c0, len, seg,
                 s.cand_d[0].get(), s.cand_i[0].get(), width, cols);
        cols += nseg * K;
      }
      base += p.rows;
    }
    int src = 0;  // merge the candidates until one sorted list of K is left per query
    while (cols > size_t(K)) {
      const size_t sl = s.seg_len(qt, cols), nseg = ceil_div(cols, sl);
      s.select(K, nseg, qt, s.cand_d[src].get(), s.cand_i[src].get(), true, width, 0, cols, sl,
               s.cand_d[1 - src].get(), s.cand_i[1 - src].get(), width, 0);
      src = 1 - src;
      cols = nseg * K;
    }
    s.download(s.cand_i[src].get(), width, sizeof(uint32_t), out_ids + q0 * k, qt, k);
    s.download(s.cand_d[src].get(), width, sizeof(float), out_dists + q0 * k, qt, k);
  }
}

}  // namespace vecsearch
