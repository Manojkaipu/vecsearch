// Distance-kernel choice shared by the CUDA and OpenCL brute-force indexes.
#pragma once
#include <cstdint>

namespace vecsearch {

enum class GpuMethod : uint32_t {
  Auto = 0,    // Skinny for small batches, Tiled otherwise
  Naive = 1,   // one thread per (query, vector) pair: the AVX2 loop, ported as is
  Skinny = 2,  // one warp per vector, up to 8 queries at a time; for memory-bound small batches
  Tiled = 3,   // 128x128 register-blocked tiles of q.x (a hand-written SGEMM); for big batches
};

}  // namespace vecsearch
