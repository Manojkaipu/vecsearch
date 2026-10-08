// OpenCL C port of src/gpu_brute_force.cu. The host (src/opencl_brute_force.cpp) compiles one
// piece of this file at a time, picked with a -D define, so a search only pays to build the
// kernels it uses:
//   VS_NORMS   norms_kernel          VS_NAIVE  naive_kernel         VS_TILED  tiled_kernel
//   VS_QB=n    skinny_kernel (n queries per pass)                    VS_K=n    select_kernel
// VS_SUBGROUPS is also defined when the device has sub-groups, which swaps the local-memory
// reductions in norms_kernel and skinny_kernel for sub_group_reduce_add.
//
// CUDA -> OpenCL, as used here:
//   threadIdx.x / blockIdx.x / blockDim.x / gridDim.x  get_local_id(0) / get_group_id(0) /
//                                                      get_local_size(0) / get_num_groups(0)
//   __shared__, __syncthreads()      __local, barrier(CLK_LOCAL_MEM_FENCE)
//   warp (32 lanes), __shfl_xor_sync sub-group (8, 16 or 32 lanes, chosen by the compiler),
//                                    sub_group_reduce_add. The loops read get_sub_group_size()
//                                    instead of assuming 32, and use grid-stride loops so the host
//                                    never needs to know the size.
//   __launch_bounds__(n)             __attribute__((reqd_work_group_size(n, 1, 1)))
//   __ldg, __restrict__              const + restrict
//   fmaf, fmaxf                      fma, fmax
//   bool kernel arguments            int (OpenCL C has no bool arguments)
//   pointer + offset                 a separate offset argument (a cl_mem is a handle, not a pointer)
//   gridDim as a 2-D grid of blocks  a 2-D NDRange whose global size is groups * local size

#ifdef VS_SUBGROUPS
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define TILE 128   // tiled kernel: 128x128 outputs, 8 dims per step
#define TILE_K 8
#define SLOTS 4096  // candidates one select work-group sorts in local memory
#define NO_ID 0xFFFFFFFFu

#ifdef VS_NORMS
// Squared norm of each row, one sub-group (or one 32-lane slice of the group) per row.
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void norms_kernel(
    __global const float* restrict x, ulong row0, uint n, uint dp, __global float* restrict out,
    ulong out0) {
#ifdef VS_SUBGROUPS
  const uint lane = get_sub_group_local_id(), w = get_sub_group_size();
  const ulong nsg = (ulong)get_num_groups(0) * get_num_sub_groups();
  for (ulong row = (ulong)get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); row < n;
       row += nsg) {
    __global const float* r = x + (row0 + row) * dp;
    float s = 0.f;
    for (uint i = lane; i < dp; i += w) s = fma(r[i], r[i], s);
    s = sub_group_reduce_add(s);
    if (lane == 0) out[out0 + row] = s;
  }
#else
  __local float red[256];
  const uint lid = get_local_id(0), lane = lid % 32, vw = lid / 32;
  for (ulong rb = (ulong)get_group_id(0) * 8; rb < n; rb += (ulong)get_num_groups(0) * 8) {
    const ulong row = rb + vw;
    float s = 0.f;
    if (row < n) {
      __global const float* r = x + (row0 + row) * dp;
      for (uint i = lane; i < dp; i += 32) s = fma(r[i], r[i], s);
    }
    red[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = 16; o > 0; o >>= 1) {
      if (lane < o) red[lid] += red[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && row < n) out[out0 + row] = red[lid];
    barrier(CLK_LOCAL_MEM_FENCE);  // red is reused by the next row block
  }
#endif
}
#endif  // VS_NORMS

#ifdef VS_NAIVE
// The CPU loop as it is: one work-item per (query, vector) pair, both read from global memory.
// Neighbouring work-items read rows dp floats apart, so loads don't coalesce.
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void naive_kernel(
    __global const float* restrict q, __global const float* restrict x, ulong x_row0, uint nq,
    uint n, uint dp, int l2, __global float* restrict out, ulong ld) {
  const uint j = get_global_id(0);
  const uint i = get_global_id(1);
  if (j >= n || i >= nq) return;
  __global const float* a = q + (ulong)i * dp;
  __global const float* b = x + (x_row0 + j) * dp;
  float r = 0.f;
  if (l2) {
    for (uint t = 0; t < dp; ++t) {
      const float e = a[t] - b[t];
      r = fma(e, e, r);
    }
  } else {
    for (uint t = 0; t < dp; ++t) r = fma(a[t], b[t], r);
    r = 1.f - r;
  }
  out[(ulong)i * ld + j] = r;
}
#endif  // VS_NAIVE

#ifdef VS_QB
// One sub-group per vector: lanes read consecutive float4s (coalesced), each vector is read once
// for QB queries, and a sub-group reduction finishes each distance. Small batches are bound by
// reading the data, so this aims to stream it at full bandwidth.
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void skinny_kernel(
    __global const float4* restrict q, ulong q_off4, __global const float4* restrict x, ulong x_row0,
    uint n, uint dp, int l2, __global float* restrict out, ulong out_off, ulong ld) {
  const uint d4 = dp / 4;
  q += q_off4;
#ifdef VS_SUBGROUPS
  const uint lane = get_sub_group_local_id(), w = get_sub_group_size();
  const ulong nsg = (ulong)get_num_groups(0) * get_num_sub_groups();
  for (ulong j = (ulong)get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); j < n; j += nsg) {
    const bool valid = true;
#else
  __local float red[256];
  const uint lid = get_local_id(0), lane = lid % 32, w = 32, vw = lid / 32;
  for (ulong jb = (ulong)get_group_id(0) * 8; jb < n; jb += (ulong)get_num_groups(0) * 8) {
    const ulong j = jb + vw;
    const bool valid = j < n;
#endif
    __global const float4* x4 = x + (x_row0 + j) * d4;
    float acc[VS_QB];
#pragma unroll
    for (int b = 0; b < VS_QB; ++b) acc[b] = 0.f;
    if (valid) {
      for (uint t = lane; t < d4; t += w) {
        const float4 v = x4[t];
#pragma unroll
        for (int b = 0; b < VS_QB; ++b) {
          const float4 u = q[b * d4 + t];
          if (l2) {
            float e = u.x - v.x;
            acc[b] = fma(e, e, acc[b]);
            e = u.y - v.y;
            acc[b] = fma(e, e, acc[b]);
            e = u.z - v.z;
            acc[b] = fma(e, e, acc[b]);
            e = u.w - v.w;
            acc[b] = fma(e, e, acc[b]);
          } else {
            acc[b] = fma(u.x, v.x, acc[b]);
            acc[b] = fma(u.y, v.y, acc[b]);
            acc[b] = fma(u.z, v.z, acc[b]);
            acc[b] = fma(u.w, v.w, acc[b]);
          }
        }
      }
    }
#pragma unroll
    for (int b = 0; b < VS_QB; ++b) {
#ifdef VS_SUBGROUPS
      const float s = sub_group_reduce_add(acc[b]);
#else
      red[lid] = acc[b];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint o = 16; o > 0; o >>= 1) {
        if (lane < o) red[lid] += red[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      const float s = red[lid - lane];
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
      if (lane == 0 && valid) out[out_off + b * ld + j] = l2 ? s : 1.f - s;
    }
  }
}
#endif  // VS_QB

#ifdef VS_TILED
// q.x for a 128x128 tile of (query, vector) pairs: the classic local-memory SGEMM. Each step
// stages 8 dims of 128 queries and 128 vectors in local memory, and each of the 256 work-items
// accumulates an 8x8 block in registers, so every value loaded from local memory feeds 8 FMAs.
// The next step's global loads are issued before this step's math to hide their latency.
// L2 uses |q|^2 + |x|^2 - 2 q.x, so the data norms are computed once at add().
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void tiled_kernel(
    __global const float* restrict q, __global const float* restrict x, ulong x_row0, uint nq,
    uint n, uint dp, int l2, __global const float* restrict qn, __global const float* restrict xn,
    __global float* restrict out, ulong ld) {
  __local float qs[TILE_K * TILE] __attribute__((aligned(16)));  // [dim][query]
  __local float xs[TILE_K * TILE] __attribute__((aligned(16)));  // [dim][vector]
  const uint tid = get_local_id(0);
  const uint row0 = get_group_id(1) * TILE, col0 = get_group_id(0) * TILE;

  // Loading: each work-item fetches 4 consecutive dims of one query row and one data row.
  const uint lr = tid / 2, lk = (tid % 2) * 4;
  const bool q_ok = row0 + lr < nq, x_ok = col0 + lr < n;
  __global const float4* qp = (__global const float4*)(q + (ulong)(row0 + lr) * dp + lk);
  __global const float4* xp = (__global const float4*)(x + (x_row0 + col0 + lr) * dp + lk);
  const float4 zero = (float4)(0.f);
  float4 qv = q_ok ? qp[0] : zero;
  float4 xv = x_ok ? xp[0] : zero;

  // Computing: rows ty*4+{0..3} and 64+ty*4+{0..3}, columns likewise with tx. Splitting each
  // work-item's block in two halves keeps the float4 reads from local memory free of bank conflicts.
  const uint ty = tid / 16, tx = tid % 16;
  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

  for (uint k0 = 0; k0 < dp; k0 += TILE_K) {
    qs[(lk + 0) * TILE + lr] = qv.x;
    qs[(lk + 1) * TILE + lr] = qv.y;
    qs[(lk + 2) * TILE + lr] = qv.z;
    qs[(lk + 3) * TILE + lr] = qv.w;
    xs[(lk + 0) * TILE + lr] = xv.x;
    xs[(lk + 1) * TILE + lr] = xv.y;
    xs[(lk + 2) * TILE + lr] = xv.z;
    xs[(lk + 3) * TILE + lr] = xv.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k0 + TILE_K < dp) {
      qv = q_ok ? qp[(k0 + TILE_K) / 4] : zero;
      xv = x_ok ? xp[(k0 + TILE_K) / 4] : zero;
    }
#pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      const float4 a0 = vload4(kk * (TILE / 4) + ty, qs);
      const float4 a1 = vload4(kk * (TILE / 4) + 16 + ty, qs);
      const float4 b0 = vload4(kk * (TILE / 4) + tx, xs);
      const float4 b1 = vload4(kk * (TILE / 4) + 16 + tx, xs);
      const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (r >= nq) continue;
    const float qr = l2 ? qn[r] : 0.f;
    __global float* o = out + (ulong)r * ld;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint c = col0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
      if (c >= n) continue;
      const float s = acc[i][j];
      o[c] = l2 ? fmax(qr + xn[x_row0 + c] - 2.f * s, 0.f) : 1.f - s;
    }
  }
}
#endif  // VS_TILED

#ifdef VS_K
// Top K of one segment of one row. Each work-item keeps a sorted list of the K best distances it
// has seen; once the list is full most distances lose to its last entry, so inserts are rare.
// The lists (4096 entries in all) are then bitonic-sorted in local memory and the first K kept.
// use_ids == 0 means the id is id_base + column.
#define NTHREADS (SLOTS / VS_K)
__kernel __attribute__((reqd_work_group_size(NTHREADS, 1, 1))) void select_kernel(
    __global const float* restrict dist, __global const uint* restrict ids, int use_ids,
    ulong ld_in, uint id_base, uint len, uint seg_len, __global float* restrict out_d,
    __global uint* restrict out_i, ulong ld_out, uint out_col) {
  __local float sd[SLOTS];
  __local uint si[SLOTS];
  const uint tid = get_local_id(0);
  const ulong row = get_group_id(1);
  __global const float* d_row = dist + row * ld_in;
  __global const uint* i_row = ids + row * ld_in;
  const uint begin = get_group_id(0) * seg_len, end = min(len, begin + seg_len);

  float td[VS_K];
  uint ti[VS_K];
#pragma unroll
  for (int j = 0; j < VS_K; ++j) {
    td[j] = INFINITY;
    ti[j] = NO_ID;
  }
  for (uint c = begin + tid; c < end; c += NTHREADS) {
    float d = d_row[c];
    if (d < td[VS_K - 1]) {
      uint id = use_ids ? i_row[c] : id_base + c;
#pragma unroll
      for (int j = 0; j < VS_K; ++j) {
        if (d < td[j]) {
          const float t = td[j];
          td[j] = d;
          d = t;
          const uint u = ti[j];
          ti[j] = id;
          id = u;
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < VS_K; ++j) {  // order doesn't matter before the sort; this one avoids bank conflicts
    sd[j * NTHREADS + tid] = td[j];
    si[j * NTHREADS + tid] = ti[j];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint size = 2; size <= SLOTS; size <<= 1) {
    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      for (uint t = tid; t < SLOTS / 2; t += NTHREADS) {
        const uint lo = 2 * t - (t & (stride - 1)), hi = lo + stride;
        const bool ascending = (lo & size) == 0;
        const float a = sd[lo], b = sd[hi];
        const uint ia = si[lo], ib = si[hi];
        const bool a_after_b = a > b || (a == b && ia > ib);
        if (a_after_b == ascending) {
          sd[lo] = b;
          sd[hi] = a;
          si[lo] = ib;
          si[hi] = ia;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  const ulong o = row * ld_out + out_col + (ulong)get_group_id(0) * VS_K;
  for (uint j = tid; j < VS_K; j += NTHREADS) {
    out_d[o + j] = sd[j];
    out_i[o + j] = si[j];
  }
}
#endif  // VS_K
