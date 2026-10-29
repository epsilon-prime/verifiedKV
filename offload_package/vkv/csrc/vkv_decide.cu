#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cg = cooperative_groups;

namespace vkv {

constexpr int NT = 512;          // threads per decide block
constexpr int NW = NT / 32;
constexpr int KMAX = 4096;       // capacity of the sorted heads (t_max, k_max <= KMAX)
constexpr int NTE = 256;         // threads per emit block
constexpr int HC = KMAX / NT;    // head entries per thread (contiguous chunk)
constexpr int TKB = 8;           // keys a thread loads at once in the top-K sweeps

__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }

// float64 <-> uint64 with the same order (-0.0 as +0.0; never 0 for a non-NaN value).
__device__ __forceinline__ uint64_t okey(double v) {
  if (v == 0.0) v = 0.0;
  long long b = __double_as_longlong(v);
  b ^= ((b >> 63) & 0x7FFFFFFFFFFFFFFFLL);
  return (uint64_t)b ^ 0x8000000000000000ULL;
}
__device__ __forceinline__ double okey_inv(uint64_t u) {
  long long b = (long long)(u ^ 0x8000000000000000ULL);
  b ^= ((b >> 63) & 0x7FFFFFFFFFFFFFFFLL);
  return __longlong_as_double(b);
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <typename T>
__device__ __forceinline__ T warp_max(T v) {
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <typename T>
__device__ __forceinline__ T warp_min(T v) {
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}


template <int NTH, typename T>
__device__ T block_red(T v, int op, T ident, void* red_) {
  T* red = reinterpret_cast<T*>(red_);
  constexpr int W = NTH / 32;
  v = op == 0 ? warp_sum(v) : (op == 1 ? warp_max(v) : warp_min(v));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    T x = threadIdx.x < W ? red[threadIdx.x] : ident;
    x = op == 0 ? warp_sum(x) : (op == 1 ? warp_max(x) : warp_min(x));
    if (threadIdx.x == 0) red[W] = x;
  }
  __syncthreads();
  return red[W];
}
template <int NTH, typename T>
__device__ T block_sum(T v, void* red) { return block_red<NTH, T>(v, 0, T(0), red); }
template <int NTH, typename T>
__device__ T block_max(T v, T ident, void* red) { return block_red<NTH, T>(v, 1, ident, red); }
template <int NTH, typename T>
__device__ T block_min(T v, T ident, void* red) { return block_red<NTH, T>(v, 2, ident, red); }


template <int NTH>
__device__ int block_excl_scan(int v, int* red, int& total) {
  constexpr int W = NTH / 32;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  int x = v;
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) red[w] = x;
  __syncthreads();
  if (threadIdx.x < 32) {
    int t = threadIdx.x < W ? red[threadIdx.x] : 0;
    int s = t;
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += y;
    }
    if (threadIdx.x < W) red[threadIdx.x] = s - t;
    if (threadIdx.x == 31) red[W] = s;
  }
  __syncthreads();
  int r = red[w] + x - v;
  total = red[W];
  __syncthreads();
  return r;
}

// Inclusive scan of a[0, L) float64 in shared memory, in place; returns the total.
__device__ double block_scan_f64(double* a, int L, double* red) {
  const int C = (L + NT - 1) / NT;
  const int lo = threadIdx.x * C, hi = min(L, lo + C);
  double s = 0.0;
  for (int i = lo; i < hi; ++i) { s += a[i]; a[i] = s; }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  double x = s;
  for (int o = 1; o < 32; o <<= 1) {
    double y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) red[w] = x;
  __syncthreads();
  if (threadIdx.x < 32) {
    double t = threadIdx.x < NW ? red[threadIdx.x] : 0.0;
    double u = t;
    for (int o = 1; o < 32; o <<= 1) {
      double y = __shfl_up_sync(0xffffffffu, u, o);
      if (lane >= o) u += y;
    }
    if (threadIdx.x < NW) red[threadIdx.x] = u - t;
    if (threadIdx.x == NW - 1) red[NW] = u;
  }
  __syncthreads();
  const double off = red[w] + (x - s);
  for (int i = lo; i < hi; ++i) a[i] += off;
  const double tot = red[NW];
  __syncthreads();
  return tot;
}

// Sums over the threads after this one (exclusive suffix), of two values at once.
__device__ __forceinline__ void block_rexcl2(double a, double b, double& ra, double& rb,
                                             double* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  double xa = a, xb = b;
  for (int o = 1; o < 32; o <<= 1) {
    const double ya = __shfl_down_sync(0xffffffffu, xa, o);
    const double yb = __shfl_down_sync(0xffffffffu, xb, o);
    if (lane + o < 32) { xa += ya; xb += yb; }
  }
  __syncthreads();
  if (lane == 0) { red[w] = xa; red[NW + w] = xb; }
  __syncthreads();
  double wa = 0.0, wb = 0.0;
  for (int i = w + 1; i < NW; ++i) { wa += red[i]; wb += red[NW + i]; }
  ra = wa + (xa - a);
  rb = wb + (xb - b);
  __syncthreads();
}

// Sums over the threads after this one (exclusive suffix) of NV values (red: NV * NW slots).
template <int NV>
__device__ __forceinline__ void block_rexcl(const double (&v)[NV], double (&r)[NV], double* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  double x[NV];
#pragma unroll
  for (int i = 0; i < NV; ++i) x[i] = v[i];
  for (int o = 1; o < 32; o <<= 1) {
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      const double y = __shfl_down_sync(0xffffffffu, x[i], o);
      if (lane + o < 32) x[i] += y;
    }
  }
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NV; ++i) red[i * NW + w] = x[i];
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    double wsum = 0.0;
    for (int u = w + 1; u < NW; ++u) wsum += red[i * NW + u];
    r[i] = wsum + (x[i] - v[i]);
  }
  __syncthreads();
}

// ---- the row's cluster ------------------------------------------------------------------------------

template <int CS>
struct Row {
  cg::cluster_group cl;
  int cr;            // block rank in the row's cluster
  double* xs;        // [8 slots, 8] exchange ring (this block's)
  int slot;
  __device__ Row() : cl(cg::this_cluster()), cr(CS > 1 ? (int)cg::this_cluster().block_rank() : 0),
                     xs(nullptr), slot(0) {}
  __device__ __forceinline__ void sync() {
    if (CS > 1) cl.sync();
    else __syncthreads();
  }
  template <typename P>
  __device__ __forceinline__ P* at(P* p, int r) {
    return CS > 1 ? cl.map_shared_rank(p, r) : p;
  }
  // block reductions, then the same combination over the cluster in rank order
  template <int NV>
  __device__ void allreduce(double (&v)[NV], const int (&op)[NV], double* red) {
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      const double id = op[i] == 0 ? 0.0 : (op[i] == 1 ? -INFINITY : INFINITY);
      v[i] = block_red<NT, double>(v[i], op[i], id, red);
    }
    if (CS == 1) return;
    double* mine = xs + slot * 8;
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < NV; ++i) mine[i] = v[i];
    }
    cl.sync();
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      double acc = op[i] == 0 ? 0.0 : (op[i] == 1 ? -INFINITY : INFINITY);
      for (int r = 0; r < CS; ++r) {
        const double x = cl.map_shared_rank(mine, r)[i];
        acc = op[i] == 0 ? acc + x : (op[i] == 1 ? fmax(acc, x) : fmin(acc, x));
      }
      v[i] = acc;
    }
    slot = (slot + 1) & 7;
  }
  __device__ double sum(double v, double* red) { double a[1] = {v}; const int o[1] = {0}; allreduce<1>(a, o, red); return a[0]; }
  __device__ double max(double v, double* red) { double a[1] = {v}; const int o[1] = {1}; allreduce<1>(a, o, red); return a[0]; }
  __device__ double min(double v, double* red) { double a[1] = {v}; const int o[1] = {2}; allreduce<1>(a, o, red); return a[0]; }
  // per-block values v[0..NV) (already block totals, uniform in the block): the sums over the
  // blocks before this one, and the totals
  template <int NV>
  __device__ void exscan(const double (&v)[NV], double (&before)[NV], double (&total)[NV]) {
    if (CS == 1) {
#pragma unroll
      for (int i = 0; i < NV; ++i) { before[i] = 0.0; total[i] = v[i]; }
      return;
    }
    double* mine = xs + slot * 8;
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < NV; ++i) mine[i] = v[i];
    }
    cl.sync();
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      double b = 0.0, t = 0.0;
      for (int r = 0; r < CS; ++r) {
        const double x = cl.map_shared_rank(mine, r)[i];
        if (r < cr) b += x;
        t += x;
      }
      before[i] = b;
      total[i] = t;
    }
    slot = (slot + 1) & 7;
  }
};

// ---- top-K over the row (the cluster): keys of positions [p0, p1) in this block -------------------------

template <int CS>
struct Layout {               // dynamic shared memory
  static constexpr int SLICE = ((KMAX + CS - 1) / CS + NT - 1) / NT * NT;   // per-block sort slice
  static constexpr size_t CK = 0, CP = CK + KMAX * 8, SK = CP + KMAX * 4, SP = SK + SLICE * 8,
                          F64 = ((SP + SLICE * 4 + 15) / 16) * 16,     // f64a, f64b / CUB temp
                          HIST = F64 + KMAX * 16, RED = HIST + 768 * 4, XS = RED + (4 * NW + 2) * 8,
                          QS = XS + 64 * 8, QSK = QS + 128 * 4, ZS = QSK + 64 * 4,
                          WHIST = ZS + NW * 128 * 4, BYTES = WHIST + NW * 256 * 4;
};

struct TopkSmem {
  uint64_t* ck;     // [KMAX] the result (every block): sorted keys (sort) / the lead's candidates
  int32_t* cp;      // [KMAX] positions
  uint64_t* sk;     // [KMAX / CS] this block's sorted slice
  int32_t* sp;
  int* hist;        // [3, 256]: two passes' histograms, the merged one
  int* ired;        // >= 2 NW + 2 ints
  void* cub;        // CUB temp storage
  int* whist;       // [NW, 256] per-warp histograms
};

// Radix select of the row's K-th largest key: the threshold's top bits ``pref`` (key >> shift),
// and krem = how many of the keys equal to it (at that resolution) are among the K.
template <int CS, typename KeyF>
__device__ void radix_select(Row<CS>& rw, KeyF keyf, int p0, int p1, int K, TopkSmem& t,
                             uint64_t& pref, int& shift, int& krem) {
  __shared__ int s_dsel, s_cd, s_above;
  __syncthreads();                                       // the keys are written
  pref = 0;
  shift = 64;
  krem = K;
  bool done = false;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int p = 0; p < 8 && !done; ++p) {
    shift -= 8;
    int* hs = t.hist + (p & 1) * 256;                    // double-buffered: one cluster sync a pass
    int* wh_ = t.whist;                                  // per-warp copies, summed into hs
    for (int i = threadIdx.x; i < NW * 256; i += NT) wh_[i] = 0;
    __syncthreads();
    for (int base = p0; base < p1; base += TKB * NT) {
      uint64_t u[TKB];
#pragma unroll
      for (int q = 0; q < TKB; ++q) {                    // the loads first, then the warp syncs
        const int pos = base + q * NT + threadIdx.x;
        u[q] = pos < p1 ? keyf(pos) : 0;
      }
#pragma unroll
      for (int q = 0; q < TKB; ++q) {
        const int pos = base + q * NT + threadIdx.x;
        if (pos < p1 && (p == 0 || (u[q] >> (shift + 8)) == pref)) {
          const int bin = (int)((u[q] >> shift) & 255);
          // lanes agreeing with lane 0's bin (the common case: most keys share a few bins) add
          // once; the rest add per lane into the warp's own copy
          atomicAdd(&wh_[w * 256 + bin], 1);
        }
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += NT) {
      int x = 0;
#pragma unroll
      for (int u2 = 0; u2 < NW; ++u2) x += wh_[u2 * 256 + i];
      hs[i] = x;
    }
    rw.sync();
    int* hm = t.hist + 512;                              // the cluster's merged histogram
    if (CS > 1) {
      for (int i = threadIdx.x; i < 256; i += NT) {
        int x = 0;
#pragma unroll
        for (int r = 0; r < CS; ++r) x += rw.at(hs, r)[i];
        hm[i] = x;
      }
      __syncthreads();
    } else {
      hm = hs;
    }
    if (threadIdx.x < 32) {
      int c[8], s = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = hm[lane * 8 + i];
        s += c[i];
      }
      int x = s;
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_down_sync(0xffffffffu, x, o);
        if (lane + o < 32) x += y;
      }
      const int above = x - s;
      int best = -1, ge = above;
      for (int i = 7; i >= 0; --i) {
        ge += c[i];
        if (ge >= krem && best < 0) best = lane * 8 + i;
      }
      const int d = warp_max(best);
      if (lane == (d >> 3)) {
        int g2 = above;
        for (int i = 7; i >= (d & 7); --i) g2 += c[i];
        s_dsel = d;
        s_cd = c[d & 7];
        s_above = g2 - c[d & 7];
      }
    }
    __syncthreads();
    const int d = s_dsel, cd = s_cd;
    krem -= s_above;
    pref = (p == 0) ? (uint64_t)d : ((pref << 8) | (uint64_t)d);
    done = (cd == krem);
    __syncthreads();
  }
}

// The row's positions in this block's [p0, p1), in the order the compaction and the marking
// count them (block rank, then warp segment, then lane), and each warp's first offsets of the
// keys above the threshold (og) and equal to it (oe) among the whole row.
template <int CS, typename KeyF>
__device__ void class_offsets(Row<CS>& rw, KeyF keyf, int p0, int p1, uint64_t pref, int shift,
                              TopkSmem& t, int& s0, int& s1, int& og, int& oe) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int n_b = max(0, p1 - p0);
  const int seg = ((n_b + NW - 1) / NW + 31) & ~31;
  s0 = p0 + w * seg;
  s1 = min(p1, s0 + seg);
  int ng = 0, ne = 0;
  for (int base = s0; base < s1; base += 32 * TKB) {
    uint64_t v[TKB];
#pragma unroll
    for (int q = 0; q < TKB; ++q) {
      const int pos = base + q * 32 + lane;
      v[q] = pos < s1 ? keyf(pos) >> shift : 0;
    }
#pragma unroll
    for (int q = 0; q < TKB; ++q) {
      const bool in = base + q * 32 + lane < s1;
      ng += __popc(__ballot_sync(0xffffffffu, in && v[q] > pref));
      ne += __popc(__ballot_sync(0xffffffffu, in && v[q] == pref));
    }
  }
  if (lane == 0) { t.ired[w] = ng; t.ired[NW + w] = ne; }
  __syncthreads();
  og = 0; oe = 0;
  int tg = 0, te = 0;
  for (int i = 0; i < NW; ++i) {
    if (i < w) { og += t.ired[i]; oe += t.ired[NW + i]; }
    tg += t.ired[i];
    te += t.ired[NW + i];
  }
  {
    const double v[2] = {(double)tg, (double)te};
    double bf[2], tot[2];
    rw.template exscan<2>(v, bf, tot);
    og += (int)bf[0];
    oe += (int)bf[1];
  }
}

// The row's top-K positions, unsorted: f(pos) for each (ties at the threshold by position, as in
// topk). Every block calls it for its own [p0, p1).
template <int CS, typename KeyF, typename F>
__device__ void mark_top(Row<CS>& rw, KeyF keyf, int p0, int p1, int K, TopkSmem& t, F f) {
  uint64_t pref;
  int shift, krem;
  radix_select<CS>(rw, keyf, p0, p1, K, t, pref, shift, krem);
  int s0, s1, og, oe;
  class_offsets<CS>(rw, keyf, p0, p1, pref, shift, t, s0, s1, og, oe);
  const int G0 = K - krem, lane = threadIdx.x & 31;
  const unsigned lt = (1u << lane) - 1u;
  for (int base = s0; base < s1 && (og < G0 || oe < krem); base += 32) {
    const int pos = base + lane;
    const uint64_t v = pos < s1 ? keyf(pos) >> shift : 0;
    const bool gt = pos < s1 && v > pref, eq = pos < s1 && v == pref;
    const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
    if (gt || (eq && oe + __popc(be & lt) < krem)) f(pos);
    og += __popc(bg);
    oe += __popc(be);
  }
  rw.sync();
}

template <int CS, typename KeyF>
__device__ void topk(Row<CS>& rw, KeyF keyf, int p0, int p1, int K, bool sort, TopkSmem& t) {
  uint64_t pref;
  int shift, krem;
  radix_select<CS>(rw, keyf, p0, p1, K, t, pref, shift, krem);
  const int lane = threadIdx.x & 31;
  // Order-preserving compaction into the lead block: above the threshold class, then krem of its
  // equals by ascending position. Positions in order: block rank, then warp segment, then lane.
  const int G0 = K - krem;
  int s0, s1, og, oe;
  class_offsets<CS>(rw, keyf, p0, p1, pref, shift, t, s0, s1, og, oe);
  uint64_t* ckL = rw.at(t.ck, 0);
  int32_t* cpL = rw.at(t.cp, 0);
  const unsigned lt = (1u << lane) - 1u;
  for (int base = s0; base < s1 && (og < G0 || oe < krem); base += 32 * TKB) {
    uint64_t uu[TKB];
#pragma unroll
    for (int q = 0; q < TKB; ++q) {
      const int pos = base + q * 32 + lane;
      uu[q] = pos < s1 ? keyf(pos) : 0;
    }
#pragma unroll
    for (int q = 0; q < TKB; ++q) {
      const int pos = base + q * 32 + lane;
      const uint64_t u = uu[q], v = u >> shift;
      const bool gt = pos < s1 && v > pref, eq = pos < s1 && v == pref;
      const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
      if (gt) { const int i = og + __popc(bg & lt); ckL[i] = u; cpL[i] = pos; }
      if (eq) {
        const int i = G0 + oe + __popc(be & lt);
        if (i < K) { ckL[i] = u; cpL[i] = pos; }
      }
      og += __popc(bg);
      oe += __popc(be);
    }
  }
  rw.sync();
  if (!sort) return;
  // Sort: block rank c sorts candidates [c SL, c SL + SL) (CUB, stable, descending: equal keys
  // keep the compaction's position order), then places each by binary search in the others.
  constexpr int SLICE = Layout<CS>::SLICE, IPT = SLICE / NT;
  int NP = 2;
  while (NP < K) NP <<= 1;
  const int SL = (NP + CS - 1) / CS;
  const int lo = rw.cr * SL;
  uint64_t key[IPT];
  int32_t val[IPT];
#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    const int i = threadIdx.x * IPT + j;
    const int gi = lo + i;
    if (i < SL && gi < K) { key[j] = ckL[gi]; val[j] = cpL[gi]; }
    else { key[j] = 0; val[j] = 0x7FFFFFFF; }
  }
  rw.sync();                                             // every slice read from the lead
  using BRS = cub::BlockRadixSort<uint64_t, NT, IPT, int32_t>;
  BRS(*reinterpret_cast<typename BRS::TempStorage*>(t.cub)).SortDescending(key, val);
#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    t.sk[threadIdx.x * IPT + j] = key[j];
    t.sp[threadIdx.x * IPT + j] = val[j];
  }
  const int mylen = max(0, min(SL, K - lo));
  if (CS == 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < K; i += NT) { t.ck[i] = t.sk[i]; t.cp[i] = t.sp[i]; }
    __syncthreads();
    return;
  }
  rw.sync();
  // every slice, copied here (the CUB region is free again): the searches stay local
  uint64_t* allk = reinterpret_cast<uint64_t*>(t.cub);
  int32_t* allp = reinterpret_cast<int32_t*>(allk + KMAX);
  for (int o = 0; o < CS; ++o) {
    const int olen = max(0, min(SL, K - o * SL));
    const uint64_t* rk = rw.at(t.sk, o);
    const int32_t* rp = rw.at(t.sp, o);
    for (int i = threadIdx.x; i < olen; i += NT) { allk[o * SL + i] = rk[i]; allp[o * SL + i] = rp[i]; }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < mylen; i += NT) {
    const uint64_t ek = t.sk[i];
    const int32_t ep = t.sp[i];
    int gi = i;
    for (int o = 0; o < CS; ++o) {
      if (o == rw.cr) continue;
      const int olen = max(0, min(SL, K - o * SL));
      const uint64_t* ok = allk + o * SL;
      const int32_t* opp = allp + o * SL;
      int a = 0, b = olen;                       // first index not before e
      while (a < b) {
        const int mid = (a + b) >> 1;
        const uint64_t mk = ok[mid];
        const bool before = (mk > ek) || (mk == ek && opp[mid] < ep);
        if (before) a = mid + 1;
        else b = mid;
      }
      gi += a;
    }
    for (int r = 0; r < CS; ++r) {
      rw.at(t.ck, r)[gi] = ek;
      rw.at(t.cp, r)[gi] = ep;
    }
  }
  rw.sync();
}

// ---- parameters ----------------------------------------------------------------------------------

struct Params {
  const void* q;            // [B, Hq, D] (T)
  float sm_scale;
  int B, H, G, Hq, R, n, WA, max_len, M_sl, M_t, Kc, NG, bennett;
  int HH;                   // the certificate's exact head (>= Kc; kstar_fast's head)
  int row0;                 // first row of this launch (rows launched in chunks of KV heads)
  const int64_t* slots;     // [B]
  const float* r;           // [S, max_len, H]
  const int32_t* cell;      // [S, max_len, H]
  const int32_t* cpu_row;   // [S, max_len, H]
  const int32_t* alist;     // [S, H, max_len]
  const int32_t* acnt;      // [S, H]
  const void* kv;           // [cells, C2] (T)
  const void* pool;         // [rows, PC2] (T, pinned, device-mapped)
  int C2;
  long long PC2;
  const uint8_t* codes;     // layer slice [S, H, max_len, NB]
  long long c_s, c_h;
  const __half* zero;       // layer slice [S, H, ng, NCH]
  const __half* scale;
  long long z_s, z_h;
  const int64_t* channels;  // layer slice [S, H, NCH]
  long long ch_s, ch_h;
  int NCH, NB, PER, BITS, GROUP;
  double eps0, delta, thr, L, third;
  const double* grid;       // [NG]
  float* s_hat;             // [R, n]
  double* jbuf;             // [R, n] scratch
  double* wh;               // [R, n] scratch: exp(s_hat - m) on the tail
  int8_t* Tm;               // [R, n]
  int32_t* rank;            // [R, n]
  float* sA;                // [R, WA]
  float* rA;
  int32_t* pA;
  int64_t* physA;
  int32_t* o_i;             // [R, Kc]
  float* m;
  double* sure;
  int64_t* kst;
  double* beta_k;
  double* Bt;
  int8_t* nonmono;
  double* tsize;
  int8_t* scr;
  bool* fb;
  int32_t* fbl;             // [R + 1], fbl[R] zeroed
  int32_t* keys_cnt;        // [1] zeroed
  int32_t* bar;             // [B * H] zeroed
  unsigned long long* tprof;  // VKV_TPROF builds: [TP_N] per-phase cycle sums (see TP_MARK)
};

// VKV_TPROF builds: the lead block of every row adds each phase's SM cycles (thread 0's clock64
// between consecutive marks) to tprof[phase]; tprof[TP_ROWS] counts rows, tprof[TP_UNSCR] the
// rows that ran the certificate. Other builds compile the marks away.
constexpr int TP_N = 20, TP_ROWS = 18, TP_UNSCR = 19;
#ifdef VKV_TPROF
#define TP_MARK(i)                                                          \
  do {                                                                      \
    if (tp_on) {                                                            \
      const long long tp_now = clock64();                                   \
      atomicAdd(p.tprof + (i), (unsigned long long)(tp_now - tp_last));     \
      tp_last = tp_now;                                                     \
    }                                                                       \
  } while (0)
#else
#define TP_MARK(i) do {} while (0)
#endif

__device__ __forceinline__ void unit_barrier(int32_t* bar, int target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(bar, 1);
    while (atomicAdd(bar, 0) < target) { __nanosleep(64); }
    __threadfence();
  }
  __syncthreads();
}

// Per-entry quantities of reference.prepare (Triton ``_entry``), wh = exp(f64(s - m)) on the tail
// (computed once, phase 5, and cached).
struct Entry {
  double rd, wh;
  bool tail, posm, r0;
};
__device__ __forceinline__ Entry entry_c(float rv, double wh) {
  Entry e;
  e.tail = rv < 1.0f;
  e.wh = wh;
  e.rd = (double)rv;
  e.posm = e.tail && (rv > 0.0f);
  e.r0 = e.tail && (rv <= 0.0f);
  return e;
}
struct Moves {
  bool T;
  double b, v, up, dn;
};
__device__ __forceinline__ Moves moves(const Entry& e, bool tm) {
  Moves o;
  o.T = tm || e.r0;
  const bool Tp = o.T && e.posm, Hp = e.posm && !o.T;
  const double rr = fmax(e.rd, 1e-300);
  const double om = 1.0 - e.rd;
  o.b = o.T ? e.wh * om : 0.0;
  o.v = Tp ? e.wh * e.wh * e.rd * om : (Hp ? e.wh * e.wh * om / rr : 0.0);
  o.up = Tp ? e.wh * om : (Hp ? e.wh * om / rr : 0.0);
  o.dn = Tp ? e.wh * e.rd : (Hp ? e.wh : 0.0);
  return o;
}
__device__ __forceinline__ double tail1(double u, double V, double bm) {
  const double den = 2.0 * (V + bm * u / 3.0);
  const double e = den > 0.0 ? -(u * u) / fmax(den, 1e-300) : -INFINITY;
  return u > 0.0 ? exp(e) : 1.0;
}
__device__ __forceinline__ float lae32(float x, float y) {
  return fmaxf(x, y) + log1pf(expf(-fabsf(x - y)));
}
__device__ __forceinline__ double bennett(double lam, double Vr, double M) {
  const double x = fmin(lam * M, 700.0);
  return Vr > 0.0 ? Vr * (expm1(x) - x) / (M * M) : 0.0;
}

// q . k over D elements, 4 threads per key (each D / 4 contiguous elements, 16-byte loads).
template <typename T, int D>
__device__ __forceinline__ float dot4(const T* k, const float* qs, int sub) {
  constexpr int E = D / 4;
  const uint4* k4 = reinterpret_cast<const uint4*>(k + sub * E);
  float s = 0.0f;
#pragma unroll
  for (int v = 0; v < E / 8; ++v) {
    const uint4 x = k4[v];
    const T* e = reinterpret_cast<const T*>(&x);
#pragma unroll
    for (int t = 0; t < 8; ++t) s += to_f(e[t]) * qs[sub * E + v * 8 + t];
  }
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  return s;
}


template <typename T, int D, int CS>
__global__ void __launch_bounds__(NT, 1) decide_kernel(Params p) {
  extern __shared__ __align__(16) unsigned char smem[];
  using LY = Layout<CS>;
  using BRS = cub::BlockRadixSort<uint64_t, NT, LY::SLICE / NT, int32_t>;
  static_assert(sizeof(typename BRS::TempStorage) <= KMAX * 16, "CUB temp storage");
  TopkSmem ts;
  ts.ck = reinterpret_cast<uint64_t*>(smem + LY::CK);
  ts.cp = reinterpret_cast<int32_t*>(smem + LY::CP);
  ts.sk = reinterpret_cast<uint64_t*>(smem + LY::SK);
  ts.sp = reinterpret_cast<int32_t*>(smem + LY::SP);
  ts.hist = reinterpret_cast<int*>(smem + LY::HIST);
  double* red = reinterpret_cast<double*>(smem + LY::RED);
  ts.ired = reinterpret_cast<int*>(red);
  ts.cub = smem + LY::F64;
  ts.whist = reinterpret_cast<int*>(smem + LY::WHIST);
  double* f64a = reinterpret_cast<double*>(smem + LY::F64);
  double* f64b = f64a + KMAX;
  float* qs = reinterpret_cast<float*>(smem + LY::QS);
  float* qsk = reinterpret_cast<float*>(smem + LY::QSK);
  float* zs = reinterpret_cast<float*>(smem + LY::ZS);
  Row<CS> rw;
  rw.xs = reinterpret_cast<double*>(smem + LY::XS);
  const int row = p.row0 + blockIdx.x / CS, cr = rw.cr;
  const bool lead = cr == 0;
  const int H = p.H, G = p.G, n = p.n, WA = p.WA, max_len = p.max_len;
  const int b = row / (H * G), h = (row / G) % H, g = row % G;
  const int unit = b * H + h;
  const long long slot = p.slots[b];
  const long long rb = (long long)row * n;
  const T* qrow = reinterpret_cast<const T*>(p.q) + (long long)row * D;
  const T* kv = reinterpret_cast<const T*>(p.kv);
  const T* pool = reinterpret_cast<const T*>(p.pool);
  const float* rS = p.r + slot * max_len * H;          // r[slot], index pos * H + h
  const int32_t* cellS = p.cell + slot * max_len * H;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  double* WH = p.wh + rb;
  double* J = p.jbuf + rb;
  int8_t* Tm = p.Tm + rb;
  int32_t* rank = p.rank + rb;
  // this block's context segment
  const int seg = (((n + CS - 1) / CS) + 31) & ~31;
  const int p0 = min(n, cr * seg), p1 = min(n, p0 + seg);
#ifdef VKV_TPROF
  const bool tp_on = p.tprof != nullptr && lead && tid == 0;
  long long tp_last = clock64();
  if (tp_on) atomicAdd(p.tprof + TP_ROWS, 1ull);
#endif

  for (int d = tid; d < D; d += NT) qs[d] = to_f(qrow[d]) * p.sm_scale;
  __syncthreads();
  const int64_t* ch = p.channels + slot * p.ch_s + h * p.ch_h;
  for (int c = tid; c < p.NCH; c += NT) qsk[c] = qs[ch[c]];
  for (int i = p0 + tid; i < p1; i += NT) { Tm[i] = 0; rank[i] = 0; }
  __syncthreads();
  TP_MARK(0);                                            // init

  // ---- 1 exact logits over A ------------------------------------------------------------------------
  const int cnt = p.acnt[slot * H + h];
  const int32_t* al = p.alist + (slot * H + h) * (long long)max_len;
  double mA;
  {
    float mx = -INFINITY;
    const int sub = tid & 3;
    for (int j0 = cr * (NT / 4); j0 < cnt; j0 += CS * (NT / 4)) {
      const int j = j0 + (tid >> 2);
      const bool ok = j < cnt;
      int pos = 0;
      long long ph = 0;
      float rv = 1.0f;
      if (ok) {
        pos = al[j];
        const long long mi = (long long)pos * H + h;
        ph = cellS[mi];
        rv = rS[mi];
      }
      const float s = dot4<T, D>(kv + (ok ? ph : 0) * p.C2, qs, sub);
      if (ok && sub == 0) {
        const long long o = (long long)row * WA + j;
        p.sA[o] = s;
        p.rA[o] = rv;
        p.pA[o] = pos;
        p.physA[o] = ph;
        if (rv >= 1.0f) mx = fmaxf(mx, s);
      }
    }
    mA = rw.max((double)mx, red);
  }
  TP_MARK(1);                                            // 1 exact logits over A

  // ---- 2 sketch logits: warp-sized groups of positions sharing zero / scale ----------------------
  {
    const uint8_t* cb = p.codes + slot * p.c_s + h * p.c_h;
    const __half* zb = p.zero + slot * p.z_s + h * p.z_h;
    const __half* sb = p.scale + slot * p.z_s + h * p.z_h;
    const int lev = (1 << p.BITS) - 1;
    const int lp = __ffs(p.PER) - 1, pm = p.PER - 1;
    const int GR = p.GROUP, ngr = (n + GR - 1) / GR;
    float* zw = zs + w * 128;
    float* sw = zw + 64;
    for (int gi = cr * NW + w; gi < ngr; gi += CS * NW) {
      const long long gb = (long long)gi * p.NCH;
      for (int c = lane; c < p.NCH; c += 32) {
        zw[c] = __half2float(zb[gb + c]);
        sw[c] = __half2float(sb[gb + c]);
      }
      __syncwarp();
      const int e1 = min(n, (gi + 1) * GR);
      for (int pos = gi * GR + lane; pos < e1; pos += 32) {
        uint32_t by[4];
        const uint8_t* src = cb + (long long)pos * p.NB;
        for (int i = 0; i < 4; ++i) by[i] = 0;
        for (int i = 0; i < p.NB; ++i) by[i >> 2] |= (uint32_t)src[i] << ((i & 3) * 8);
        float acc = 0.0f;
#pragma unroll 8
        for (int c = 0; c < p.NCH; ++c) {
          const int bi = c >> lp;
          const int code = (by[bi >> 2] >> ((bi & 3) * 8 + (c & pm) * p.BITS)) & lev;
          acc += (zw[c] + (float)code * sw[c]) * qsk[c];
        }
        p.s_hat[rb + pos] = acc;
      }
      __syncwarp();
    }
  }
#ifdef VKV_TPROF
  __syncthreads();
#endif
  TP_MARK(2);                                            // 2 sketch logits

  // ---- 3 shortlist ---------------------------------------------------------------------------------
  int32_t* bar = p.bar + unit;
  if (p.M_sl > 0) {
    unit_barrier(bar, G * CS);                           // every query head's sketch logits
    TP_MARK(3);                                          // 3a wait for the KV head's rows
    float* best = reinterpret_cast<float*>(J);           // this row's copy of the ranking
    const float* s_unit = p.s_hat + (long long)(unit * G) * n;
    for (int pos = p0 + tid; pos < p1; pos += NT) {
      float v = -INFINITY;
      for (int gg = 0; gg < G; ++gg) v = fmaxf(v, __ldcg(s_unit + (long long)gg * n + pos));
      best[pos] = rS[(long long)pos * H + h] < 1.0f ? v : -INFINITY;
    }
    const int M = p.M_sl;
    topk<CS>(rw, [&](int pos) { return okey((double)best[pos]); }, p0, p1, M, false, ts);
    TP_MARK(4);                                          // 3b shortlist top-M
    unit_barrier(bar, 2 * G * CS);                       // every block read every row
    TP_MARK(3);
    const uint64_t* ckL = rw.at(ts.ck, 0);
    const int32_t* cpL = rw.at(ts.cp, 0);
    const int sub = tid & 3;
    double nhost = 0;
    for (int j0 = cr * (NT / 4); j0 < M; j0 += CS * (NT / 4)) {
      const int j = j0 + (tid >> 2);
      bool ok = false, host = false;
      int pos = 0;
      const T* kr = kv;
      if (j < M) {
        ok = okey_inv(ckL[j]) > -INFINITY;
        pos = cpL[j];
      }
      if (ok) {
        const long long mi = (long long)pos * H + h;
        const int c = cellS[mi];
        host = c < 0;
        kr = host ? pool + (long long)p.cpu_row[slot * max_len * H + mi] * p.PC2
                  : kv + (long long)c * p.C2;
      }
      const float s = dot4<T, D>(kr, qs, sub);
      if (ok && sub == 0) {
        p.s_hat[rb + pos] = s;
        nhost += host;
      }
    }
    nhost = rw.sum(nhost, red);
    if (g == 0 && lead && tid == 0 && nhost > 0) atomicAdd(p.keys_cnt, (int)nhost);
  }
  rw.sync();                                             // s_hat final everywhere
  TP_MARK(5);                                            // 3c shortlist exact logits

  // ---- 4 m, sure ---------------------------------------------------------------------------------
  float m_row;
  double sure;
  {
    float mT = -INFINITY;
    for (int pos = p0 + tid; pos < p1; pos += NT)
      if (rS[(long long)pos * H + h] < 1.0f) mT = fmaxf(mT, __ldcg(p.s_hat + rb + pos));
    m_row = fmaxf((float)mA, (float)rw.max((double)mT, red));
    double su = 0.0;
    for (int j = cr * NT + tid; j < cnt; j += CS * NT) {
      const long long o = (long long)row * WA + j;
      if (p.rA[o] >= 1.0f) su += exp((double)(p.sA[o] - m_row));
    }
    sure = rw.sum(su, red);
  }
  TP_MARK(6);                                            // 4 m, sure

  // ---- 5 wh (cached), jumps, partials --------------------------------------------------------------
  double b0, s_wh;
  int npos, ntail;
  {
    double v[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll 4
    for (int pos = p0 + tid; pos < p1; pos += NT) {
      const float rv = rS[(long long)pos * H + h];
      const float x = __ldcg(p.s_hat + rb + pos) - m_row;
      const double wh = rv < 1.0f ? exp((double)x) : 0.0;
      WH[pos] = wh;
      const Entry e = entry_c(rv, wh);
      J[pos] = e.posm ? e.wh * (1.0 - e.rd) / fmax(e.rd, 1e-300) : -1.0;
      if (e.r0) v[0] += e.wh;
      v[1] += e.wh;
      v[2] += e.posm;
      v[3] += e.tail;
    }
    const int op[4] = {0, 0, 0, 0};
    rw.template allreduce<4>(v, op, red);
    b0 = v[0]; s_wh = v[1]; npos = (int)v[2]; ntail = (int)v[3];
  }
  TP_MARK(7);                                            // 5 wh, jumps

  // ---- 6 truncation walk over the top-M_t jumps -----------------------------------------------------
  bool cap_t;
  {
    // The walk over the top-1024 jumps first (their order is the full order's prefix); the full
    // top-M_t only if it does not stop there.
    const int MT = p.M_t;
    int tsv = MT;
    for (int M = min(MT, 1024);; M = MT) {
    topk<CS>(rw, [&](int pos) { return okey(J[pos]); }, p0, p1, M, true, ts);
    // block rank c walks the sorted indices [c MC, c MC + MC)
    const int MC = (M + CS - 1) / CS, i0 = min(M, cr * MC), i1 = min(M, i0 + MC);
    double* a = f64a;
    double* inc = f64b;
    for (int i = i0 + tid; i < i1; i += NT) {
      const int ji = ts.cp[i];
      const Entry e = entry_c(rS[(long long)ji * H + h], __ldcg(WH + ji));
      a[i - i0] = e.posm ? e.wh * (1.0 - e.rd) : 0.0;
      inc[i - i0] = a[i - i0];
    }
    __syncthreads();
    const double tot = block_scan_f64(inc, i1 - i0, red);
    double bf[1], tt[1];
    const double vv[1] = {tot};
    rw.template exscan<1>(vv, bf, tt);
    int tsl = M;                                         // (M: no stop among the M)
    for (int i = i0 + tid; i < i1; i += NT) {
      const double js = okey_inv(ts.ck[i]);
      const double bb = (b0 + (bf[0] + inc[i - i0])) - a[i - i0];
      if (js <= 2.0 * bb || js < 0.0) tsl = min(tsl, i);
    }
    tsv = (int)rw.min((double)tsl, red);
    if (tsv < M || M == MT) {
      for (int i = i0 + tid; i < min(i1, tsv); i += NT) Tm[ts.cp[i]] = 1;
      break;
    }
    }
    cap_t = (tsv >= MT) && (npos > MT);
  }
  rw.sync();                                             // T* written everywhere
  TP_MARK(8);                                            // 6 truncation walk (top-M_t)

  // ---- 7 totals, dt/dV, the "grad" key -----------------------------------------------------------------
  {
    double v[2] = {0.0, 0.0};
    const int op[2] = {0, 1};
#pragma unroll 4
    for (int pos = p0 + tid; pos < p1; pos += NT) {
      const Moves mv = moves(entry_c(rS[(long long)pos * H + h], WH[pos]), __ldcg(Tm + pos) != 0);
      v[0] += mv.v;
      v[1] = fmax(v[1], fmax(mv.up, mv.dn));
    }
    rw.template allreduce<2>(v, op, red);
    const double V0 = v[0], m0 = fmax(v[1], 0.0);
    const double a = (m0 * p.L) * p.third;
    const double t = a * a + (2.0 * V0) * p.L;
    const double dtdv = (1.0 / fmax(sqrt(t), 1e-300)) * p.L;
#pragma unroll 4
    for (int pos = p0 + tid; pos < p1; pos += NT) {
      const Entry e = entry_c(rS[(long long)pos * H + h], WH[pos]);
      const Moves mv = moves(e, __ldcg(Tm + pos) != 0);
      J[pos] = e.tail ? mv.b + mv.v * dtdv : -1.0;
    }
  }
  TP_MARK(9);                                            // 7 totals, grad key

  // ---- 8 fetch order: top-Kc by the key, ranks -------------------------------------------------------
  const int K = p.Kc;
  topk<CS>(rw, [&](int pos) { return okey(J[pos]); }, p0, p1, K, true, ts);
  {
    const int KC = (K + CS - 1) / CS, k0 = min(K, cr * KC), k1 = min(K, k0 + KC);
    for (int i = k0 + tid; i < k1; i += NT) {
      const int pos = ts.cp[i];
      rank[pos] = i + 1;
      p.o_i[(long long)row * K + i] = pos;
    }
  }
  rw.sync();                                             // ranks written everywhere
  TP_MARK(10);                                           // 8 fetch order top-Kc

  // ---- 9 complement partials; the head's terms (thread t owns k in [t C, t C + C)) ---------------------
  double Rb, RV, Rup, Rdn;
  int sT;
  {
    double v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const int op[5] = {0, 0, 1, 1, 0};
#pragma unroll 4
    for (int pos = p0 + tid; pos < p1; pos += NT) {
      if (__ldcg(rank + pos) != 0) continue;
      const Entry e = entry_c(rS[(long long)pos * H + h], WH[pos]);
      const Moves mv = moves(e, __ldcg(Tm + pos) != 0);
      v[0] += mv.b;
      v[1] += mv.v;
      v[2] = fmax(v[2], mv.up);
      v[3] = fmax(v[3], mv.dn);
      v[4] += mv.T && e.tail;
    }
    rw.template allreduce<5>(v, op, red);
    Rb = v[0]; RV = v[1]; Rup = fmax(v[2], 0.0); Rdn = fmax(v[3], 0.0); sT = (int)v[4];
  }
  const double Bt = sure + s_wh;
  // the head, replicated in every block of the row (each takes a share of the lambdas)
  const int C = (K + NT - 1) / NT;
  const int lo = tid * C, hi = min(K, lo + C);
  // the head's float terms in shared memory (the sort region, free from here; entry j of thread
  // t at j * NT + t: no bank conflicts): c, log r, log(1 - r), r
  float* shc = reinterpret_cast<float*>(f64a);
  float* shlr = shc + KMAX;
  float* shl1r = shc + 2 * KMAX;
  float* shr32 = shc + 3 * KMAX;
  double* sbeta = reinterpret_cast<double*>(ts.ck);      // beta(k), same layout (ck is free too)
#define beta(j) sbeta[(j) * NT + tid]
  double sb = 0.0, sv = 0.0, hup = 0.0, hdn = 0.0;
  int hT = 0;
#pragma unroll
  for (int j = 0; j < HC; ++j) {
    float hc = 0.0f, hlr = 0.0f, hl1r = 0.0f, hr32 = 0.0f;
    double bj = 0.0;
    const int k = lo + j;
    if (j < C && k < hi) {
      const int pos = ts.cp[k];
      const Entry e = entry_c(rS[(long long)pos * H + h], __ldcg(WH + pos));
      const Moves mv = moves(e, __ldcg(Tm + pos) != 0);
      bj = mv.b;
      sb += mv.b;
      sv += mv.v;
      hup = fmax(hup, mv.up);
      hdn = fmax(hdn, mv.dn);
      hT += mv.T && e.tail;
      const bool Tp = mv.T && e.posm, Hp = e.posm && !mv.T;
      hc = (float)(Tp ? e.wh : (Hp ? e.wh / fmax(e.rd, 1e-300) : 0.0));
      if (hc > 0.0f) {                                   // only live entries' terms are used
        hlr = (float)log(fmin(fmax(e.rd, 1e-300), 1.0));
        hl1r = (float)log1p(-fmin(fmax(e.rd, 0.0), 1.0 - 1e-16));
      }
      hr32 = (float)e.rd;
    }
    beta(j) = bj;
    shc[j * NT + tid] = hc;
    shlr[j * NT + tid] = hlr;
    shl1r[j * NT + tid] = hl1r;
    shr32[j * NT + tid] = hr32;
  }
  sb = block_sum<NT, double>(sb, red);
  sv = block_sum<NT, double>(sv, red);
  hup = block_max<NT, double>(hup, 0.0, red);
  hdn = block_max<NT, double>(hdn, 0.0, red);
  hT = block_sum<NT, int>(hT, red);
  {   // beta(k) = sum_{i >= k} b_i + Rb (suffix within the chunk, then the later chunks)
    double run = 0.0;
#pragma unroll
    for (int j = HC - 1; j >= 0; --j) { run += beta(j); beta(j) = run; }
    double after, unused;
    block_rexcl2(run, 0.0, after, unused, red);
#pragma unroll
    for (int j = 0; j < HC; ++j) beta(j) = (after + beta(j)) + Rb;
  }
  const double e0Bt = p.eps0 * Bt;
  const double beta0 = sb + Rb, V0h = sv + RV;
  const double up0 = fmax(hup, Rup), dn0 = fmax(hdn, Rdn);
  const bool screened = (tail1(e0Bt + beta0, V0h, up0) + tail1(e0Bt - beta0, V0h, dn0)) <= p.delta;
  int BK = 2;
  while (BK < K) BK <<= 1;
  long long kstar = 0;
  bool cap = false, nonmono = false;
  TP_MARK(11);                                           // 9 complement partials, head terms
  if (!screened) {
#ifdef VKV_TPROF
    if (tp_on) atomicAdd(p.tprof + TP_UNSCR, 1ull);
#endif
    // The harness's head (kstar_fast with head HH >= K): the curve at k <= K also carries the
    // exact terms of the entries ranked K .. HH - 1 (mid, per lambda), and Bennett's bound is over
    // the entries ranked from HH on (RVh, Muh, Mdh). Those ranked K .. HH - 1 are marked rank -1
    // (top-HH by the key, ties by position as in topk); with HH >= n every non-head entry is mid.
    __shared__ double s_mid[2 * 64], s_midT[2 * 64], s_wp[NW][16];
    const int K2 = max(min(p.HH, n), K);
    const bool all_mid = K2 >= n;
    if (K < K2 && K2 < n)
      mark_top<CS>(rw, [&](int pos) { return okey(J[pos]); }, p0, p1, K2, ts,
                   [&](int pos) { if (rank[pos] == 0) rank[pos] = -1; });
    // One pass: the Bennett inputs from HH on, and each mid entry's float terms cached in place
    // of its key / wh (J, WH: neither is read again by this kernel): (c, r) and (log r,
    // log(1 - r)); c = 0 marks the entries without a term.
    __shared__ float s_lam[64];
    for (int gi = tid; gi < p.NG; gi += NT) s_lam[gi] = (float)(p.grid[gi] / Bt);
    float2* TC = reinterpret_cast<float2*>(J);
    float2* TL = reinterpret_cast<float2*>(WH);
    double RVh = RV, Muh = fmax(Rup, 1e-300), Mdh = fmax(Rdn, 1e-300);
    for (int i = tid; i < 128; i += NT) s_midT[i] = 0.0;
    TP_MARK(12);                                         // 10a mark top-HH
    if (K2 > K) {                // else no mid entries, and Bennett's set is the complement
    {
      double v[3] = {0.0, 0.0, 0.0};
      const int op[3] = {0, 1, 1};
      for (int pos = p0 + tid; pos < p1; pos += NT) {
        const int rk = __ldcg(rank + pos);
        const bool mid = rk == -1 || (rk == 0 && all_mid);
        if (!mid && rk != 0) continue;                   // the top-K: neither
        const Entry e = entry_c(rS[(long long)pos * H + h], WH[pos]);
        const Moves mv = moves(e, __ldcg(Tm + pos) != 0);
        if (!mid) {                                      // from HH on: rank 0, not marked
          v[0] += mv.v;
          v[1] = fmax(v[1], mv.up);
          v[2] = fmax(v[2], mv.dn);
          TC[pos] = make_float2(0.0f, 0.0f);
          continue;
        }
        const bool Tp = mv.T && e.posm, Hp = e.posm && !mv.T;
        const float hc = (float)(Tp ? e.wh : (Hp ? e.wh / fmax(e.rd, 1e-300) : 0.0));
        TC[pos] = make_float2(hc, (float)e.rd);
        if (hc > 0.0f)
          TL[pos] = make_float2((float)log(fmin(fmax(e.rd, 1e-300), 1.0)),
                                (float)log1p(-fmin(fmax(e.rd, 0.0), 1.0 - 1e-16)));
      }
      rw.template allreduce<3>(v, op, red);
      RVh = v[0]; Muh = fmax(v[1], 1e-300); Mdh = fmax(v[2], 1e-300);
    }
    TP_MARK(13);                                         // 10b mid terms, Bennett inputs
    for (int g0 = 0; g0 < p.NG; g0 += 8) {
      double acc[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] = 0.0;
      for (int pos = p0 + tid; pos < p1; pos += NT) {
        const int rk = __ldcg(rank + pos);
        if (!(rk == -1 || (rk == 0 && all_mid))) continue;
        const float2 cr2 = TC[pos];
        const float hc = cr2.x, hr32 = cr2.y;
        if (!(hc > 0.0f)) continue;
        const float2 l2 = TL[pos];
        const float hlr = l2.x, hl1r = l2.y;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (g0 + j < p.NG) {
            const float ac = s_lam[g0 + j] * hc;
            acc[2 * j] += (double)lae32(hlr + ac * (1.0f - hr32), hl1r - ac * hr32);
            acc[2 * j + 1] += (double)lae32(hlr - ac * (1.0f - hr32), hl1r + ac * hr32);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        double x = acc[i];
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
        if (lane == 0) s_wp[w][i] = x;
      }
      __syncthreads();
      if (tid < 16 && g0 + tid / 2 < p.NG) {
        double x = 0.0;
        for (int u = 0; u < NW; ++u) x += s_wp[u][tid];
        s_mid[(tid & 1) * 64 + g0 + tid / 2] = x;
      }
      __syncthreads();
    }
    rw.sync();                                           // every block's partial sums
    for (int i = tid; i < 2 * p.NG; i += NT) {
      const int o = (i / p.NG) * 64 + i % p.NG;
      double x = 0.0;
      for (int r = 0; r < CS; ++r) x += rw.at(s_mid, r)[o];
      s_midT[o] = x;
    }
    TP_MARK(14);                                         // 10c mid sums over the lambda grid
    }
    __syncthreads();
    // min over lambda of the bound's exponent; exp is monotone, so exp(min(min_lam z, 0)) is the
    // Triton kernel's min over lambda of exp(min(z, 0))
    double zu[HC], zd[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) { zu[j] = INFINITY; zd[j] = INFINITY; }
    double zKu = INFINITY, zKd = INFINITY;
    const double uK_up = e0Bt + Rb, uK_dn = e0Bt - Rb;
    for (int gi = cr; gi < p.NG; gi += CS) {
      const double lam = p.grid[gi] / Bt;
      const float a = (float)lam;
      float tu[HC], td[HC];
      double s2[2] = {0.0, 0.0};
#pragma unroll
      for (int j = HC - 1; j >= 0; --j) {
        tu[j] = 0.0f; td[j] = 0.0f;
        const float hc = shc[j * NT + tid];
        if (hc > 0.0f) {
          const float hlr = shlr[j * NT + tid], hl1r = shl1r[j * NT + tid];
          const float hr32 = shr32[j * NT + tid];
          const float ac = a * hc;
          tu[j] = lae32(hlr + ac * (1.0f - hr32), hl1r - ac * hr32);
          td[j] = lae32(hlr - ac * (1.0f - hr32), hl1r + ac * hr32);
        }
        s2[0] += (double)tu[j];
        s2[1] += (double)td[j];
      }
      double af[2];
      block_rexcl<2>(s2, af, red);
      double ex_up = 0.0, ex_dn = 0.0;
      if (p.bennett) { ex_up = bennett(lam, RVh, Muh); ex_dn = bennett(lam, RVh, Mdh); }
      const double mu_ = s_midT[gi], md_ = s_midT[64 + gi];
      double Ku = af[0] + mu_, Kd = af[1] + md_;
#pragma unroll
      for (int j = HC - 1; j >= 0; --j) {
        Ku += (double)tu[j];
        Kd += (double)td[j];
        const double bj = beta(j);
        zu[j] = fmin(zu[j], (Ku + ex_up) - lam * (e0Bt + bj));
        zd[j] = fmin(zd[j], (Kd + ex_dn) - lam * (e0Bt - bj));
      }
      zKu = fmin(zKu, (mu_ + ex_up) - lam * uK_up);
      zKd = fmin(zKd, (md_ + ex_dn) - lam * uK_dn);
    }
    TP_MARK(15);                                         // 10d head curve over the lambda grid
    // the blocks' minima merged: block c finishes k in its share (f64a / f64b overlay the terms)
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HC; ++j)
      if (j < C && lo + j < hi) { f64a[lo + j] = zu[j]; f64b[lo + j] = zd[j]; }
    {
      double v[2] = {zKu, zKd};
      const int op[2] = {2, 2};
      rw.template allreduce<2>(v, op, red);              // (its cluster sync publishes f64a/b)
      zKu = v[0]; zKd = v[1];
    }
    if (CS == 1) __syncthreads();
    int first = BK, last_fail = -1;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      const int k = lo + j;
      if (j < C && k < hi && (k % CS) == cr) {
        double mu = INFINITY, md = INFINITY;
        for (int r = 0; r < CS; ++r) {
          mu = fmin(mu, rw.at(f64a, r)[k]);
          md = fmin(md, rw.at(f64b, r)[k]);
        }
        const double u_dn = e0Bt - beta(j);
        const double t_up = exp(fmin(mu, 0.0));
        const double t_dn = u_dn > 0.0 ? exp(fmin(md, 0.0)) : 1.0;
        if ((t_up + t_dn) <= p.thr) first = min(first, k);
        else last_fail = max(last_fail, k);
      }
    }
    {
      double v[2] = {(double)first, (double)last_fail};
      const int op[2] = {2, 1};
      rw.template allreduce<2>(v, op, red);
      first = (int)v[0];
      last_fail = (int)v[1];
    }
    const double tK_up = exp(fmin(zKu, 0.0));
    const double tK_dn = uK_dn > 0.0 ? exp(fmin(zKd, 0.0)) : 1.0;
    const bool okK = (tK_up + tK_dn) <= p.thr;
    const bool found = first < BK;
    kstar = found ? first : (okK ? K : 0);
    cap = !found && !okK;
    const int first_ok = found ? first : (okK ? K : BK + 1);
    if (!okK) last_fail = K;
    nonmono = last_fail > first_ok;
  }
  TP_MARK(16);                                           // 10e k* search (screened rows: 0)
  kstar = min(kstar, (long long)ntail);
#undef beta
  if (lead) {
    __shared__ double s_beta_at;
    if (tid == 0) s_beta_at = Rb;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HC; ++j)
      if (j < C && lo + j < hi && lo + j == kstar) s_beta_at = sbeta[j * NT + tid];
    __syncthreads();
    if (tid == 0) {
      p.m[row] = m_row;
      p.sure[row] = sure;
      p.kst[row] = kstar;
      p.scr[row] = screened;
      p.beta_k[row] = s_beta_at;
      p.Bt[row] = Bt;
      p.nonmono[row] = nonmono;
      p.tsize[row] = (double)sT + (double)hT;
      const bool f = cap_t || cap;
      p.fb[row] = f;
      if (f) p.fbl[atomicAdd(&p.fbl[p.R], 1)] = row;
    }
  }
  // no block may exit while another still reads its shared memory
  if (CS > 1) rw.cl.sync();
  TP_MARK(17);                                           // outputs, final sync
}

// ---- emit: A's weights, F's cold entries restored (or packed for staging) ----------------------------

struct EParams {
  int H, G, R, n, WA, Kc, max_len, W1, W2, C2;
  long long PC2;
  const int64_t* slots;
  const int32_t* acnt;
  const int32_t* cell;      // [S, max_len, H]
  const int32_t* cpu_row;
  const int32_t* home;
  int32_t* restored;
  const void* pool;
  void* kv;
  const float* rA;
  const int32_t* pA;
  const int64_t* physA;
  const int8_t* Tm;
  const int32_t* rank;
  const int64_t* kst;
  const bool* fb;
  const int32_t* o_i;
  int32_t* scratch;         // [R, n] (the decide kernel's jbuf)
  int64_t* idx;  float* w;  int32_t* slen;   // [R, W1]
  int64_t* idx2; float* w2; int32_t* slen2;  // [R, W2]
  unsigned long long* xfer; // [1] zeroed
  // Compaction (home == nullptr): F is not copied here but listed for the staging fetch: the
  // first ncold[r] entries of posc [R, n] are row r's cold positions in fetch order; idx / w
  // hold A alone (slen = its count).
  int64_t* posc;
  int32_t* ncold;
};

template <typename T>
__global__ void __launch_bounds__(NTE) emit_kernel(EParams p) {
  __shared__ int ired[NTE / 32 + 1];
  __shared__ long long s_src[NTE], s_dst[NTE];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int H = p.H, G = p.G, n = p.n;
  const int b = row / (H * G), h = (row / G) % H;
  const long long slot = p.slots[b];
  const long long rb = (long long)row * n;
  const int cnt = p.acnt[slot * H + h];
  const long long kst = p.kst[row];
  // A with Algorithm 2 (e) weights
  for (int j = tid; j < cnt; j += NTE) {
    const long long o = (long long)row * p.WA + j;
    const int pos = p.pA[o];
    const float rv = p.rA[o];
    const bool inctx = pos < n;
    const bool inT = inctx && p.Tm[rb + pos] != 0;
    const int rk = inctx ? p.rank[rb + pos] : 0;
    const bool inF = rk > 0 && rk <= kst;
    const bool samp = rv < 1.0f;
    p.idx[(long long)row * p.W1 + j] = p.physA[o];
    p.w[(long long)row * p.W1 + j] = (samp && !inT && !inF) ? 1.0f / fmaxf(rv, 1e-30f) : 1.0f;
  }
  // F in fetch order: the head for an ordinary row; the ranks for a capped row (the fallback's)
  const int32_t* list = p.o_i + (long long)row * p.Kc;
  const long long kk = min(kst, (long long)n);
  if (p.fb[row]) {
    int32_t* ord = p.scratch + rb;
    for (int pos = tid; pos < n; pos += NTE) {
      const int rk = p.rank[rb + pos];
      if (rk > 0 && rk <= kst) ord[rk - 1] = pos;
    }
    __syncthreads();
    list = ord;
  }
  const int32_t* cellS = p.cell + slot * p.max_len * H;
  if (p.home == nullptr) {
    // Compaction: the cold part of F packed at the front of posc's row, in fetch order, and
    // its count (the staging fetch and placement read only those).
    long long nc = 0;
    for (long long a0 = 0; a0 < kk; a0 += NTE) {
      const long long i = a0 + tid;
      bool c = false;
      int pos = 0;
      if (i < kk) {
        pos = list[i];
        c = cellS[(long long)pos * H + h] < 0;
      }
      int tot;
      const int e = block_excl_scan<NTE>((int)c, ired, tot);
      if (c) p.posc[rb + nc + e] = pos;
      nc += tot;
    }
    if (tid == 0) {
      p.slen[row] = cnt;
      p.ncold[row] = (int32_t)nc;
      if (nc) atomicAdd(p.xfer, (unsigned long long)nc);
    }
    return;
  }
  const T* pool = reinterpret_cast<const T*>(p.pool);
  T* kv = reinterpret_cast<T*>(p.kv);
  long long off = cnt;
  unsigned long long ncold = 0;
  for (long long a0 = 0; a0 < kk; a0 += NTE) {
    const long long i = a0 + tid;
    bool cold = false;
    long long hcell = 0, src = -1;
    if (i < kk) {
      const int pos = list[i];
      const long long mi = (long long)pos * H + h;
      cold = cellS[mi] < 0;
      if (cold) {
        const long long gi = slot * p.max_len * H + mi;
        hcell = p.home[gi];
        if (atomicExch(&p.restored[gi], 1) == 0) src = p.cpu_row[gi];
      }
    }
    int tot;
    const int e = block_excl_scan<NTE>((int)cold, ired, tot);
    if (cold) {
      const long long dst = off + e;
      if (dst < p.W1) {
        p.idx[(long long)row * p.W1 + dst] = hcell;
        p.w[(long long)row * p.W1 + dst] = 1.0f;
      } else {
        p.idx2[(long long)row * p.W2 + dst - p.W1] = hcell;
        p.w2[(long long)row * p.W2 + dst - p.W1] = 1.0f;
      }
    }
    // this chunk's copies, all threads together (16-byte pieces)
    s_src[tid] = src;
    s_dst[tid] = hcell;
    __syncthreads();
    constexpr int PIECE = 16 / sizeof(T);
    const int per = p.C2 / PIECE;
    for (int t = tid; t < NTE * per; t += NTE) {
      const int which = t / per, piece = t % per;
      const long long s0 = s_src[which];
      if (s0 >= 0) {
        const uint4* sp = reinterpret_cast<const uint4*>(pool + s0 * p.PC2) + piece;
        uint4* dp = reinterpret_cast<uint4*>(kv + s_dst[which] * p.C2) + piece;
        *dp = *sp;
      }
    }
    __syncthreads();
    off += tot;
    ncold += tot;
  }
  if (tid == 0) {
    p.slen[row] = (int32_t)min(off, (long long)p.W1);
    p.slen2[row] = (int32_t)max(off - (long long)p.W1, 0LL);
    if (ncold) atomicAdd(p.xfer, ncold);
  }
}

}  // namespace vkv

// ---- host ----------------------------------------------------------------------------------------------

using vkv::Params;
using vkv::EParams;

template <typename T, int D, int CS>
static void launch_cs(const Params& p, int rows, cudaStream_t st) {
  auto k = vkv::decide_kernel<T, D, CS>;
  const size_t sm = vkv::Layout<CS>::BYTES;
  // The attribute once per instantiation (never inside a graph capture after the first call).
  static bool attr = (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)sm), true);
  (void)attr;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(rows * CS);
  cfg.blockDim = dim3(vkv::NT);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CS;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, p));
}

// Rows (clusters) of size CS that can be resident at once (clusters stay within a GPC).
template <typename T, int D, int CS>
static int active_clusters() {
  static int n = -1;
  if (n < 0) {
    auto k = vkv::decide_kernel<T, D, CS>;
    const size_t sm = vkv::Layout<CS>::BYTES;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS * 64);
    cfg.blockDim = dim3(vkv::NT);
    cfg.dynamicSmemBytes = sm;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CS;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    int c = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&c, k, &cfg));
    n = c;
  }
  return n;
}

// The largest cluster size <= cs_max with every row resident (the rows of a KV head wait on each
// other at the shortlist), else 0.
template <typename T, int D>
static int pick_cs(int R, int cs_max) {
  if (cs_max >= 4 && active_clusters<T, D, 4>() >= R) return 4;
  if (cs_max >= 3 && active_clusters<T, D, 3>() >= R) return 3;
  if (cs_max >= 2 && active_clusters<T, D, 2>() >= R) return 2;
  if (active_clusters<T, D, 1>() >= R) return 1;
  return 0;
}

template <typename T, int D>
static void launch_d(const Params& p, int cs_max, cudaStream_t st) {
  const int cs = pick_cs<T, D>(p.R, cs_max);
  if (cs == 4) launch_cs<T, D, 4>(p, p.R, st);
  else if (cs == 3) launch_cs<T, D, 3>(p, p.R, st);
  else if (cs == 2) launch_cs<T, D, 2>(p, p.R, st);
  else if (cs == 1) launch_cs<T, D, 1>(p, p.R, st);
  else {
    // More rows than can be resident: launch whole KV heads (the rows that wait on each other)
    // in chunks that are, one after another on the stream.
    const int chunk = (active_clusters<T, D, 1>() / p.G) * p.G;
    TORCH_CHECK(chunk > 0, "vkv cuda decide: a KV head's ", p.G, " rows cannot be resident");
    Params c = p;
    for (c.row0 = 0; c.row0 < p.R; c.row0 += chunk)
      launch_cs<T, D, 1>(c, std::min(chunk, p.R - c.row0), st);
  }
}

template <typename T>
static void launch_decide(const Params& p, int D, int cs, cudaStream_t st) {
  if (D == 128) launch_d<T, 128>(p, cs, st);
  else if (D == 64) launch_d<T, 64>(p, cs, st);
  else TORCH_CHECK(false, "vkv cuda decide: head_dim ", D, " unsupported (64, 128)");
}

#define PTR(t, ty) reinterpret_cast<ty*>((t).data_ptr())

// VKV_TPROF builds: the phase counters every later decide launch adds to (vkv::TP_N uint64).
static unsigned long long* g_tprof = nullptr;

void set_tprof(torch::Tensor t) {
  TORCH_CHECK(t.numel() >= vkv::TP_N && t.dtype() == torch::kInt64 && t.is_cuda(),
              "vkv set_tprof: int64 CUDA tensor of ", vkv::TP_N);
  g_tprof = reinterpret_cast<unsigned long long*>(t.data_ptr());
}

void decide(torch::Tensor q, double sm_scale, torch::Tensor slots, torch::Tensor r,
            torch::Tensor cell, torch::Tensor cpu_row, torch::Tensor alist, torch::Tensor acnt,
            torch::Tensor kv, torch::Tensor pool, torch::Tensor codes, torch::Tensor zero,
            torch::Tensor scale, torch::Tensor channels, torch::Tensor grid,
            std::vector<int64_t> ints, std::vector<double> dbls,
            torch::Tensor s_hat, torch::Tensor jbuf, torch::Tensor wh, torch::Tensor Tm,
            torch::Tensor rank,
            torch::Tensor sA, torch::Tensor rA, torch::Tensor pA, torch::Tensor physA,
            torch::Tensor o_i, torch::Tensor m, torch::Tensor sure, torch::Tensor kst,
            torch::Tensor beta_k, torch::Tensor Bt, torch::Tensor nonmono, torch::Tensor tsize,
            torch::Tensor scr, torch::Tensor fb, torch::Tensor fbl, torch::Tensor keys_cnt,
            torch::Tensor bar) {
  // ints: B, H, G, n, WA, max_len, M_sl, M_t, Kc, bennett, NCH, PER, BITS, GROUP, CS, HH
  Params p{};
  p.q = q.data_ptr();
  p.sm_scale = (float)sm_scale;
  p.B = ints[0]; p.H = ints[1]; p.G = ints[2]; p.Hq = p.H * p.G; p.R = p.B * p.Hq;
  p.n = ints[3]; p.WA = ints[4]; p.max_len = ints[5]; p.M_sl = ints[6]; p.M_t = ints[7];
  p.Kc = ints[8]; p.bennett = ints[9]; p.NCH = ints[10]; p.PER = ints[11]; p.BITS = ints[12];
  p.GROUP = ints[13];
  p.HH = (int)ints[15];
  const int cs = (int)ints[14];
  TORCH_CHECK(cs >= 1 && cs <= 4, "vkv cuda decide: cluster size 1 .. 4");
  p.NB = p.NCH / p.PER;
  p.NG = (int)grid.numel();
  TORCH_CHECK(p.M_t <= vkv::KMAX && p.Kc <= vkv::KMAX && p.M_sl <= vkv::KMAX,
              "vkv cuda decide: heads larger than ", vkv::KMAX);
  TORCH_CHECK(p.NCH <= 64 && q.size(-1) <= 128, "vkv cuda decide: sketch/head too wide");
  p.eps0 = dbls[0]; p.delta = dbls[1]; p.thr = dbls[2]; p.L = dbls[3]; p.third = dbls[4];
  p.slots = PTR(slots, const int64_t);
  p.r = PTR(r, const float);
  p.cell = PTR(cell, const int32_t);
  p.cpu_row = PTR(cpu_row, const int32_t);
  p.alist = PTR(alist, const int32_t);
  p.acnt = PTR(acnt, const int32_t);
  p.kv = kv.data_ptr();
  p.pool = pool.data_ptr();
  p.C2 = (int)kv.stride(0);
  p.PC2 = pool.stride(0);
  p.codes = PTR(codes, const uint8_t);
  p.c_s = codes.stride(0); p.c_h = codes.stride(1);
  p.zero = PTR(zero, const __half);
  p.scale = PTR(scale, const __half);
  p.z_s = zero.stride(0); p.z_h = zero.stride(1);
  p.channels = PTR(channels, const int64_t);
  p.ch_s = channels.stride(0); p.ch_h = channels.stride(1);
  p.grid = PTR(grid, const double);
  p.s_hat = PTR(s_hat, float);
  p.jbuf = PTR(jbuf, double);
  p.wh = PTR(wh, double);
  p.Tm = PTR(Tm, int8_t);
  p.rank = PTR(rank, int32_t);
  p.sA = PTR(sA, float); p.rA = PTR(rA, float); p.pA = PTR(pA, int32_t);
  p.physA = PTR(physA, int64_t);
  p.o_i = PTR(o_i, int32_t);
  p.m = PTR(m, float); p.sure = PTR(sure, double); p.kst = PTR(kst, int64_t);
  p.beta_k = PTR(beta_k, double); p.Bt = PTR(Bt, double); p.nonmono = PTR(nonmono, int8_t);
  p.tsize = PTR(tsize, double); p.scr = PTR(scr, int8_t); p.fb = PTR(fb, bool);
  p.fbl = PTR(fbl, int32_t); p.keys_cnt = PTR(keys_cnt, int32_t); p.bar = PTR(bar, int32_t);
  p.tprof = g_tprof;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (kv.scalar_type() == at::kBFloat16) launch_decide<__nv_bfloat16>(p, (int)q.size(-1), cs, st);
  else if (kv.scalar_type() == at::kHalf) launch_decide<__half>(p, (int)q.size(-1), cs, st);
  else TORCH_CHECK(false, "vkv cuda decide: KV dtype must be bf16 or fp16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void emit(torch::Tensor slots, torch::Tensor acnt, torch::Tensor cell, torch::Tensor cpu_row,
          torch::Tensor home, torch::Tensor restored, torch::Tensor pool, torch::Tensor kv,
          torch::Tensor rA, torch::Tensor pA, torch::Tensor physA, torch::Tensor Tm,
          torch::Tensor rank, torch::Tensor kst, torch::Tensor fb, torch::Tensor o_i,
          torch::Tensor scratch, torch::Tensor idx, torch::Tensor w, torch::Tensor slen,
          torch::Tensor idx2, torch::Tensor w2, torch::Tensor slen2, torch::Tensor xfer,
          std::vector<int64_t> ints, c10::optional<torch::Tensor> posc,
          c10::optional<torch::Tensor> ncold) {
  // ints: H, G, n, WA, Kc, max_len
  EParams p{};
  p.H = ints[0]; p.G = ints[1]; p.n = ints[2]; p.WA = ints[3]; p.Kc = ints[4];
  p.max_len = ints[5];
  p.R = (int)kst.numel();
  p.W1 = (int)idx.size(-1); p.W2 = (int)idx2.size(-1);
  p.C2 = (int)kv.stride(0); p.PC2 = pool.stride(0);
  TORCH_CHECK((p.C2 * kv.element_size()) % 16 == 0 && (p.PC2 * pool.element_size()) % 16 == 0,
              "vkv cuda emit: rows must be 16-byte multiples");
  p.slots = PTR(slots, const int64_t); p.acnt = PTR(acnt, const int32_t);
  p.cell = PTR(cell, const int32_t); p.cpu_row = PTR(cpu_row, const int32_t);
  if (posc.has_value()) {   // compaction: home / restored unused
    TORCH_CHECK(ncold.has_value(), "vkv cuda emit: posc needs ncold");
    p.posc = PTR((*posc), int64_t);
    p.ncold = PTR((*ncold), int32_t);
  } else {
    p.home = PTR(home, const int32_t); p.restored = PTR(restored, int32_t);
  }
  p.pool = pool.data_ptr(); p.kv = kv.data_ptr();
  p.rA = PTR(rA, const float); p.pA = PTR(pA, const int32_t);
  p.physA = PTR(physA, const int64_t);
  p.Tm = PTR(Tm, const int8_t); p.rank = PTR(rank, const int32_t);
  p.kst = PTR(kst, const int64_t); p.fb = PTR(fb, const bool); p.o_i = PTR(o_i, const int32_t);
  p.scratch = PTR(scratch, int32_t);
  p.idx = PTR(idx, int64_t); p.w = PTR(w, float); p.slen = PTR(slen, int32_t);
  p.idx2 = PTR(idx2, int64_t); p.w2 = PTR(w2, float); p.slen2 = PTR(slen2, int32_t);
  p.xfer = PTR(xfer, unsigned long long);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (kv.scalar_type() == at::kBFloat16)
    vkv::emit_kernel<__nv_bfloat16><<<p.R, vkv::NTE, 0, st>>>(p);
  else if (kv.scalar_type() == at::kHalf)
    vkv::emit_kernel<__half><<<p.R, vkv::NTE, 0, st>>>(p);
  else TORCH_CHECK(false, "vkv cuda emit: KV dtype must be bf16 or fp16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_tprof", &set_tprof, "VKV_TPROF builds: the decide phase counters",
        py::arg("t"));
  m.def("decide", &decide, "vkv decode decision (one layer)", py::arg("q"), py::arg("sm_scale"),
        py::arg("slots"), py::arg("r"), py::arg("cell"), py::arg("cpu_row"), py::arg("alist"),
        py::arg("acnt"), py::arg("kv"), py::arg("pool"), py::arg("codes"), py::arg("zero"),
        py::arg("scale"), py::arg("channels"), py::arg("grid"), py::arg("ints"), py::arg("dbls"),
        py::arg("s_hat"), py::arg("jbuf"), py::arg("wh"), py::arg("Tm"), py::arg("rank"), py::arg("sA"),
        py::arg("rA"), py::arg("pA"), py::arg("physA"), py::arg("o_i"), py::arg("m"),
        py::arg("sure"), py::arg("kst"), py::arg("beta_k"), py::arg("Bt"), py::arg("nonmono"),
        py::arg("tsize"), py::arg("scr"), py::arg("fb"), py::arg("fbl"), py::arg("keys_cnt"),
        py::arg("bar"));
  m.def("emit", &emit, "vkv attention list: A weights + F restored in place (or listed for "
        "the staging fetch)", py::arg("slots"), py::arg("acnt"), py::arg("cell"),
        py::arg("cpu_row"), py::arg("home"), py::arg("restored"), py::arg("pool"), py::arg("kv"),
        py::arg("rA"), py::arg("pA"), py::arg("physA"), py::arg("Tm"), py::arg("rank"),
        py::arg("kst"), py::arg("fb"), py::arg("o_i"), py::arg("scratch"), py::arg("idx"),
        py::arg("w"), py::arg("slen"), py::arg("idx2"), py::arg("w2"), py::arg("slen2"),
        py::arg("xfer"), py::arg("ints"), py::arg("posc") = py::none(),
        py::arg("ncold") = py::none());
}
