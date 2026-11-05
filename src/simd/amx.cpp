
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Intel AMX (Advanced Matrix Extensions) SIMD implementations.
//
// This translation unit is compiled with -mamx-tile -mamx-int8 -mamx-bf16
// plus AVX-512F/BW/DQ/VL/VNNI/BF16 so AMX kernels can use AVX-512 to
// preprocess inputs (pack/quantize/FP32->BF16 convert) and to handle tail
// elements that don't fill an AMX tile.

#include "amx_bf16_matmul.h"
#include "simd_status.h"
#include "sq8_uniform_simd.h"

#if defined(ENABLE_AMX)
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>
#endif

namespace vsag::amx {

// Single-pair scalar IP. AMX is wasted on a single (query, code) pair
// because tile-config + tile-load already costs more than AVX-512
// finishes one IP in. Fall back to AVX-512 here; AMX value is delivered
// through SQ8UniformComputeCodesIPBatch.
float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AMX)
    return avx512::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#else
    return generic::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

#if defined(ENABLE_AMX)

// AMX tile-config used by the batch IP kernel.
//
//   palette   = 1
//   start_row = 0
//
//   tile 0 (A): 16 rows x 64 bytes  -- 16 codes loaded row-major
//                                     (row stride = code_stride)
//   tile 1 (B): 16 rows x 64 bytes  -- query, broadcast as 16 identical
//                                     columns in VNNI form
//   tile 2 (C): 16 rows x 64 bytes  -- 16x16 INT32 accumulator (16*4 B/row)
//
// _tile_dpbuud(C, A, B) [u8 x u8 -> i32]:
//   C[m][n] += sum_{k=0..K-1} A[m][k] * B[k][n]
// where physically:
//   A   : M rows of K bytes (K up to 64)
//   B   : (K/4) rows of N dwords; each dword packs 4 consecutive K
//         B_phys[r][n*4 + i] == B_logical[r*4 + i][n], i in 0..3
//   C   : M rows of N dwords (INT32)
struct alignas(64) AmxTileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};
static_assert(sizeof(AmxTileConfig) == 64, "AmxTileConfig must be exactly 64 bytes for LDTILECFG");

// Tile assignment for the SQ8-uniform batch IP kernel:
//   tile 0 (A) : 16 codes laid out row-major in the codes array, loaded
//                directly via _tile_loadd with row stride = dim.  No
//                packing is needed because A is u8 row-major in AMX.
//   tile 1 (B) : a "broadcast query" VNNI tile (see PackQueryAsBTile).
//   tile 2 (C) : 16x16 INT32 accumulator.  Because B's columns are all
//                identical (every column carries the query), C[m][n] is
//                identical across n and equals <query, code_m>.
static const AmxTileConfig kBatchIpTileConfig = {
    /*palette_id=*/1,
    /*start_row=*/0,
    /*reserved=*/{0},
    /*colsb=*/
    {
        64,
        64,
        64,
        0,
        0,
        0,
        0,
        0,  //
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    /*rows=*/
    {
        16,
        16,
        16,
        0,
        0,
        0,
        0,
        0,  //
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
};

// Pack the query into a VNNI-laid B tile such that *every* output
// column n yields the dot product <query, code_m> for the m-th row
// of the A tile. Since codes go into A (row m = code m of the block,
// with row-stride = dim, see TileDot*Block below), we want
//   B_logical[k][n] = query[k]  for all n in 0..15.
// VNNI requires B_phys[r][n*4 + i] = B_logical[r*4 + i][n], which
// here is just query[r*4 + i] -- independent of n. So row r is the
// 4-byte chunk query[4r..4r+3] replicated 16 times = one _mm512
// register built with _mm512_set1_epi32(*(uint32_t*)&query[4r]).
//
// Caller passes the 64-byte query window for the current K-block.
static inline void
PackQueryAsBTile_AVX512(const uint8_t* query_block, uint8_t b_packed[16 * 64]) {
    // query_block is not guaranteed to be 4-byte aligned, so we use
    // std::memcpy to load each 4-byte chunk without invoking undefined
    // behavior from an unaligned reinterpret_cast.
    auto load_dword = [](const uint8_t* p) {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int>(v);
    };
    _mm512_storeu_si512(b_packed + 0 * 64, _mm512_set1_epi32(load_dword(query_block + 0 * 4)));
    _mm512_storeu_si512(b_packed + 1 * 64, _mm512_set1_epi32(load_dword(query_block + 1 * 4)));
    _mm512_storeu_si512(b_packed + 2 * 64, _mm512_set1_epi32(load_dword(query_block + 2 * 4)));
    _mm512_storeu_si512(b_packed + 3 * 64, _mm512_set1_epi32(load_dword(query_block + 3 * 4)));
    _mm512_storeu_si512(b_packed + 4 * 64, _mm512_set1_epi32(load_dword(query_block + 4 * 4)));
    _mm512_storeu_si512(b_packed + 5 * 64, _mm512_set1_epi32(load_dword(query_block + 5 * 4)));
    _mm512_storeu_si512(b_packed + 6 * 64, _mm512_set1_epi32(load_dword(query_block + 6 * 4)));
    _mm512_storeu_si512(b_packed + 7 * 64, _mm512_set1_epi32(load_dword(query_block + 7 * 4)));
    _mm512_storeu_si512(b_packed + 8 * 64, _mm512_set1_epi32(load_dword(query_block + 8 * 4)));
    _mm512_storeu_si512(b_packed + 9 * 64, _mm512_set1_epi32(load_dword(query_block + 9 * 4)));
    _mm512_storeu_si512(b_packed + 10 * 64, _mm512_set1_epi32(load_dword(query_block + 10 * 4)));
    _mm512_storeu_si512(b_packed + 11 * 64, _mm512_set1_epi32(load_dword(query_block + 11 * 4)));
    _mm512_storeu_si512(b_packed + 12 * 64, _mm512_set1_epi32(load_dword(query_block + 12 * 4)));
    _mm512_storeu_si512(b_packed + 13 * 64, _mm512_set1_epi32(load_dword(query_block + 13 * 4)));
    _mm512_storeu_si512(b_packed + 14 * 64, _mm512_set1_epi32(load_dword(query_block + 14 * 4)));
    _mm512_storeu_si512(b_packed + 15 * 64, _mm512_set1_epi32(load_dword(query_block + 15 * 4)));
}

// Run one 16-codes block against the query.  dim_aligned must be a
// multiple of 64; the caller handles the dim-tail with AVX-512.
//
// Codes go into A (no packing!): A row m = codes_block_base + m*code_stride,
// loaded with row stride = code_stride. Query goes into B via
// PackQueryAsBTile. The result C[m][n] is the same for all n (since B is
// column-broadcast), and equals <code_m, query>; we read column 0
// (i.e. row 0 of int32 dwords from each row, picking dword 0).
static inline void
TileDotOne16Block(const uint8_t* query,
                  const uint8_t* codes_block_base,  // pointer to first of 16 codes
                  uint64_t code_stride,
                  uint64_t dim_aligned,
                  int32_t out_int32[16]) {
    alignas(64) int32_t c_full[16 * 16];
    alignas(64) uint8_t b_tile[16 * 64];

    _tile_zero(2);

    for (uint64_t k0 = 0; k0 < dim_aligned; k0 += 64) {
        PackQueryAsBTile_AVX512(query + k0, b_tile);
        // A is loaded directly from the codes array. Row stride = code_stride.
        // 16 rows of 64 bytes each -- starting at codes_block_base + k0.
        // _tile_loadd takes a signed 32-bit stride; guard against an
        // implausibly large code_stride that would overflow the cast.
        assert(code_stride <= static_cast<uint64_t>(INT32_MAX));
        _tile_loadd(0, codes_block_base + k0, static_cast<int>(code_stride));
        _tile_loadd(1, b_tile, 64);
        _tile_dpbuud(2, 0, 1);
    }

    _tile_stored(2, c_full, 16 * sizeof(int32_t));

    // C[m][n] is the same across all n. Take column 0.
    for (int m = 0; m < 16; ++m) {
        out_int32[m] = c_full[m * 16];
    }
}

// Hot path: process 64 codes per outer step. Currently implemented as
// four sequential calls to TileDotOne16Block, which is enough for AMX
// to beat AVX-512 on this workload because the dominant cost was
// previously the VNNI repacking of the codes (now removed by mapping
// codes directly into the A tile via row-stride loads).
//
// A multi-accumulator version (4 C tiles 2,3,4,5 sharing one B-tile
// load per K-iteration) was attempted to amortize the query packing
// across 4 sub-blocks. It produced wrong results in some configurations
// (likely a tile-dependency hazard issue worth a closer look), and the
// expected speedup is small because PackQueryAsBTile_AVX512 is already
// ~16 set1+stores -- not the bottleneck. Skipped for the PoC.
static inline void
TileDotOne64Block(const uint8_t* query,
                  const uint8_t* codes_block_base,
                  uint64_t code_stride,
                  uint64_t dim_aligned,
                  int32_t out_int32[64]) {
    TileDotOne16Block(
        query, codes_block_base + 0 * 16 * code_stride, code_stride, dim_aligned, out_int32 + 0);
    TileDotOne16Block(
        query, codes_block_base + 1 * 16 * code_stride, code_stride, dim_aligned, out_int32 + 16);
    TileDotOne16Block(
        query, codes_block_base + 2 * 16 * code_stride, code_stride, dim_aligned, out_int32 + 32);
    TileDotOne16Block(
        query, codes_block_base + 3 * 16 * code_stride, code_stride, dim_aligned, out_int32 + 48);
}

#endif  // ENABLE_AMX

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
#if defined(ENABLE_AMX)
    if (n_codes == 0) {
        return;
    }
    if (dim == 0) {
        // Match other backends: empty-vector inner product is 0 for each code.
        std::fill_n(out, n_codes, 0.0F);
        return;
    }

    // Per-call AMX overhead (tile_loadconfig + tile_release plus the
    // 16-broadcast B-tile setup) does not amortize for tiny batches.
    // Profiling sift1m / HGraph+IVF SQ8U showed ~99% of calls land at
    // n_codes <= 32 (IVF scan block) and never above 32. Below this
    // threshold AVX-512 wins; switch to it instead of paying AMX
    // setup cost.
    constexpr uint64_t kAmxMinCount = 48;
    // dim split into (dim_aligned = floor(dim/64)*64) handled by AMX,
    // tail handled by AVX-512 per-pair on the residue. When dim < 64,
    // dim_aligned == 0 so the AMX inner loop does no tdpbuud work and
    // every output is computed by the AVX-512 tail anyway; in that case
    // skip AMX entirely to avoid paying tile_loadconfig / tile_release
    // and the per-iteration tile_zero / tile_stored overhead.
    uint64_t dim_aligned = dim & ~uint64_t{63};
    uint64_t tail = dim - dim_aligned;
    // _tile_loadd takes a signed 32-bit stride. In practice code_stride is
    // O(dim) bytes and never anywhere near 2 GiB, but fall back to the
    // AVX-512 scalar loop in release builds rather than truncating the cast.
    if (n_codes < kAmxMinCount || dim_aligned == 0 ||
        code_stride > static_cast<uint64_t>(INT32_MAX)) {
        for (uint64_t j = 0; j < n_codes; ++j) {
            out[j] = avx512::SQ8UniformComputeCodesIP(query, codes + j * code_stride, dim);
        }
        return;
    }

    _tile_loadconfig(&kBatchIpTileConfig);

    uint64_t i = 0;
    // Hot path: 64 codes per outer step (4 C-tile accumulators).
    for (; i + 64 <= n_codes; i += 64) {
        const uint8_t* base = codes + i * code_stride;
        int32_t partial[64];
        TileDotOne64Block(query, base, code_stride, dim_aligned, partial);

        if (tail) {
            for (int c = 0; c < 64; ++c) {
                int32_t t = static_cast<int32_t>(avx512::SQ8UniformComputeCodesIP(
                    query + dim_aligned, base + c * code_stride + dim_aligned, tail));
                partial[c] += t;
            }
        }
        for (int c = 0; c < 64; ++c) {
            out[i + c] = static_cast<float>(partial[c]);
        }
    }
    // Mid path: 16 codes per outer step for the n_codes-tail of size
    // 16..63. Uses one C-tile accumulator (tile 2).
    for (; i + 16 <= n_codes; i += 16) {
        const uint8_t* base = codes + i * code_stride;
        int32_t partial[16];
        TileDotOne16Block(query, base, code_stride, dim_aligned, partial);

        if (tail) {
            for (int c = 0; c < 16; ++c) {
                int32_t t = static_cast<int32_t>(avx512::SQ8UniformComputeCodesIP(
                    query + dim_aligned, base + c * code_stride + dim_aligned, tail));
                partial[c] += t;
            }
        }
        for (int c = 0; c < 16; ++c) {
            out[i + c] = static_cast<float>(partial[c]);
        }
    }
    _tile_release();

    // Trailing codes (n_codes % 16) -- AMX overhead doesn't pay off
    // for fewer than 16 codes; just call the scalar IP per pair.
    for (; i < n_codes; ++i) {
        out[i] = avx512::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
#else
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = generic::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
#endif
}

// =============================================================================
// AMX BF16 GEMM (Intel Sapphire Rapids / Granite Rapids AMX_BF16)
// =============================================================================
//
// Targets the KMeans centroid-assignment GEMM in
// `KMeansCluster::find_nearest_one_with_blas`, which currently calls cblas
// SGEMM with ColMajor / Trans / NoTrans / alpha=-2 / beta=0 to form
//   distances[k x N] = -2 * centroids^T * queries
// where `centroids` is logically [k x dim] row-major and `queries` is
// [N x dim] row-major. We replace it with a BF16 inner-product kernel that
// produces the same column-major output.
//
// Tile shape: M_T=16, N_T=16, K_T=32 BF16 elements.  Each `_tile_dpbf16ps`
// does 16*16*32 = 8192 BF16-FMAs (16384 flops) on a single tile mul.
//
// Layout choices:
//  - A (centroids) is pre-packed into row-major blocks of 16 rows x K_pad
//    BF16. The 16x32 tile at (m_blk, k_blk) lives at offset
//    (m_blk * K_pad + k_blk * 16) * 2 bytes, stride 64 bytes per row.
//  - B (queries) is pre-packed into VNNI form per N-block of 16 rows:
//    for K-block k_blk, the 16x32-BF16 tile in VNNI layout (16 rows of 32
//    BF16 = 64 bytes per row).  Offset (n_blk * K_pad + k_blk * 16) * 2.
//  - C is column-major in the caller's buffer; we stage one M_T x N_T FP32
//    tile (16x16) at a time, scale by alpha, and scatter into the output.
//
// Tail handling:
//  - K is padded with zeros up to K_pad = round_up(K, 32). Zero BF16 lanes
//    contribute zero to the dot product so the result is identical to a
//    K-exact kernel up to BF16 rounding error.
//  - M and N tails (< 16) are computed by a scalar FP32 fallback (rare and
//    bounded by 15 rows/cols per dimension).

#if defined(ENABLE_AMX)
namespace {

// AMX BF16 tile config: tile 0 = A (M_T=16, 64 B/row = 32 BF16),
// tile 1 = B in VNNI (rows=K_T/2=16, colsb=N_T*2*2=64 B/row),
// wait -- BF16 VNNI packs 2 K per dword so B rows = K_T/2 = 16, and each
// row has N_T dwords = 16 * 4 bytes = 64 bytes. tile 2 = C (M_T=16,
// N_T*4=64 B/row).
struct alignas(64) AmxBF16TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};
static_assert(sizeof(AmxBF16TileConfig) == 64,
              "AmxBF16TileConfig must be exactly 64 bytes for LDTILECFG");

constexpr AmxBF16TileConfig kBF16GemmTileConfig = {
    /*palette_id=*/1,
    /*start_row=*/0,
    /*reserved=*/{0},
    /*colsb=*/
    {
        64,
        64,
        64,
        0,
        0,
        0,
        0,
        0,  //
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    /*rows=*/
    {
        16,  // A : 16 BF16 rows, 32 BF16 elems each
        16,  // B : 16 VNNI rows (= K_T/2), each row = 16 dwords = 64 B
        16,  // C : 16 rows of 16 FP32 dwords = 64 B
        0,
        0,
        0,
        0,
        0,  //
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
};

constexpr int64_t kMT = 16;
constexpr int64_t kNT = 16;
constexpr int64_t kKT = 32;

// Convert 32 floats -> 32 BF16 with round-to-nearest-even using
// AVX-512-BF16's _mm512_cvtne2ps_pbh; falls back to a manual RNE conversion
// if the lo/hi inputs aren't both available (we always pass both here).
static inline __m512i
CvtFp32x32ToBF16(const float* p) {
    __m512 lo = _mm512_loadu_ps(p);
    __m512 hi = _mm512_loadu_ps(p + 16);
    return reinterpret_cast<__m512i>(_mm512_cvtne2ps_pbh(hi, lo));
}

// Load up to 32 floats, padding with zeros if `count` < 32, then convert
// to 32 BF16. Used for the K-tail of a K-block.
static inline __m512i
CvtFp32xUpTo32ToBF16(const float* p, int64_t count) {
    alignas(64) float buf[32] = {0};
    for (int64_t i = 0; i < count; ++i) {
        buf[i] = p[i];
    }
    __m512 lo = _mm512_load_ps(buf);
    __m512 hi = _mm512_load_ps(buf + 16);
    return reinterpret_cast<__m512i>(_mm512_cvtne2ps_pbh(hi, lo));
}

// Pack the A matrix (row-major M x K floats) into AMX-A tile layout:
//   For each m_blk in [0, M_blocks) and k_blk in [0, K_blocks):
//     A_packed[m_blk][k_blk] is a 16x32 BF16 tile, stored row-major,
//     64 bytes per row.
//   Total bytes per m_blk = K_blocks * 16 * 64 = K_blocks * 1024.
// `m_rows` is the actual number of rows in this m_blk (1..16); rows beyond
// `m_rows` are zero-filled so the dot product over the missing rows yields
// zero (we just ignore those output entries).
static void
PackARowMajorToBF16Tiles(
    const float* a, int64_t m, int64_t k, int64_t k_padded, uint16_t* out_bf16) {
    const int64_t k_blocks = k_padded / kKT;
    for (int64_t m_blk = 0; m_blk < (m + kMT - 1) / kMT; ++m_blk) {
        const int64_t m_base = m_blk * kMT;
        const int64_t m_rows = std::min<int64_t>(kMT, m - m_base);
        uint16_t* blk_dst = out_bf16 + m_blk * k_blocks * kMT * kKT;
        for (int64_t k_blk = 0; k_blk < k_blocks; ++k_blk) {
            const int64_t k_base = k_blk * kKT;
            const int64_t k_count = std::min<int64_t>(kKT, k - k_base);
            uint16_t* tile_dst = blk_dst + k_blk * kMT * kKT;
            for (int64_t r = 0; r < m_rows; ++r) {
                const float* row_src = a + (m_base + r) * k + k_base;
                __m512i bf = (k_count == kKT) ? CvtFp32x32ToBF16(row_src)
                                              : CvtFp32xUpTo32ToBF16(row_src, k_count);
                _mm512_storeu_si512(tile_dst + r * kKT, bf);
            }
            // zero-fill rows m_rows..15 so tile_loadd reads defined zeros
            for (int64_t r = m_rows; r < kMT; ++r) {
                _mm512_storeu_si512(tile_dst + r * kKT, _mm512_setzero_si512());
            }
        }
    }
}

// Pack the B matrix (row-major N x K floats) into AMX-B VNNI tiles:
//   For each n_blk in [0, N_blocks) and k_blk in [0, K_blocks):
//     B_vnni[n_blk][k_blk] is a (K_T/2=16) x 32 BF16 tile in VNNI form,
//     stored row-major, 64 bytes per row.
//   VNNI layout: tile_row r (0..15) contains for each n in [0,16):
//     [ B_logical[n][k_base + 2r + 0], B_logical[n][k_base + 2r + 1] ]
//   packed as one dword (low BF16, high BF16). So the byte layout per row
//   is 16 * 4 = 64 bytes, where bytes [4n..4n+1]=BF16(B[n, 2r]) and
//   bytes [4n+2..4n+3]=BF16(B[n, 2r+1]).
//
// Implementation: for each tile row r (=> two K positions ka=2r, kb=2r+1),
// gather the n_rows BF16 pairs (one pair per n in the block) into a 64-byte
// row.  We do this scalar-per-pair; this is the cold path during KMeans
// (run once per outer query-batch iteration) and packing throughput is not
// the bottleneck (the GEMM itself is).
static void
PackBRowMajorToBF16VnniTiles(
    const float* b, int64_t n, int64_t k, int64_t k_padded, uint16_t* out_bf16) {
    const int64_t k_blocks = k_padded / kKT;
    auto fp32_to_bf16 = [](float f) -> uint16_t {
        // Round-to-nearest-even, matching _mm512_cvtne2ps_pbh semantics
        // closely enough for the packed-zero-padding case; we only use
        // this for the K-tail and N-tail-of-block where vectorising is
        // not worth the complexity.
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        if (((bits >> 16) & 1) == 1) {
            bits += 0x8000;  // round up if bit-16 set (RNE tie -> even)
        } else {
            bits += 0x7FFF;
        }
        return static_cast<uint16_t>(bits >> 16);
    };
    for (int64_t n_blk = 0; n_blk < (n + kNT - 1) / kNT; ++n_blk) {
        const int64_t n_base = n_blk * kNT;
        const int64_t n_cols = std::min<int64_t>(kNT, n - n_base);
        uint16_t* blk_dst = out_bf16 + n_blk * k_blocks * (kKT / 2) * (kNT * 2);
        for (int64_t k_blk = 0; k_blk < k_blocks; ++k_blk) {
            const int64_t k_base = k_blk * kKT;
            uint16_t* tile_dst = blk_dst + k_blk * (kKT / 2) * (kNT * 2);
            std::memset(tile_dst, 0, (kKT / 2) * (kNT * 2) * sizeof(uint16_t));
            for (int64_t r = 0; r < kKT / 2; ++r) {
                const int64_t ka = k_base + 2 * r;
                const int64_t kb = k_base + 2 * r + 1;
                const bool ka_in = (ka < k);
                const bool kb_in = (kb < k);
                uint16_t* row_dst = tile_dst + r * (kNT * 2);
                for (int64_t nc = 0; nc < n_cols; ++nc) {
                    const float* nrow = b + (n_base + nc) * k;
                    row_dst[nc * 2 + 0] = ka_in ? fp32_to_bf16(nrow[ka]) : uint16_t{0};
                    row_dst[nc * 2 + 1] = kb_in ? fp32_to_bf16(nrow[kb]) : uint16_t{0};
                }
            }
        }
    }
}

}  // namespace
#endif  // ENABLE_AMX

bool
SgemmBF16IPColMajorOut(int64_t m,
                       int64_t n,
                       int64_t k,
                       float alpha,
                       const float* RESTRICT a_row_major,
                       const float* RESTRICT b_row_major,
                       float* RESTRICT c_col_major,
                       int64_t ldc) {
#if defined(ENABLE_AMX)
    if (!SimdStatus::SupportAMXBF16()) {
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0 || ldc < m) {
        return false;
    }

    const int64_t k_padded = (k + kKT - 1) & ~(kKT - 1);
    const int64_t k_blocks = k_padded / kKT;
    const int64_t m_blocks = (m + kMT - 1) / kMT;
    const int64_t n_blocks = (n + kNT - 1) / kNT;

    // Pre-pack A and B into BF16 tile layouts.
    // A packed bytes = m_blocks * k_blocks * 16 * 32 * sizeof(uint16_t)
    // B packed bytes = n_blocks * k_blocks * 16 * 32 * sizeof(uint16_t)
    std::vector<uint16_t> a_packed(static_cast<size_t>(m_blocks) * k_blocks * kMT * kKT);
    std::vector<uint16_t> b_packed(static_cast<size_t>(n_blocks) * k_blocks * kMT * kKT);
    PackARowMajorToBF16Tiles(a_row_major, m, k, k_padded, a_packed.data());
    PackBRowMajorToBF16VnniTiles(b_row_major, n, k, k_padded, b_packed.data());

    _tile_loadconfig(&kBF16GemmTileConfig);

    alignas(64) float c_tile[kMT * kNT];
    const __m512 v_alpha = _mm512_set1_ps(alpha);

    // Cache blocking on N to keep the working set in L2.
    // Per (mb,nb) we consume k_padded*2 bytes of A-block plus k_padded*2 bytes of B-block.
    // For k=960 this is ~3.75 KB each. Choosing nc_step=512 keeps the active
    // B-chunk around k_padded*2*nc_step bytes (~1.9 MB at k=960) which fits L2
    // on Granite Rapids while A-block re-use across nb amortises A streaming.
    constexpr int64_t kNcStep = 512;
    const int64_t a_block_stride = k_blocks * kMT * kKT;
    const int64_t b_block_stride = k_blocks * kMT * kKT;

    for (int64_t n_outer = 0; n_outer < n_blocks; n_outer += kNcStep / kNT) {
        const int64_t nb_end = std::min<int64_t>(n_outer + kNcStep / kNT, n_blocks);
        for (int64_t mb = 0; mb < m_blocks; ++mb) {
            const int64_t m_base = mb * kMT;
            const int64_t m_rows = std::min<int64_t>(kMT, m - m_base);
            const uint16_t* a_block_base = a_packed.data() + mb * a_block_stride;

            for (int64_t nb = n_outer; nb < nb_end; ++nb) {
                const int64_t n_base = nb * kNT;
                const int64_t n_cols = std::min<int64_t>(kNT, n - n_base);
                const uint16_t* b_block_base = b_packed.data() + nb * b_block_stride;

                _tile_zero(2);
                for (int64_t kb = 0; kb < k_blocks; ++kb) {
                    _tile_loadd(0, a_block_base + kb * kMT * kKT, 64);
                    _tile_loadd(1, b_block_base + kb * kMT * kKT, 64);
                    _tile_dpbf16ps(2, 0, 1);
                }
                _tile_stored(2, c_tile, kNT * sizeof(float));

                // Scale by alpha and write to col-major output.
                // c_tile[r * 16 + c] corresponds to logical (m_base+r, n_base+c).
                // Output is col-major with ldc: c_col_major[m + n * ldc].
                if (m_rows == kMT && n_cols == kNT && alpha != 1.0F) {
                    for (int64_t r = 0; r < kMT; ++r) {
                        __m512 row = _mm512_load_ps(c_tile + r * kNT);
                        __m512 scaled = _mm512_mul_ps(row, v_alpha);
                        alignas(64) float scaled_arr[kNT];
                        _mm512_store_ps(scaled_arr, scaled);
                        for (int64_t c = 0; c < kNT; ++c) {
                            c_col_major[(m_base + r) + (n_base + c) * ldc] = scaled_arr[c];
                        }
                    }
                } else {
                    for (int64_t r = 0; r < m_rows; ++r) {
                        for (int64_t c = 0; c < n_cols; ++c) {
                            c_col_major[(m_base + r) + (n_base + c) * ldc] =
                                alpha * c_tile[r * kNT + c];
                        }
                    }
                }
            }
        }
    }
    _tile_release();
    return true;
#else
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a_row_major;
    (void)b_row_major;
    (void)c_col_major;
    (void)ldc;
    return false;
#endif
}

}  // namespace vsag::amx
