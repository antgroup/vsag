
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
// This translation unit is compiled with -mamx-tile -mamx-int8 plus
// AVX-512F/BW/DQ/VL/VNNI so AMX kernels can use AVX-512 to preprocess
// inputs (pack/quantize) and to handle tail elements that don't fill
// an AMX tile.

#include "sq8_uniform_simd.h"

#if defined(ENABLE_AMX)
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
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
    // _tile_loadd takes a signed 32-bit stride. In practice code_stride is
    // O(dim) bytes and never anywhere near 2 GiB, but fall back to the
    // AVX-512 scalar loop in release builds rather than truncating the cast.
    if (n_codes < kAmxMinCount || code_stride > static_cast<uint64_t>(INT32_MAX)) {
        for (uint64_t j = 0; j < n_codes; ++j) {
            out[j] = avx512::SQ8UniformComputeCodesIP(query, codes + j * code_stride, dim);
        }
        return;
    }

    // dim split into (dim_aligned = floor(dim/64)*64) handled by AMX,
    // tail handled by AVX-512 per-pair on the residue.
    uint64_t dim_aligned = dim & ~uint64_t{63};
    uint64_t tail = dim - dim_aligned;

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

}  // namespace vsag::amx
