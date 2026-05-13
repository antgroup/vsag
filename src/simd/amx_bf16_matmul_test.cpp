
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

#include "amx_bf16_matmul.h"

#include <cmath>
#include <random>
#include <vector>

#include "impl/blas/blas_function.h"
#include "simd_status.h"
#include "unittest.h"

using namespace vsag;

namespace {

// Reference SGEMM for our specific call shape:
//   C[m + n*ldc] = alpha * < A_row[m], B_row[n] >
// (A is row-major M x K, B is row-major N x K, C is col-major with ldc).
void
ReferenceIPColMajor(int64_t m,
                    int64_t n,
                    int64_t k,
                    float alpha,
                    const float* a,
                    const float* b,
                    float* c,
                    int64_t ldc) {
    BlasFunction::Sgemm(BlasFunction::ColMajor,
                        BlasFunction::Trans,
                        BlasFunction::NoTrans,
                        static_cast<int32_t>(m),
                        static_cast<int32_t>(n),
                        static_cast<int32_t>(k),
                        alpha,
                        a,
                        static_cast<int32_t>(k),
                        b,
                        static_cast<int32_t>(k),
                        0.0F,
                        c,
                        static_cast<int32_t>(ldc));
}

std::vector<float>
RandomFloats(int64_t count, uint32_t seed, float lo = -1.0F, float hi = 1.0F) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(count));
    for (auto& f : v) {
        f = dist(gen);
    }
    return v;
}

// Tolerance model for BF16 accumulation against FP32 SGEMM.
//
// BF16 has 8 mantissa bits => roughly 4e-3 relative precision per element.
// Each output is a sum of K BF16 products; for random inputs uniform in
// [-1, 1] the dot product magnitude grows as O(sqrt(K)) so the absolute
// error from BF16 rounding scales as |alpha| * sqrt(K) * 4e-3.  We use
// a generous 0.02 * |alpha| * sqrt(K) absolute tolerance (~5x predicted)
// plus a small per-element floor.
//
// Relative tolerance alone is not meaningful for inner-product outputs
// that happen to land near zero (the SGEMM reference and AMX kernel can
// both be tiny and the relative error is unbounded for the divide).
constexpr float kAbsTolFloor = 1e-3F;

float
AbsTolerance(int64_t k, float alpha) {
    return kAbsTolFloor + 0.02F * std::fabs(alpha) * std::sqrt(static_cast<float>(k));
}

bool
CloseEnough(float ref, float got, float abs_tol) {
    return std::fabs(ref - got) <= abs_tol;
}

}  // namespace

TEST_CASE("AMX BF16 GEMM matches SGEMM (ColMajor IP output)", "[ut][simd][amx]") {
    if (!SimdStatus::SupportAMXBF16()) {
        // CI may run on non-AMX hosts; the fallback path is exercised by the
        // higher-level KMeans tests, so we just skip this microtest there.
        SUCCEED("AMX-BF16 unavailable on this host; skipping.");
        return;
    }

    struct Shape {
        int64_t m;
        int64_t n;
        int64_t k;
    };
    // A spread of M/N/K combos that exercise:
    //   * exact multiples of (16, 16, 32)
    //   * M-tail, N-tail, K-tail (separately and together)
    //   * dims similar to real workloads (sift=128, gist=960)
    const std::vector<Shape> shapes = {
        {16, 16, 32},
        {16, 16, 64},
        {17, 16, 32},
        {16, 17, 32},
        {16, 16, 33},
        {31, 31, 31},
        {32, 64, 128},
        {48, 80, 96},
        {64, 64, 960},
        {100, 50, 128},
    };

    for (const auto& s : shapes) {
        DYNAMIC_SECTION("M=" << s.m << " N=" << s.n << " K=" << s.k) {
            auto a = RandomFloats(s.m * s.k, 0x114u);
            auto b = RandomFloats(s.n * s.k, 0x514u);
            std::vector<float> c_ref(static_cast<size_t>(s.m) * s.n, 0.0F);
            std::vector<float> c_amx(static_cast<size_t>(s.m) * s.n, 0.0F);

            const float alpha = -2.0F;
            ReferenceIPColMajor(s.m, s.n, s.k, alpha, a.data(), b.data(), c_ref.data(), s.m);
            bool ok = amx::SgemmBF16IPColMajorOut(
                s.m, s.n, s.k, alpha, a.data(), b.data(), c_amx.data(), s.m);
            REQUIRE(ok);

            const float abs_tol = AbsTolerance(s.k, alpha);
            int mismatches = 0;
            float worst_abs = 0.0F;
            float worst_rel = 0.0F;
            for (int64_t j = 0; j < s.n; ++j) {
                for (int64_t i = 0; i < s.m; ++i) {
                    float ref = c_ref[i + j * s.m];
                    float got = c_amx[i + j * s.m];
                    if (!CloseEnough(ref, got, abs_tol)) {
                        ++mismatches;
                    }
                    float diff = std::fabs(ref - got);
                    worst_abs = std::max(worst_abs, diff);
                    if (std::fabs(ref) > 1e-3F) {
                        worst_rel = std::max(worst_rel, diff / std::fabs(ref));
                    }
                }
            }
            INFO("abs_tol=" << abs_tol << " worst_abs=" << worst_abs << " worst_rel=" << worst_rel
                            << " mismatches=" << mismatches);
            REQUIRE(mismatches == 0);
        }
    }
}

TEST_CASE("AMX BF16 GEMM returns false for invalid shapes", "[ut][simd][amx]") {
    if (!SimdStatus::SupportAMXBF16()) {
        SUCCEED("AMX-BF16 unavailable on this host; skipping.");
        return;
    }
    float a = 0.0F, b = 0.0F, c = 0.0F;
    // m <= 0
    REQUIRE_FALSE(amx::SgemmBF16IPColMajorOut(0, 1, 1, 1.0F, &a, &b, &c, 1));
    // n <= 0
    REQUIRE_FALSE(amx::SgemmBF16IPColMajorOut(1, 0, 1, 1.0F, &a, &b, &c, 1));
    // k <= 0
    REQUIRE_FALSE(amx::SgemmBF16IPColMajorOut(1, 1, 0, 1.0F, &a, &b, &c, 1));
    // ldc < m
    REQUIRE_FALSE(amx::SgemmBF16IPColMajorOut(4, 1, 1, 1.0F, &a, &b, &c, 1));
}
