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

#include "hamming_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "simd.h"
#include "simd_status.h"

using namespace vsag;

namespace {

// Ask the compiled simd library about ARM support: simd_test does not receive ENABLE_SVE, so the
// inline SimdStatus::SupportSVE() is always false in this file even where the library dispatches
// to its SVE kernels.
const SimdStatus&
library_simd_status() {
    static const SimdStatus status = [] {
        SimdStatus::Init();
        return setup_simd();
    }();
    return status;
}

bool
library_supports_neon() {
    const auto& status = library_simd_status();
    return status.dist_support_neon and status.runtime_has_neon;
}

bool
library_supports_sve() {
    const auto& status = library_simd_status();
    return status.dist_support_sve and status.runtime_has_sve;
}

struct HammingKernel {
    std::string name;
    HammingDistanceType single;
    HammingDistanceBatch4Type batch4;
    HammingDistanceScanType scan;
};

#define HAMMING_KERNEL(ns)                                                           \
    HammingKernel {                                                                  \
        #ns, ns::HammingDistance, ns::HammingDistanceBatch4, ns::HammingDistanceScan \
    }

// Every implementation the running CPU can execute, plus the dispatched entry points.
std::vector<HammingKernel>
available_kernels() {
    std::vector<HammingKernel> kernels{
        HAMMING_KERNEL(generic),
        HammingKernel{"dispatch",
                      vsag::HammingDistance,
                      vsag::HammingDistanceBatch4,
                      vsag::HammingDistanceScan},
    };
    if (SimdStatus::SupportPOPCNT()) {
        kernels.push_back(HAMMING_KERNEL(sse));
    }
    if (SimdStatus::SupportAVX2()) {
        kernels.push_back(HAMMING_KERNEL(avx2));
    }
    if (SimdStatus::SupportAVX512()) {
        kernels.push_back(HAMMING_KERNEL(avx512));
    }
    if (SimdStatus::SupportAVX512() and SimdStatus::SupportAVX512VPOPCNTDQ()) {
        kernels.push_back(HAMMING_KERNEL(avx512vpopcntdq));
    }
    if (library_supports_neon()) {
        kernels.push_back(HAMMING_KERNEL(neon));
    }
    if (library_supports_sve()) {
        kernels.push_back(HAMMING_KERNEL(sve));
    }
    return kernels;
}

uint64_t
bytes_of(uint64_t dim) {
    return (dim + 7) / 8;
}

// Independent reference: compares the vectors bit by bit, reading logical bit i as
// (bytes[i / 8] >> (7 - i % 8)) & 1 (MSB-first packing). No popcount, no word tricks.
uint64_t
reference_hamming(const uint8_t* x, const uint8_t* y, uint64_t dim) {
    uint64_t distance = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        const auto shift = 7 - (i % 8);
        const auto bit_x = (x[i / 8] >> shift) & 1U;
        const auto bit_y = (y[i / 8] >> shift) & 1U;
        distance += bit_x != bit_y ? 1 : 0;
    }
    return distance;
}

std::vector<uint8_t>
random_bytes(uint64_t count, std::mt19937_64& rng) {
    std::vector<uint8_t> bytes(count);
    for (auto& value : bytes) {
        value = static_cast<uint8_t>(rng());
    }
    return bytes;
}

void
check_all_kernels(const uint8_t* query, const uint8_t* const* codes, uint64_t dim) {
    uint64_t expected[4];
    for (uint64_t k = 0; k < 4; ++k) {
        expected[k] = reference_hamming(query, codes[k], dim);
    }
    for (const auto& kernel : available_kernels()) {
        INFO("kernel=" << kernel.name << " dim=" << dim);
        for (uint64_t k = 0; k < 4; ++k) {
            REQUIRE(kernel.single(query, codes[k], dim) == expected[k]);
            REQUIRE(kernel.single(codes[k], query, dim) == expected[k]);
        }
        uint64_t results[4] = {~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}};
        kernel.batch4(query, codes[0], codes[1], codes[2], codes[3], dim, results);
        for (uint64_t k = 0; k < 4; ++k) {
            REQUIRE(results[k] == expected[k]);
        }
    }
}

// Dimensions required by the binary-vector specification plus every SIMD block boundary
// (8, 32, 64 and 128 bytes) and both sides of the NEON accumulator flush.
const std::vector<uint64_t> K_CHECKED_DIMS = {
    0,    1,    2,    3,    4,    5,     6,      7,      8,      9,      15,    16,   17,   31,
    32,   33,   63,   64,   65,   127,   128,    129,    191,    192,    193,   255,  256,  257,
    511,  512,  513,  767,  768,  1023,  1024,   1025,   1535,   2047,   2048,  2049, 4095, 4096,
    4097, 8191, 8192, 8193, 9999, 65536, 262144, 262151, 262152, 262153, 300007};

}  // namespace

TEST_CASE("Hamming kernels match a bit-by-bit reference", "[ut][simd][hamming]") {
    std::mt19937_64 rng(2872);
    for (const auto dim : K_CHECKED_DIMS) {
        const auto num_bytes = bytes_of(dim);
        // Random contents everywhere, including the padding bits of the last byte. The buffers
        // are exactly ceil(dim / 8) bytes long, so sanitizers flag any read past a vector.
        auto query = random_bytes(num_bytes, rng);
        std::vector<std::vector<uint8_t>> codes;
        for (int k = 0; k < 4; ++k) {
            codes.push_back(random_bytes(num_bytes, rng));
        }
        const uint8_t* code_ptrs[4] = {
            codes[0].data(), codes[1].data(), codes[2].data(), codes[3].data()};
        check_all_kernels(query.data(), code_ptrs, dim);
    }
}

TEST_CASE("Hamming kernels on random dimensions and offsets", "[ut][simd][hamming]") {
    std::mt19937_64 rng(424242);
    std::uniform_int_distribution<uint64_t> dim_dist(1, 6000);
    std::uniform_int_distribution<uint64_t> offset_dist(0, 63);
    for (int round = 0; round < 200; ++round) {
        const auto dim = dim_dist(rng);
        const auto num_bytes = bytes_of(dim);
        // Unaligned starting addresses exercise the unaligned and masked loads.
        const auto query_offset = offset_dist(rng);
        auto query_storage = random_bytes(num_bytes + query_offset, rng);
        std::vector<std::vector<uint8_t>> code_storage;
        const uint8_t* code_ptrs[4];
        for (uint64_t k = 0; k < 4; ++k) {
            const auto offset = offset_dist(rng);
            code_storage.push_back(random_bytes(num_bytes + offset, rng));
            code_ptrs[k] = code_storage.back().data() + offset;
        }
        check_all_kernels(query_storage.data() + query_offset, code_ptrs, dim);
    }
}

TEST_CASE("Hamming kernels ignore padding bits", "[ut][simd][hamming]") {
    std::mt19937_64 rng(7);
    for (uint64_t dim = 1; dim <= 1100; ++dim) {
        if (dim % 8 == 0) {
            continue;
        }
        const auto num_bytes = bytes_of(dim);
        auto x = random_bytes(num_bytes, rng);
        auto y = random_bytes(num_bytes, rng);
        const uint8_t padding = static_cast<uint8_t>(0xFFU >> (dim % 8));
        auto x_clean = x;
        auto y_clean = y;
        x_clean.back() &= static_cast<uint8_t>(~padding);
        y_clean.back() &= static_cast<uint8_t>(~padding);
        // Opposite garbage in the padding of the two vectors must not add any distance.
        auto x_dirty = x_clean;
        auto y_dirty = y_clean;
        x_dirty.back() |= padding;
        const auto expected = reference_hamming(x_clean.data(), y_clean.data(), dim);
        for (const auto& kernel : available_kernels()) {
            INFO("kernel=" << kernel.name << " dim=" << dim);
            REQUIRE(kernel.single(x_clean.data(), y_clean.data(), dim) == expected);
            REQUIRE(kernel.single(x_dirty.data(), y_dirty.data(), dim) == expected);
            REQUIRE(kernel.single(x.data(), y.data(), dim) == expected);
            uint64_t results[4];
            kernel.batch4(x_dirty.data(),
                          y_clean.data(),
                          y_dirty.data(),
                          y.data(),
                          x_clean.data(),
                          dim,
                          results);
            REQUIRE(results[0] == expected);
            REQUIRE(results[1] == expected);
            REQUIRE(results[2] == expected);
            REQUIRE(results[3] == 0);
        }
    }
}

TEST_CASE("Hamming kernels on structured vectors", "[ut][simd][hamming]") {
    for (const uint64_t dim : {1, 7, 8, 9, 63, 64, 65, 127, 128, 1000, 4096}) {
        const auto num_bytes = bytes_of(dim);
        const std::vector<uint8_t> zeros(num_bytes, 0x00);
        const std::vector<uint8_t> ones(num_bytes, 0xFF);
        std::vector<uint8_t> alternating(num_bytes, 0xAA);
        for (const auto& kernel : available_kernels()) {
            INFO("kernel=" << kernel.name << " dim=" << dim);
            REQUIRE(kernel.single(zeros.data(), zeros.data(), dim) == 0);
            REQUIRE(kernel.single(ones.data(), ones.data(), dim) == 0);
            REQUIRE(kernel.single(zeros.data(), ones.data(), dim) == dim);
            // 0xAA sets bits 0, 2, 4, ...: ceil(dim / 2) of the first dim bits.
            REQUIRE(kernel.single(zeros.data(), alternating.data(), dim) == (dim + 1) / 2);
            REQUIRE(kernel.single(ones.data(), alternating.data(), dim) == dim / 2);
        }
        // A single set bit at every logical position is at distance one from zero, which pins
        // the MSB-first bit order: bit 0 is 0x80 of byte 0, bit 7 is 0x01 of byte 0.
        for (uint64_t bit = 0; bit < dim; bit += (dim > 200 ? 37 : 1)) {
            std::vector<uint8_t> single(num_bytes, 0x00);
            single[bit / 8] = static_cast<uint8_t>(0x80U >> (bit % 8));
            for (const auto& kernel : available_kernels()) {
                INFO("kernel=" << kernel.name << " dim=" << dim << " bit=" << bit);
                REQUIRE(kernel.single(zeros.data(), single.data(), dim) == 1);
                // The same byte pattern lies past the end of a vector that holds only `bit` bits.
                REQUIRE(kernel.single(zeros.data(), single.data(), bit) == 0);
            }
        }
    }
}

TEST_CASE("Hamming scan kernels match per-code distances", "[ut][simd][hamming]") {
    std::mt19937_64 rng(2026);
    std::vector<uint64_t> dims = K_CHECKED_DIMS;
    // Every dimension whose codes take 8, 16 or 32 bytes, the sizes with a packed scan layout.
    for (const uint64_t last : {64, 128, 256}) {
        for (uint64_t dim = last - 7; dim <= last; ++dim) {
            dims.push_back(dim);
        }
    }
    for (const auto dim : dims) {
        const auto num_bytes = bytes_of(dim);
        const std::vector<uint64_t> counts =
            dim > 10000 ? std::vector<uint64_t>{1, 9}
                        : std::vector<uint64_t>{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31};
        for (const auto count : counts) {
            // Codes start at an unaligned address and the buffer ends right after the last one,
            // so sanitizers flag any read past it. Padding bits hold random values.
            const uint64_t offset = rng() % 64;
            const auto storage = random_bytes(offset + count * num_bytes, rng);
            const uint8_t* codes = storage.data() + offset;
            const auto query = random_bytes(num_bytes, rng);
            std::vector<float> expected(count);
            for (uint64_t k = 0; k < count; ++k) {
                expected[k] =
                    static_cast<float>(reference_hamming(query.data(), codes + k * num_bytes, dim));
            }
            for (const auto& kernel : available_kernels()) {
                INFO("kernel=" << kernel.name << " dim=" << dim << " count=" << count);
                std::vector<float> dists(count + 1, -1.0F);
                kernel.scan(query.data(), codes, dim, count, dists.data());
                for (uint64_t k = 0; k < count; ++k) {
                    REQUIRE(dists[k] == expected[k]);
                }
                REQUIRE(dists[count] == -1.0F);
            }
        }
    }
}

TEST_CASE("Hamming dispatch selects the best available kernel", "[ut][simd][hamming]") {
    HammingKernel expected = HAMMING_KERNEL(generic);
    if (library_supports_neon()) {
        expected = HAMMING_KERNEL(neon);
    }
    if (library_supports_sve()) {
        expected = HAMMING_KERNEL(sve);
    }
    if (SimdStatus::SupportPOPCNT()) {
        expected = HAMMING_KERNEL(sse);
    }
    if (SimdStatus::SupportAVX2()) {
        expected = HAMMING_KERNEL(avx2);
    }
    if (SimdStatus::SupportAVX512()) {
        expected = HAMMING_KERNEL(avx512);
    }
    if (SimdStatus::SupportAVX512() and SimdStatus::SupportAVX512VPOPCNTDQ()) {
        expected = HAMMING_KERNEL(avx512vpopcntdq);
    }
    INFO("expected=" << expected.name);
    REQUIRE(vsag::HammingDistance == expected.single);
    REQUIRE(vsag::HammingDistanceBatch4 == expected.batch4);
    REQUIRE(vsag::HammingDistanceScan == expected.scan);
}

#define BENCHMARK_HAMMING(Simd)                                                           \
    BENCHMARK_ADVANCED(#Simd) {                                                           \
        uint64_t sum = 0;                                                                 \
        for (uint64_t i = 0; i < count; ++i) {                                            \
            sum += Simd::HammingDistance(query.data(), base.data() + i * num_bytes, dim); \
        }                                                                                 \
        return sum;                                                                       \
    }

TEST_CASE("Hamming kernel benchmark", "[!benchmark][simd][hamming]") {
    const uint64_t dim = 1024;
    const uint64_t num_bytes = bytes_of(dim);
    const uint64_t count = 10000;
    std::mt19937_64 rng(1);
    const auto query = random_bytes(num_bytes, rng);
    const auto base = random_bytes(num_bytes * count, rng);
    BENCHMARK_HAMMING(generic);
    if (SimdStatus::SupportPOPCNT()) {
        BENCHMARK_HAMMING(sse);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_HAMMING(avx2);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_HAMMING(avx512);
    }
    if (SimdStatus::SupportAVX512() and SimdStatus::SupportAVX512VPOPCNTDQ()) {
        BENCHMARK_HAMMING(avx512vpopcntdq);
    }
    if (library_supports_neon()) {
        BENCHMARK_HAMMING(neon);
    }
    if (library_supports_sve()) {
        BENCHMARK_HAMMING(sve);
    }
}
