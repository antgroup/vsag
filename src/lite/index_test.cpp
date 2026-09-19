// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "vsag/lite/index.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_map>

#include "fp32_distance.h"

using vsag::lite::Index;

TEST_CASE("Lite validates inputs and preserves IDs through physical deletion", "[lite]") {
    REQUIRE_FALSE(Index::Create(0));
    REQUIRE_FALSE(Index::Create(UINT64_MAX));
    auto created = Index::Create(2);
    REQUIRE(created);
    auto& index = **created;
    float a[]{1, 2};
    float b[]{3, 4};
    float invalid[]{NAN, 0};
    REQUIRE(index.Search(a, 2, 10)->empty());
    REQUIRE_FALSE(index.Add(1, nullptr, 2));
    REQUIRE_FALSE(index.Add(1, a, 1));
    REQUIRE_FALSE(index.Add(1, invalid, 2));
    REQUIRE(index.Add(INT64_MIN, a, 2));
    REQUIRE(index.Add(INT64_MAX, b, 2));
    REQUIRE_FALSE(index.Add(INT64_MIN, b, 2));
    REQUIRE(index.Size() == 2);
    REQUIRE_FALSE(index.Update(99, a, 2));
    REQUIRE_FALSE(index.Update(INT64_MIN, nullptr, 2));
    REQUIRE_FALSE(index.Search(invalid, 2, 1));
    REQUIRE_FALSE(index.Search(a, 1, 1));
    REQUIRE(index.Search(a, 2, 0)->empty());
    REQUIRE(index.Search(a, 2, UINT64_MAX)->size() == 2);
    REQUIRE(index.Remove(INT64_MIN));
    REQUIRE_FALSE(index.Remove(INT64_MIN));
    REQUIRE(index.Update(INT64_MAX, a, 2));
    REQUIRE(index.Search(a, 2, 1)->front().id == INT64_MAX);
    REQUIRE(index.Add(INT64_MIN, a, 2));
    REQUIRE(index.Search(a, 2, 2)->front().id == INT64_MIN);
    REQUIRE(index.Remove(INT64_MAX));
    REQUIRE(index.Remove(INT64_MIN));
    REQUIRE(index.Size() == 0);
}

TEST_CASE("Lite search filters external IDs across CRUD and snapshot load", "[lite]") {
    auto created = Index::Create(2);
    auto& index = **created;
    const float query[]{0, 0};
    const float first[]{0, 0};
    const float second[]{1, 0};
    const float third[]{2, 0};
    const float fourth[]{3, 0};
    REQUIRE(index.Add(101, first, 2));
    REQUIRE(index.Add(-7, second, 2));
    REQUIRE(index.Add(303, third, 2));
    REQUIRE(index.Add(404, fourth, 2));

    const auto unfiltered = index.Search(query, 2, 10);
    REQUIRE(unfiltered);
    const vsag::lite::IdFilter empty_filter;
    const auto empty_filter_result = index.Search(query, 2, 10, empty_filter);
    REQUIRE(empty_filter_result);
    REQUIRE(empty_filter_result->size() == unfiltered->size());
    for (uint64_t i = 0; i < unfiltered->size(); ++i) {
        REQUIRE((*empty_filter_result)[i].id == (*unfiltered)[i].id);
        REQUIRE((*empty_filter_result)[i].distance == (*unfiltered)[i].distance);
    }

    std::vector<int64_t> checked_ids;
    const vsag::lite::IdFilter selected = [&checked_ids](int64_t id) {
        checked_ids.push_back(id);
        return id == -7 or id == 404;
    };
    const auto filtered = index.Search(query, 2, 10, selected);
    REQUIRE(filtered);
    REQUIRE(checked_ids == std::vector<int64_t>{101, -7, 303, 404});
    REQUIRE(filtered->size() == 2);
    REQUIRE((*filtered)[0].id == -7);
    REQUIRE((*filtered)[1].id == 404);

    const auto rejected = index.Search(query, 2, 4, [](int64_t) { return false; });
    REQUIRE(rejected);
    REQUIRE(rejected->empty());
    const auto fewer_than_k = index.Search(query, 2, 4, [](int64_t id) { return id == 303; });
    REQUIRE(fewer_than_k);
    REQUIRE(fewer_than_k->size() == 1);
    REQUIRE(fewer_than_k->front().id == 303);

    const float updated[]{-1, 0};
    REQUIRE(index.Update(404, updated, 2));
    REQUIRE(index.Remove(-7));
    const auto after_crud = index.Search(query, 2, 4, [](int64_t id) { return id % 2 == 0; });
    REQUIRE(after_crud);
    REQUIRE(after_crud->size() == 1);
    REQUIRE(after_crud->front().id == 404);
    REQUIRE(after_crud->front().distance == 1);

    std::stringstream stream;
    REQUIRE(index.Save(stream));
    auto loaded = Index::Load(stream);
    REQUIRE(loaded);
    const auto restored = (*loaded)->Search(query, 2, 4, [](int64_t id) { return id % 2 == 0; });
    REQUIRE(restored);
    REQUIRE(restored->size() == after_crud->size());
    REQUIRE(restored->front().id == after_crud->front().id);
    REQUIRE(restored->front().distance == after_crud->front().distance);
}

TEST_CASE("Lite v1 snapshot has fixed bytes", "[lite]") {
    auto created = Index::Create(1);
    const float value = 1.25F;
    REQUIRE((*created)->Add(-1, &value, 1));
    std::stringstream output;
    REQUIRE((*created)->Save(output));
    constexpr std::array<unsigned char, 60> expected{
        'V', 'S', 'A', 'G', 'L', 'T', '0', '1', 1, 0, 0, 0, 0, 0, 0, 0,  // version
        1,   0,   0,   0,   0,   0,   0,   0,                            // dimension
        1,   0,   0,   0,   0,   0,   0,   0,                            // count
        12,  0,   0,   0,   0,   0,   0,   0,                            // payload bytes
        1,   0,   0,   0,   0,   0,   0,   0,                            // FP32 squared L2
        255, 255, 255, 255, 255, 255, 255, 255,                          // ID -1
        0,   0,   160, 63,                                               // 1.25f
    };
    const auto bytes = output.str();
    REQUIRE(bytes.size() == expected.size());
    REQUIRE(
        std::equal(expected.begin(), expected.end(), bytes.begin(), [](unsigned char a, char b) {
            return a == static_cast<unsigned char>(b);
        }));
    std::stringstream input(
        std::string(reinterpret_cast<const char*>(expected.data()), expected.size()));
    auto loaded = Index::Load(input);
    REQUIRE(loaded);
    REQUIRE((*loaded)->Search(&value, 1, 1)->front().id == -1);
}

TEST_CASE("Lite snapshot roundtrip and malformed input", "[lite]") {
    auto created = Index::Create(2);
    auto& index = **created;
    float a[]{1.25F, -2.5F};
    REQUIRE(index.Add(INT64_MIN, a, 2));
    REQUIRE(index.Add(INT64_MAX, a, 2));
    REQUIRE(index.Add(9, a, 2));
    REQUIRE(index.Remove(9));
    std::stringstream stream;
    REQUIRE(index.Save(stream));
    const auto bytes = stream.str();
    REQUIRE(bytes.size() == 80);
    auto loaded = Index::Load(stream);
    REQUIRE(loaded);
    REQUIRE((*loaded)->Size() == 2);
    REQUIRE((*loaded)->Dim() == 2);
    auto result = (*loaded)->Search(a, 2, 2);
    REQUIRE(result);
    REQUIRE(result->front().id == INT64_MIN);
    REQUIRE(result->back().id == INT64_MAX);
    REQUIRE(result->front().distance == 0);
    REQUIRE((*loaded)->Update(INT64_MIN, a, 2));
    REQUIRE((*loaded)->Remove(INT64_MAX));
    REQUIRE((*loaded)->Add(9, a, 2));

    for (uint64_t n = 0; n < bytes.size(); ++n) {
        std::stringstream truncated(bytes.substr(0, n));
        REQUIRE_FALSE(Index::Load(truncated));
    }
    for (uint64_t offset : {0, 8, 16, 24, 32, 40}) {
        auto corrupt = bytes;
        corrupt[offset] ^= 0x7f;
        std::stringstream input(corrupt);
        REQUIRE_FALSE(Index::Load(input));
    }
    auto duplicate = bytes;
    duplicate.replace(56, 8, duplicate.substr(48, 8));
    std::stringstream duplicates(duplicate);
    REQUIRE_FALSE(Index::Load(duplicates));
    auto nan = bytes;
    nan[64] = 0;
    nan[65] = 0;
    nan[66] = static_cast<char>(0xc0);
    nan[67] = 0x7f;
    std::stringstream nonfinite(nan);
    REQUIRE_FALSE(Index::Load(nonfinite));
    std::stringstream extra(bytes + "x");
    REQUIRE_FALSE(Index::Load(extra));
    std::stringstream failing;
    failing.setstate(std::ios::badbit);
    REQUIRE_FALSE(index.Save(failing));
    std::stringstream throwing;
    throwing.exceptions(std::ios::failbit | std::ios::badbit);
    REQUIRE_FALSE(Index::Load(throwing));

    auto empty = Index::Create(2);
    std::stringstream empty_stream;
    REQUIRE((*empty)->Save(empty_stream));
    auto empty_loaded = Index::Load(empty_stream);
    REQUIRE(empty_loaded);
    REQUIRE((*empty_loaded)->Size() == 0);
    REQUIRE(index.Size() == 2);  // Bad loads never modify the existing index.
}

TEST_CASE("Lite randomized CRUD matches an independent reference", "[lite]") {
    auto created = Index::Create(7);
    auto& index = **created;
    std::unordered_map<int64_t, std::vector<float>> reference;
    std::mt19937 random(20260908);
    for (uint64_t step = 0; step < 1000; ++step) {
        const auto id = static_cast<int64_t>(random() % 80);
        std::vector<float> v(7);
        for (auto& x : v) {
            x = static_cast<float>(random() % 100) / 100;
        }
        switch (random() % 3) {
            case 0:
                REQUIRE(static_cast<bool>(index.Add(id, v.data(), 7)) ==
                        (reference.count(id) == 0));
                reference.emplace(id, v);
                break;
            case 1:
                REQUIRE(static_cast<bool>(index.Update(id, v.data(), 7)) ==
                        (reference.count(id) != 0));
                if (reference.count(id) != 0) {
                    reference[id] = v;
                }
                break;
            default:
                REQUIRE(index.Remove(id) == (reference.erase(id) != 0));
        }
        REQUIRE(index.Size() == reference.size());
        std::vector<vsag::lite::Neighbor> expected;
        for (const auto& record : reference) {
            float distance = 0;
            for (uint64_t d = 0; d < 7; ++d) {
                float delta = v[d] - record.second[d];
                distance += delta * delta;
            }
            expected.push_back({record.first, distance});
        }
        std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
            return a.distance < b.distance or (a.distance == b.distance and a.id < b.id);
        });
        const uint64_t k = random() % 20;
        expected.resize(std::min<uint64_t>(k, expected.size()));
        auto actual = index.Search(v.data(), 7, k);
        REQUIRE(actual);
        REQUIRE(actual->size() == expected.size());
        for (uint64_t i = 0; i < expected.size(); ++i) {
            REQUIRE((*actual)[i].id == expected[i].id);
            REQUIRE(std::abs((*actual)[i].distance - expected[i].distance) < 1e-5F);
        }
    }
}

TEST_CASE("Lite 100k high-dimensional end-to-end smoke", "[.][lite-scale]") {
    auto created = Index::Create(128);
    auto& index = **created;
    std::vector<float> vector(128, 0);
    for (int64_t id = 0; id < 100000; ++id) {
        vector[0] = static_cast<float>(id);
        REQUIRE(index.Add(id, vector.data(), 128));
    }
    REQUIRE(index.Remove(50000));
    vector[0] = -1;
    REQUIRE(index.Update(10, vector.data(), 128));
    auto before = index.Search(vector.data(), 128, 10);
    REQUIRE(before);
    REQUIRE(before->front().id == 10);
    std::stringstream stream;
    REQUIRE(index.Save(stream));
    auto loaded = Index::Load(stream);
    REQUIRE(loaded);
    REQUIRE((*loaded)->Size() == 99999);
    auto after = (*loaded)->Search(vector.data(), 128, 10);
    REQUIRE(after);
    REQUIRE(after->size() == before->size());
    for (uint64_t i = 0; i < before->size(); ++i) {
        REQUIRE((*after)[i].id == (*before)[i].id);
        REQUIRE((*after)[i].distance == (*before)[i].distance);
    }
}

TEST_CASE("Lite stream failure contracts", "[lite]") {
    auto created = Index::Create(1);
    float value = 0;
    REQUIRE((*created)->Add(1, &value, 1));
    struct RejectWrites : std::streambuf {
    } buffer;
    std::ostream output(&buffer);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    REQUIRE_FALSE((*created)->Save(output));
    std::istream nonseekable(&buffer);
    REQUIRE_FALSE(Index::Load(nonseekable));

    std::stringstream saved;
    REQUIRE((*created)->Save(saved));
    struct ShortRead : std::stringbuf {
        explicit ShortRead(const std::string& bytes) : std::stringbuf(bytes) {
        }
        pos_type
        seekoff(off_type offset, std::ios::seekdir direction, std::ios::openmode mode) override {
            if (direction == std::ios::end) {
                return 60;  // Claims complete input, but payload is truncated.
            }
            return std::stringbuf::seekoff(offset, direction, mode);
        }
    } short_buffer(saved.str().substr(0, 55));
    std::istream truncated(&short_buffer);
    REQUIRE_FALSE(Index::Load(truncated));

    // Empty snapshots still must reject a dimension larger than the vector container limit.
    auto empty = Index::Create(1);
    std::stringstream empty_bytes;
    REQUIRE((*empty)->Save(empty_bytes));
    auto invalid = empty_bytes.str();
    const uint64_t enormous = (UINT64_MAX - 8) / 4;
    for (uint64_t i = 0; i < 8; ++i) {
        invalid[16 + i] = static_cast<char>((enormous >> (8 * i)) & 255);
    }
    std::stringstream excessive(invalid);
    REQUIRE_FALSE(Index::Load(excessive));
}

TEST_CASE("Lite FP32 ISA kernels preserve squared-L2 across short and tail dimensions", "[lite]") {
    float query[129];
    float vector[129];
    for (uint64_t i = 0; i < 129; ++i) {
        query[i] = static_cast<float>(i % 11) * 0.125F - 0.375F;
        vector[i] = static_cast<float>(i % 7) * 0.0625F + 0.25F;
    }
    using vsag::lite::detail::FP32Distance;
    const FP32Distance generic = vsag::lite::detail::generic_fp32_distance;
    const FP32Distance selected = vsag::lite::detail::select_fp32_distance();
    std::vector<FP32Distance> kernels{generic, selected};
#ifdef VSAG_LITE_HAS_X86_SIMD
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse4.1")) {
        kernels.push_back(vsag::lite::detail::sse_fp32_distance);
    }
    if (__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) {
        kernels.push_back(vsag::lite::detail::avx2_fp32_distance);
    }
    if (__builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512dq") and
        __builtin_cpu_supports("avx512bw") and __builtin_cpu_supports("avx512vl") and
        __builtin_cpu_supports("avx2")) {
        kernels.push_back(vsag::lite::detail::avx512_fp32_distance);
        REQUIRE(selected == vsag::lite::detail::avx512_fp32_distance);
    } else if (__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) {
        REQUIRE(selected == vsag::lite::detail::avx2_fp32_distance);
    } else if (__builtin_cpu_supports("sse4.1")) {
        REQUIRE(selected == vsag::lite::detail::sse_fp32_distance);
    } else {
        REQUIRE(selected == generic);
    }
#else
    REQUIRE(selected == generic);
#endif
    for (const uint64_t dim :
         {1ULL, 3ULL, 4ULL, 5ULL, 8ULL, 9ULL, 16ULL, 17ULL, 32ULL, 33ULL, 128ULL, 129ULL}) {
        const float expected = generic(query, vector, dim);
        for (const auto kernel : kernels) {
            const float actual = kernel(query, vector, dim);
            REQUIRE(std::abs(actual - expected) <= 1e-4F * std::max(1.0F, expected));
        }
    }
}
