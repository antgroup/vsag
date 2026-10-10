// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "vsag/lite/index.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iterator>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <unordered_map>
#include <vector>

#include "fp16_distance.h"
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

TEST_CASE("Lite per-query budget preserves defaults, filters and mutation policy", "[lite]") {
    auto reference = Index::Create(2);
    REQUIRE(reference);
    for (int64_t id = -32; id < 32; ++id) {
        const int64_t row = (id + 32) / 11;
        const float vector[]{static_cast<float>((id + 32) % 11 - 5), static_cast<float>(row - 3)};
        REQUIRE((*reference)->Add(id, vector, 2));
    }
    std::stringstream base;
    REQUIRE((*reference)->Save(base));
    const float query[]{4.25F, -3.75F};
    const vsag::lite::IdFilter even = [](int64_t id) { return id % 2 == 0; };
    const auto compare = [](const auto& left, const auto& right) {
        REQUIRE(left);
        REQUIRE(right);
        REQUIRE(left->size() == right->size());
        for (uint64_t i = 0; i < left->size(); ++i) {
            REQUIRE((*left)[i].id == (*right)[i].id);
            REQUIRE((*left)[i].distance == (*right)[i].distance);
        }
    };
    for (const int mode : {0, 1, 2}) {
        std::stringstream input(base.str());
        auto created = Index::Load(input);
        REQUIRE(created);
        if (mode != 0) {
            const auto storage =
                mode == 1 ? vsag::lite::VectorStorage::FP32 : vsag::lite::VectorStorage::FP16;
            REQUIRE((*created)->BuildGraph(storage, 4, 8));
        }
        std::stringstream before;
        REQUIRE((*created)->Save(before));
        compare((*created)->Search(query, 2, 10), (*created)->SearchWithOptions(query, 2, 10, {}));
        compare((*created)->Search(query, 2, 10, {}),
                (*created)->SearchWithOptions(query, 2, 10, {}, {}));
        compare((*created)->Search(query, 2, 10, even),
                (*created)->SearchWithOptions(query, 2, 10, {}, even));
        compare((*reference)->Search(query, 2, 10),
                (*created)->SearchWithOptions(query, 2, 10, {UINT64_MAX}));
        compare((*reference)->Search(query, 2, 10, even),
                (*created)->SearchWithOptions(query, 2, 10, {UINT64_MAX}, even));
        REQUIRE((*created)->SearchWithOptions(query, 2, 10, {1})->size() == 10);
        REQUIRE((*created)->SearchWithOptions(query, 2, 0, {UINT64_MAX})->empty());
        REQUIRE_FALSE((*created)->SearchWithOptions(nullptr, 2, 10, {64}));
        REQUIRE_FALSE((*created)->SearchWithOptions(query, 1, 10, {64}));
        const vsag::lite::IdFilter reject = [](int64_t) { return false; };
        REQUIRE((*created)->SearchWithOptions(query, 2, 10, {64}, reject)->empty());
        std::stringstream after;
        REQUIRE((*created)->Save(after));
        REQUIRE(after.str() == before.str());
        std::stringstream twin_input(before.str());
        auto twin = Index::Load(twin_input);
        REQUIRE(twin);
        const float changed[]{20, 21};
        for (auto* index : {created->get(), twin->get()}) {
            REQUIRE(index->Update(-32, changed, 2));
            REQUIRE(index->Remove(31));
            REQUIRE(index->Add(100, changed, 2));
        }
        std::stringstream mutated;
        std::stringstream twin_mutated;
        REQUIRE((*created)->Save(mutated));
        REQUIRE((*twin)->Save(twin_mutated));
        REQUIRE(mutated.str() == twin_mutated.str());
    }
    auto empty = Index::Create(2);
    REQUIRE(empty);
    REQUIRE((*empty)->BuildGraph(4, 8));
    REQUIRE((*empty)->SearchWithOptions(query, 2, 10, {UINT64_MAX})->empty());
}

TEST_CASE("Lite identical graph updates preserve snapshots after validation", "[lite]") {
    for (const auto storage : {vsag::lite::VectorStorage::FP32, vsag::lite::VectorStorage::FP16}) {
        auto created = vsag::lite::Index::Create(2);
        REQUIRE(created);
        auto& index = **created;
        const float origin[]{0.0F, 0.0F};
        const float a[]{1.0F, 0.0F};
        const float b[]{0.0F, 1.0F};
        const float c[]{2.0F, 1.0F};
        REQUIRE(index.Add(1, origin, 2));
        REQUIRE(index.Add(2, a, 2));
        REQUIRE(index.Add(3, b, 2));
        REQUIRE(index.Add(4, c, 2));
        REQUIRE(index.BuildGraph(storage, 2, 8));
        auto snapshot = [&index]() {
            std::stringstream bytes;
            REQUIRE(index.Save(bytes));
            return bytes.str();
        };
        const auto before = snapshot();
        REQUIRE(index.Update(1, origin, 2));
        REQUIRE(static_cast<bool>(snapshot() == before));
        REQUIRE_FALSE(index.Update(99, origin, 2));
        REQUIRE_FALSE(index.Update(1, nullptr, 2));
        REQUIRE_FALSE(index.Update(1, origin, 1));
        const float invalid[]{std::numeric_limits<float>::infinity(), 0.0F};
        REQUIRE_FALSE(index.Update(1, invalid, 2));
        REQUIRE(static_cast<bool>(snapshot() == before));
        if (storage == vsag::lite::VectorStorage::FP16) {
            const auto encoded_before = snapshot();
            const float equivalent[]{1.0001F, 0.0F};
            REQUIRE(index.Update(2, equivalent, 2));
            REQUIRE(static_cast<bool>(snapshot() == encoded_before));
            const float overflow[]{70000.0F, 0.0F};
            REQUIRE_FALSE(index.Update(2, overflow, 2));
            REQUIRE(static_cast<bool>(snapshot() == encoded_before));
        } else {
            const float signed_zero[]{-0.0F, 0.0F};
            REQUIRE(index.Update(1, signed_zero, 2));
            REQUIRE(static_cast<bool>(snapshot() != before));
        }
        const float changed[]{0.25F, 0.0F};
        const auto old = snapshot();
        REQUIRE(index.Update(1, changed, 2));
        REQUIRE(static_cast<bool>(snapshot() != old));
        const auto found = index.Search(changed, 2, 1);
        REQUIRE(found);
        REQUIRE(found->front().id == 1);
        REQUIRE(found->front().distance == 0.0F);
    }
}

TEST_CASE("Lite filter allocation errors preserve the index", "[lite]") {
    for (const bool graph : {false, true}) {
        auto created = Index::Create(1);
        REQUIRE(created);
        const float vector = 1.0F;
        REQUIRE((*created)->Add(17, &vector, 1));
        if (graph) {
            REQUIRE((*created)->BuildGraph(2, 8));
        }
        const vsag::lite::IdFilter allocation_failure = [](int64_t) -> bool {
            throw std::bad_alloc();
        };
        const vsag::lite::IdFilter capacity_failure = [](int64_t) -> bool {
            throw std::length_error("filter capacity failure");
        };
        REQUIRE_FALSE((*created)->Search(&vector, 1, 1, allocation_failure));
        REQUIRE_FALSE((*created)->Search(&vector, 1, 1, capacity_failure));
        REQUIRE((*created)->Size() == 1);
        REQUIRE((*created)->Search(&vector, 1, 1)->front().id == 17);
    }
}

TEST_CASE("Lite FP16 graph handles tiny values and rounding boundaries", "[lite]") {
    auto created = Index::Create(8);
    REQUIRE(created);
    const float vector[]{0.0F,
                         -0.0F,
                         std::numeric_limits<float>::denorm_min(),
                         std::ldexp(1.0F, -30),
                         std::ldexp(1.5F, -24),
                         1.00146484375F,
                         std::nextafter(2.0F, 0.0F),
                         65504.0F};
    REQUIRE((*created)->Add(17, vector, 8));
    REQUIRE((*created)->BuildGraph(vsag::lite::VectorStorage::FP16, 2, 8));
    std::stringstream saved;
    REQUIRE((*created)->Save(saved));
    auto loaded = Index::Load(saved);
    REQUIRE(loaded);
    REQUIRE((*loaded)->Search(vector, 8, 1)->front().id == 17);
    auto overflow = std::vector<float>(std::begin(vector), std::end(vector));
    overflow[0] = 65520.0F;
    REQUIRE_FALSE((*loaded)->Add(18, overflow.data(), 8));
    overflow[0] = 70000.0F;
    REQUIRE_FALSE((*loaded)->Update(17, overflow.data(), 8));
    REQUIRE((*loaded)->Size() == 1);
    REQUIRE((*loaded)->Search(vector, 8, 1)->front().id == 17);
}

namespace {
class FailingPositionBuffer : public std::streambuf {
public:
    explicit FailingPositionBuffer(bool allocation) : allocation_(allocation) {
    }

protected:
    pos_type
    seekoff([[maybe_unused]] off_type offset,
            [[maybe_unused]] std::ios_base::seekdir direction,
            [[maybe_unused]] std::ios_base::openmode mode) override {
        if (allocation_) {
            throw std::bad_alloc();
        }
        throw std::length_error("stream position capacity failure");
    }

private:
    bool allocation_;
};
}  // namespace

TEST_CASE("Lite load translates stream allocation and capacity failures", "[lite]") {
    for (const bool allocation : {false, true}) {
        FailingPositionBuffer buffer(allocation);
        std::istream input(&buffer);
        input.exceptions(std::ios::badbit);
        REQUIRE_FALSE(Index::Load(input));
    }
}

#ifdef VSAG_LITE_HAS_X86_SIMD
TEST_CASE("Lite FP16 dispatch requires complete feature combinations", "[lite]") {
    using namespace vsag::lite::detail;
    REQUIRE(select_fp16_distance_for(false, false, false, false, false) == generic_fp16_distance);
    REQUIRE(select_fp16_distance_for(true, true, true, true, false) == generic_fp16_distance);
    REQUIRE(select_fp16_distance_for(false, true, false, false, true) == generic_fp16_distance);
    REQUIRE(select_fp16_distance_for(true, false, false, false, true) == avx_fp16_distance);
    REQUIRE(select_fp16_distance_for(true, true, false, false, true) == avx_fp16_distance);
    REQUIRE(select_fp16_distance_for(true, true, false, true, true) == avx2_fp16_distance);
    REQUIRE(select_fp16_distance_for(true, true, true, true, true) == avx512_fp16_distance);
}
#endif

TEST_CASE("Lite FP16 removal compacts vectors and reloads with query options", "[lite]") {
    auto created = Index::Create(2);
    REQUIRE(created);
    const float first[]{0.0F, 0.0F};
    const float middle[]{1.0F, 0.0F};
    const float last[]{2.0F, 1.0F};
    REQUIRE((*created)->Add(-10, first, 2));
    REQUIRE((*created)->Add(20, middle, 2));
    REQUIRE((*created)->Add(30, last, 2));
    REQUIRE((*created)->BuildGraph(vsag::lite::VectorStorage::FP16, 2, 8));
    REQUIRE((*created)->Remove(20));
    REQUIRE((*created)->Size() == 2);
    std::stringstream bytes;
    REQUIRE((*created)->Save(bytes));
    auto loaded = Index::Load(bytes);
    REQUIRE(loaded);
    const auto result = (*loaded)->SearchWithOptions(last, 2, 3, {UINT64_MAX});
    REQUIRE(result);
    REQUIRE(result->size() == 2);
    REQUIRE(result->front().id == 30);
    REQUIRE(result->front().distance == 0.0F);
    REQUIRE(std::none_of(result->begin(), result->end(), [](const auto& n) { return n.id == 20; }));
    const vsag::lite::IdFilter allocation_failure = [](int64_t) -> bool { throw std::bad_alloc(); };
    const auto failed = (*loaded)->SearchWithOptions(last, 2, 1, {64}, allocation_failure);
    REQUIRE_FALSE(failed);
    REQUIRE(failed.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);
    REQUIRE((*loaded)->SearchWithOptions(last, 2, 1, {})->front().id == 30);
}
