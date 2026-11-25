// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "algorithm/bruteforce/bruteforce.h"
#include "algorithm/lazy_hgraph/lazy_hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

namespace half_dense_test {
constexpr int64_t DIM = 8;

// Independent IEEE decoding: references never use the production half adapter.
inline float
Decode(uint16_t bits, bool bf16) {
    if (bf16) {
        uint32_t word = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &word, sizeof(value));
        return value;
    }
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
                                          : std::ldexp(1.0F + fraction / 1024.0F, exponent - 15);
    return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
}

struct Rows {
    std::vector<uint16_t> half;
    std::vector<float> fp32;
    std::vector<int64_t> ids;

    Rows(int64_t count, int64_t first, bool bf16)
        : half(count * DIM), fp32(count * DIM), ids(count) {
        for (int64_t i = 0; i < count; ++i) {
            ids[i] = first + i;
            for (int64_t j = 0; j < DIM; ++j) {
                const auto mantissa = static_cast<uint16_t>(((first + i) * 37 + j * 71) % 127);
                auto bits = static_cast<uint16_t>((bf16 ? 0x3F80 : 0x3C00) + mantissa);
                if ((i + j) % 3 == 0) {
                    bits |= 0x8000U;
                }
                half[i * DIM + j] = bits;
                fp32[i * DIM + j] = Decode(bits, bf16);
            }
        }
    }

    vsag::DatasetPtr
    Data(bool native, int64_t row = 0, int64_t count = -1) {
        if (count < 0) {
            count = static_cast<int64_t>(ids.size()) - row;
        }
        auto result = vsag::Dataset::Make()
                          ->Dim(DIM)
                          ->NumElements(count)
                          ->Ids(ids.data() + row)
                          ->Owner(false);
        if (native) {
            result->Float16Vectors(half.data() + row * DIM);
        } else {
            result->Float32Vectors(fp32.data() + row * DIM);
        }
        return result;
    }
};

inline vsag::InnerIndexPtr
Make(bool lazy, bool native, bool bf16, vsag::MetricType metric, uint64_t threshold) {
    vsag::IndexCommonParam common;
    common.dim_ = DIM;
    common.data_type_ =
        native ? (bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16)
               : vsag::DataTypes::DATA_TYPE_FLOAT;
    common.metric_ = metric;
    // The index retains a raw allocator pointer; keep its shared owner alive.
    static const auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.allocator_ = allocator;
    vsag::InnerIndexPtr index;
    if (lazy) {
        auto json = vsag::JsonType::Parse(
            R"({"transition_threshold":4,"hgraph":{"store_raw_vector":true,"base_quantization_type":"fp32","max_degree":16,"ef_construction":64,"build_thread_count":1}})");
        json["transition_threshold"].SetUint64(threshold);
        auto param = vsag::LazyHGraph::CheckAndMappingExternalParam(json, common);
        index = std::make_shared<vsag::LazyHGraph>(param, common);
    } else {
        auto json = vsag::JsonType::Parse(R"({"base_quantization_type":"fp32"})");
        auto param = vsag::BruteForce::CheckAndMappingExternalParam(json, common);
        index = std::make_shared<vsag::BruteForce>(param, common);
    }
    index->InitFeatures();
    return index;
}

inline void
Compare(const vsag::DatasetPtr& actual, const vsag::DatasetPtr& expected) {
    REQUIRE(actual->GetDim() == expected->GetDim());
    std::vector<std::pair<int64_t, float>> lhs;
    std::vector<std::pair<int64_t, float>> rhs;
    for (int64_t i = 0; i < actual->GetDim(); ++i) {
        lhs.emplace_back(actual->GetIds()[i], actual->GetDistances()[i]);
        rhs.emplace_back(expected->GetIds()[i], expected->GetDistances()[i]);
    }
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    for (uint64_t i = 0; i < lhs.size(); ++i) {
        REQUIRE(lhs[i].first == rhs[i].first);
        REQUIRE(std::abs(lhs[i].second - rhs[i].second) < 0.025F);
    }
}

inline void
Check(const vsag::InnerIndexPtr& actual, const vsag::InnerIndexPtr& reference, Rows& query) {
    const std::string search = R"({"hgraph":{"ef_search":128}})";
    Compare(actual->KnnSearch(query.Data(true, 0, 1), 1, search, vsag::FilterPtr{}),
            reference->KnnSearch(query.Data(false, 0, 1), 1, search, vsag::FilterPtr{}));
    Compare(actual->RangeSearch(query.Data(true, 0, 1), 100.0F, search, vsag::FilterPtr{}),
            reference->RangeSearch(query.Data(false, 0, 1), 100.0F, search, vsag::FilterPtr{}));
}

inline void
Exercise(bool lazy, bool bf16, vsag::MetricType metric, uint64_t threshold) {
    auto actual = Make(lazy, true, bf16, metric, threshold);
    auto reference = Make(lazy, false, bf16, metric, threshold);
    Rows initial(3, 100, bf16);
    REQUIRE(actual->Build(initial.Data(true)).empty());
    REQUIRE(reference->Build(initial.Data(false)).empty());
    Check(actual, reference, initial);
    // Round-trip the flat phase before the threshold transition.
    auto binary = actual->Serialize();
    actual = Make(lazy, true, bf16, metric, threshold);
    actual->Deserialize(binary);
    Check(actual, reference, initial);
    Rows added(9, 200, bf16);
    REQUIRE(actual->Add(added.Data(true)).empty());
    REQUIRE(reference->Add(added.Data(false)).empty());
    Check(actual, reference, added);
    if (lazy) {
        const auto typed = std::dynamic_pointer_cast<vsag::LazyHGraph>(actual);
        REQUIRE(typed->GetPhase() ==
                (threshold <= 12 ? vsag::LazyHGraph::Phase::GRAPH : vsag::LazyHGraph::Phase::FLAT));
    }
    Rows replacement(1, 499, bf16);
    REQUIRE(actual->UpdateVector(100, replacement.Data(true), true));
    REQUIRE(reference->UpdateVector(100, replacement.Data(false), true));
    Check(actual, reference, replacement);
    // Public exports stay FP32. During migration the half cosine flat index
    // replays original values; the FLOAT flat index replays normalized codes.
    // Graph store_raw_vector preserves the values actually received by Build.
    int64_t id = 200;
    auto check_export = [&](const vsag::InnerIndexPtr& index, bool native_input) {
        auto exported = index->GetDataByIds(&id, 1);
        REQUIRE(exported->GetFloat32Vectors() != nullptr);
        const bool original_replay = lazy and native_input;
        const float norm =
            metric == vsag::MetricType::METRIC_TYPE_COSINE and not original_replay
                ? std::sqrt(std::inner_product(
                      added.fp32.begin(), added.fp32.begin() + DIM, added.fp32.begin(), 0.0F))
                : 1.0F;
        for (int64_t j = 0; j < DIM; ++j) {
            REQUIRE(std::abs(exported->GetFloat32Vectors()[j] - added.fp32[j] / norm) < 0.015F);
        }
    };
    check_export(actual, true);
    check_export(reference, false);
    binary = actual->Serialize();
    actual = Make(lazy, true, bf16, metric, threshold);
    actual->Deserialize(binary);
    check_export(actual, true);
    Check(actual, reference, replacement);
    Rows final_row(1, 700, bf16);
    REQUIRE(actual->Add(final_row.Data(true)).empty());
    REQUIRE(reference->Add(final_row.Data(false)).empty());
    REQUIRE(actual->UpdateVector(200, replacement.Data(true), true));
    REQUIRE(reference->UpdateVector(200, replacement.Data(false), true));
    Check(actual, reference, final_row);
    auto missing = vsag::Dataset::Make()->Dim(DIM)->NumElements(1)->Ids(&id)->Owner(false);
    REQUIRE_THROWS(actual->Add(missing));
    REQUIRE_THROWS(actual->KnnSearch(missing, 1, "{}", vsag::FilterPtr{}));
    REQUIRE_THROWS(actual->RangeSearch(missing, 1.0F, "{}", vsag::FilterPtr{}));
    REQUIRE_THROWS(actual->UpdateVector(200, missing, true));
    REQUIRE_THROWS(reference->UpdateVector(200, missing, true));
    REQUIRE_THROWS(actual->UpdateVector(200, vsag::DatasetPtr{}, true));
    REQUIRE_THROWS(reference->UpdateVector(200, vsag::DatasetPtr{}, true));
    auto wrong_dim_native = replacement.Data(true);
    wrong_dim_native->Dim(DIM + 1);
    auto wrong_dim_fp32 = replacement.Data(false);
    wrong_dim_fp32->Dim(DIM + 1);
    REQUIRE_THROWS(actual->UpdateVector(200, wrong_dim_native, true));
    REQUIRE_THROWS(reference->UpdateVector(200, wrong_dim_fp32, true));
    if (lazy and threshold <= 12) {
        auto wrong_count_native = replacement.Data(true);
        wrong_count_native->NumElements(2);
        auto wrong_count_fp32 = replacement.Data(false);
        wrong_count_fp32->NumElements(2);
        REQUIRE_THROWS(actual->UpdateVector(200, wrong_count_native, true));
        REQUIRE_THROWS(reference->UpdateVector(200, wrong_count_fp32, true));
    }
}
}  // namespace half_dense_test
