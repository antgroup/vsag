// Copyright 2026-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include <iomanip>
#include <sstream>

#include "half_dense_test_fixture.h"

namespace {
struct MatrixRows {
    int64_t dim;
    std::vector<uint16_t> half;
    std::vector<float> fp32;
    std::vector<int64_t> ids;

    MatrixRows(int64_t dimension, int64_t count, int64_t first, bool bf16)
        : dim(dimension), half(count * dim), fp32(count * dim), ids(count) {
        for (int64_t i = 0; i < count; ++i) {
            ids[i] = first + i;
            for (int64_t j = 0; j < dim; ++j) {
                // Independent bit construction, not the production encoder.
                const auto hash = static_cast<uint64_t>((first + i) * 271 + j * 173);
                auto bits =
                    static_cast<uint16_t>((bf16 ? 0x3E80 : 0x3400) + hash % (bf16 ? 384 : 3072));
                if ((hash / 7) % 2 != 0) {
                    bits |= 0x8000U;
                }
                half[i * dim + j] = bits;
                fp32[i * dim + j] = half_dense_test::Decode(bits, bf16);
            }
        }
    }

    vsag::DatasetPtr
    Data(bool native, int64_t row = 0, int64_t count = -1) {
        if (count < 0) {
            count = static_cast<int64_t>(ids.size()) - row;
        }
        auto data = vsag::Dataset::Make()
                        ->Dim(dim)
                        ->NumElements(count)
                        ->Ids(ids.data() + row)
                        ->Owner(false);
        if (native) {
            data->Float16Vectors(half.data() + row * dim);
        } else {
            data->Float32Vectors(fp32.data() + row * dim);
        }
        return data;
    }
};

vsag::InnerIndexPtr
MatrixIndex(bool lazy,
            bool native,
            bool bf16,
            vsag::MetricType metric,
            int64_t dim,
            const std::string& storage) {
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.data_type_ =
        native ? (bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16)
               : vsag::DataTypes::DATA_TYPE_FLOAT;
    common.metric_ = metric;
    static const auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.allocator_ = allocator;
    auto graph = vsag::JsonType::Parse(
        R"({"store_raw_vector":true,"max_degree":16,"ef_construction":64,"build_thread_count":1})");
    graph["base_quantization_type"].SetString(storage);
    vsag::InnerIndexPtr result;
    if (lazy) {
        auto json = vsag::JsonType::Parse(R"({"transition_threshold":4})");
        json["hgraph"].SetJson(graph);
        auto param = vsag::LazyHGraph::CheckAndMappingExternalParam(json, common);
        result = std::make_shared<vsag::LazyHGraph>(param, common);
    } else {
        auto flat = vsag::JsonType::Parse(R"({"store_raw_vector":true})");
        flat["base_quantization_type"].SetString(storage);
        auto param = vsag::BruteForce::CheckAndMappingExternalParam(flat, common);
        result = std::make_shared<vsag::BruteForce>(param, common);
    }
    result->InitFeatures();
    return result;
}

void
MatrixCompare(const vsag::DatasetPtr& actual, const vsag::DatasetPtr& expected) {
    REQUIRE(actual->GetDim() == expected->GetDim());
    std::vector<std::pair<int64_t, float>> lhs;
    std::vector<std::pair<int64_t, float>> rhs;
    for (int64_t j = 0; j < actual->GetDim(); ++j) {
        lhs.emplace_back(actual->GetIds()[j], actual->GetDistances()[j]);
        rhs.emplace_back(expected->GetIds()[j], expected->GetDistances()[j]);
    }
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    for (uint64_t j = 0; j < lhs.size(); ++j) {
        INFO("compared id=" << lhs[j].first << " reference id=" << rhs[j].first
                            << " actual distance=" << std::setprecision(10) << lhs[j].second
                            << " reference distance=" << rhs[j].second);
        REQUIRE(lhs[j].first == rhs[j].first);
        REQUIRE(std::isfinite(lhs[j].second));
        // Identical independently half-rounded FP32 input and deterministic model.
        // A scale-aware numerical epsilon is not a recall allowance.
        REQUIRE(std::abs(lhs[j].second - rhs[j].second) <=
                1e-5F * std::max(1.0F, std::abs(rhs[j].second)));
    }
}

void
MatrixCheck(const vsag::InnerIndexPtr& actual,
            const vsag::InnerIndexPtr& reference,
            MatrixRows& rows) {
    const std::string search = R"({"hgraph":{"ef_search":128}})";
    for (int64_t row : {int64_t{0}, int64_t{1}}) {
        INFO("query row=" << row);
        {
            INFO("search=KNN");
            MatrixCompare(
                actual->KnnSearch(rows.Data(true, row, 1), 3, search, vsag::FilterPtr{}),
                reference->KnnSearch(rows.Data(false, row, 1), 3, search, vsag::FilterPtr{}));
        }
        {
            INFO("search=Range");
            MatrixCompare(
                actual->RangeSearch(rows.Data(true, row, 1), 10000.0F, search, vsag::FilterPtr{}),
                reference->RangeSearch(
                    rows.Data(false, row, 1), 10000.0F, search, vsag::FilterPtr{}));
        }
    }
}

void
MatrixExercise(
    bool lazy, bool bf16, vsag::MetricType metric, int64_t dim, const std::string& storage) {
    CAPTURE(lazy, bf16, metric, dim, storage);
    auto make = [&](bool native) { return MatrixIndex(lazy, native, bf16, metric, dim, storage); };
    auto actual = make(true);
    auto reference = make(false);
    MatrixRows initial(dim, 3, 100, bf16);
    INFO("phase: initial Build / flat reload");
    REQUIRE(actual->Build(initial.Data(true)).empty());
    REQUIRE(reference->Build(initial.Data(false)).empty());
    MatrixCheck(actual, reference, initial);
    auto binary = actual->Serialize();
    actual = make(true);
    actual->Deserialize(binary);
    MatrixCheck(actual, reference, initial);
    std::ostringstream replay_diagnostic;
    if (lazy and metric == vsag::MetricType::METRIC_TYPE_COSINE) {
        auto actual_old = actual->GetDataByIds(initial.ids.data(), 3);
        auto reference_old = reference->GetDataByIds(initial.ids.data(), 3);
        replay_diagnostic << std::setprecision(10) << "old-three pre-transition public export:";
        for (int64_t i = 0; i < 3; ++i) {
            replay_diagnostic << " id=" << initial.ids[i];
            for (int64_t j = 0; j < dim; ++j) {
                const auto k = i * dim + j;
                uint32_t actual_bits;
                uint32_t reference_bits;
                std::memcpy(&actual_bits, actual_old->GetFloat32Vectors() + k, sizeof(actual_bits));
                std::memcpy(&reference_bits,
                            reference_old->GetFloat32Vectors() + k,
                            sizeof(reference_bits));
                replay_diagnostic << " [rawhalf=" << initial.half[k]
                                  << " originalFP32=" << initial.fp32[k]
                                  << " nativeExport=" << actual_old->GetFloat32Vectors()[k]
                                  << " nativeExportFP32bits=" << actual_bits
                                  << " referenceExport=" << reference_old->GetFloat32Vectors()[k]
                                  << " referenceExportFP32bits=" << reference_bits << "]";
            }
        }
    }
    INFO(replay_diagnostic.str());
    MatrixRows added(dim, 37, 200, bf16);
    INFO("phase: Add batch / transition");
    REQUIRE(actual->Add(added.Data(true)).empty());
    REQUIRE(reference->Add(added.Data(false)).empty());
    MatrixCheck(actual, reference, added);
    if (lazy) {
        REQUIRE(std::dynamic_pointer_cast<vsag::LazyHGraph>(actual)->GetPhase() ==
                vsag::LazyHGraph::Phase::GRAPH);
    }
    MatrixRows replacement(dim, 2, 501, bf16);
    INFO("phase: Update / public export / reload");
    REQUIRE(actual->UpdateVector(100, replacement.Data(true, 0, 1), true));
    REQUIRE(reference->UpdateVector(100, replacement.Data(false, 0, 1), true));
    MatrixCheck(actual, reference, replacement);
    const int64_t id = 200;
    auto lhs = actual->GetDataByIds(&id, 1);
    auto rhs = reference->GetDataByIds(&id, 1);
    REQUIRE(lhs->GetFloat32Vectors() != nullptr);
    REQUIRE(rhs->GetFloat32Vectors() != nullptr);
    if (lazy and metric == vsag::MetricType::METRIC_TYPE_COSINE) {
        // The graph preserves what it received from the flat phase. The half flat
        // index replays the original row; the FLOAT flat index replays normalized codes.
        float norm_squared = 0.0F;
        for (int64_t j = 0; j < dim; ++j) {
            norm_squared += added.fp32[j] * added.fp32[j];
        }
        const float norm = std::sqrt(norm_squared);
        REQUIRE(norm > 0.0F);
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(lhs->GetFloat32Vectors()[j] == added.fp32[j]);
            REQUIRE(std::abs(rhs->GetFloat32Vectors()[j] - added.fp32[j] / norm) <= 1e-5F);
        }
    } else {
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(std::isfinite(lhs->GetFloat32Vectors()[j]));
            // Public native adapter decode rounds its reconstructed row back to
            // the external half dtype; quantized FP32 reference does not.
            REQUIRE(std::abs(lhs->GetFloat32Vectors()[j] - rhs->GetFloat32Vectors()[j]) <=
                    0.008F * std::max(1.0F, std::abs(rhs->GetFloat32Vectors()[j])));
        }
    }
    binary = actual->Serialize();
    actual = make(true);
    actual->Deserialize(binary);
    MatrixCheck(actual, reference, replacement);
    REQUIRE(actual->UpdateVector(200, replacement.Data(true, 1, 1), true));
    REQUIRE(reference->UpdateVector(200, replacement.Data(false, 1, 1), true));
    MatrixCheck(actual, reference, replacement);
}
}  // namespace

TEST_CASE("Half cosine flat mutation preserves raw replay", "[half_cosine_flat]") {
    for (bool bf16 : {false, true}) {
        for (int64_t dim : {7, 33}) {
            CAPTURE(bf16, dim);
            auto make = [&](bool native) {
                return MatrixIndex(
                    true, native, bf16, vsag::MetricType::METRIC_TYPE_COSINE, dim, "fp32");
            };
            auto actual = make(true);
            auto reference = make(false);
            MatrixRows initial(dim, 3, 100, bf16);
            REQUIRE(actual->Build(initial.Data(true)).empty());
            REQUIRE(reference->Build(initial.Data(false)).empty());
            MatrixCheck(actual, reference, initial);
            MatrixRows replacement(dim, 2, 500, bf16);
            REQUIRE(actual->UpdateVector(100, replacement.Data(true, 0, 1), true));
            REQUIRE(reference->UpdateVector(100, replacement.Data(false, 0, 1), true));
            MatrixCheck(actual, reference, replacement);
            actual->Remove({101}, vsag::RemoveMode::FORCE_REMOVE);
            reference->Remove({101}, vsag::RemoveMode::FORCE_REMOVE);
            REQUIRE_FALSE(actual->CheckIdExist(101));
            auto binary = actual->Serialize();
            actual = make(true);
            actual->Deserialize(binary);
            MatrixCheck(actual, reference, replacement);
            MatrixRows added(dim, 37, 200, bf16);
            REQUIRE(actual->Add(added.Data(true)).empty());
            REQUIRE(reference->Add(added.Data(false)).empty());
            MatrixCheck(actual, reference, added);
            binary = actual->Serialize();
            actual = make(true);
            actual->Deserialize(binary);
            REQUIRE(actual->UpdateVector(200, replacement.Data(true, 0, 1), true));
            REQUIRE(reference->UpdateVector(200, replacement.Data(false, 0, 1), true));
            MatrixCheck(actual, reference, replacement);
        }
    }
}

TEST_CASE("Odd dense dimensions preserve typed quantized lifecycle",
          "[ut][half_input][half_matrix]") {
    for (bool lazy : {false, true}) {
        const std::vector<std::string> storages =
            lazy ? std::vector<std::string>{"fp32", "sq8"}
                 : std::vector<std::string>{"fp32", "sq8", "sq4", "fp16", "bf16"};
        for (const auto& storage : storages) {
            for (int64_t dim : {7, 33}) {
                for (bool bf16 : {false, true}) {
                    for (auto metric : {vsag::MetricType::METRIC_TYPE_L2SQR,
                                        vsag::MetricType::METRIC_TYPE_IP,
                                        vsag::MetricType::METRIC_TYPE_COSINE}) {
                        MatrixExercise(lazy, bf16, metric, dim, storage);
                    }
                }
            }
        }
    }
}
