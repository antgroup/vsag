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
QueryExercise(
    bool lazy, bool bf16, vsag::MetricType metric, int64_t dim, const std::string& storage) {
    CAPTURE(lazy, bf16, metric, dim, storage);
    auto make = [&](bool native) { return MatrixIndex(lazy, native, bf16, metric, dim, storage); };
    auto actual = make(true);
    auto reference = make(false);
    MatrixRows initial(dim, 3, 100, bf16);
    {
        INFO("phase: initial Build");
        REQUIRE(actual->Build(initial.Data(true)).empty());
        REQUIRE(reference->Build(initial.Data(false)).empty());
        MatrixCheck(actual, reference, initial);
    }
    auto binary = actual->Serialize();
    actual = make(true);
    actual->Deserialize(binary);
    {
        INFO("phase: flat reload");
        MatrixCheck(actual, reference, initial);
    }
    MatrixRows added(dim, 37, 200, bf16);
    {
        INFO("phase: Add / graph transition");
        REQUIRE(actual->Add(added.Data(true)).empty());
        REQUIRE(reference->Add(added.Data(false)).empty());
        if (lazy)
            REQUIRE(std::dynamic_pointer_cast<vsag::LazyHGraph>(actual)->GetPhase() ==
                    vsag::LazyHGraph::Phase::GRAPH);
        MatrixCheck(actual, reference, added);
    }
    MatrixRows replacement(dim, 2, 501, bf16);
    {
        INFO("phase: first Update");
        REQUIRE(actual->UpdateVector(100, replacement.Data(true, 0, 1), true));
        REQUIRE(reference->UpdateVector(100, replacement.Data(false, 0, 1), true));
        MatrixCheck(actual, reference, replacement);
    }
    binary = actual->Serialize();
    actual = make(true);
    actual->Deserialize(binary);
    {
        INFO("phase: graph reload");
        MatrixCheck(actual, reference, replacement);
    }
    {
        INFO("phase: second Update after graph reload");
        REQUIRE(actual->UpdateVector(200, replacement.Data(true, 1, 1), true));
        REQUIRE(reference->UpdateVector(200, replacement.Data(false, 1, 1), true));
        MatrixCheck(actual, reference, replacement);
    }
}

void
RawInputExercise(bool bf16, int64_t dim) {
    // Separate diagnostic, not a replacement for historical FP32 export parity.
    auto make = [&] {
        return MatrixIndex(true, true, bf16, vsag::MetricType::METRIC_TYPE_COSINE, dim, "fp32");
    };
    auto index = make();
    MatrixRows initial(dim, 3, 100, bf16);
    REQUIRE(index->Build(initial.Data(true)).empty());
    auto check = [&](int64_t id, const float* original) {
        auto data = index->GetDataByIds(&id, 1);
        REQUIRE(data->GetFloat32Vectors() != nullptr);
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(data->GetFloat32Vectors()[j] == original[j]);
        }
    };
    check(100, initial.fp32.data());
    auto binary = index->Serialize();
    index = make();
    index->Deserialize(binary);
    check(100, initial.fp32.data());
    MatrixRows added(dim, 37, 200, bf16);
    REQUIRE(index->Add(added.Data(true)).empty());
    check(100, initial.fp32.data());
    check(200, added.fp32.data());
    MatrixRows replacement(dim, 2, 501, bf16);
    REQUIRE(index->UpdateVector(100, replacement.Data(true, 0, 1), true));
    check(100, replacement.fp32.data());
    binary = index->Serialize();
    index = make();
    index->Deserialize(binary);
    check(100, replacement.fp32.data());
    check(200, added.fp32.data());
    REQUIRE(index->UpdateVector(200, replacement.Data(true, 1, 1), true));
    check(200, replacement.fp32.data() + dim);
}
}  // namespace

TEST_CASE("Half query BF FP16 D7 L2 fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "fp32");
}

TEST_CASE("Half query BF FP16 D7 IP fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 7, "fp32");
}

TEST_CASE("Half query BF FP16 D7 cosine fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "fp32");
}

TEST_CASE("Half query BF BF16 D7 L2 fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "fp32");
}

TEST_CASE("Half query BF BF16 D7 IP fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 7, "fp32");
}

TEST_CASE("Half query BF BF16 D7 cosine fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "fp32");
}

TEST_CASE("Half query BF FP16 D33 L2 fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "fp32");
}

TEST_CASE("Half query BF FP16 D33 IP fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 33, "fp32");
}

TEST_CASE("Half query BF FP16 D33 cosine fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "fp32");
}

TEST_CASE("Half query BF BF16 D33 L2 fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "fp32");
}

TEST_CASE("Half query BF BF16 D33 IP fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 33, "fp32");
}

TEST_CASE("Half query BF BF16 D33 cosine fp32", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "fp32");
}

TEST_CASE("Half query BF FP16 D7 L2 sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "sq8");
}

TEST_CASE("Half query BF FP16 D7 IP sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 7, "sq8");
}

TEST_CASE("Half query BF FP16 D7 cosine sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "sq8");
}

TEST_CASE("Half query BF BF16 D7 L2 sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "sq8");
}

TEST_CASE("Half query BF BF16 D7 IP sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 7, "sq8");
}

TEST_CASE("Half query BF BF16 D7 cosine sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "sq8");
}

TEST_CASE("Half query BF FP16 D33 L2 sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "sq8");
}

TEST_CASE("Half query BF FP16 D33 IP sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 33, "sq8");
}

TEST_CASE("Half query BF FP16 D33 cosine sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "sq8");
}

TEST_CASE("Half query BF BF16 D33 L2 sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "sq8");
}

TEST_CASE("Half query BF BF16 D33 IP sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 33, "sq8");
}

TEST_CASE("Half query BF BF16 D33 cosine sq8", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "sq8");
}

TEST_CASE("Half query BF FP16 D7 L2 sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "sq4");
}

TEST_CASE("Half query BF FP16 D7 IP sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 7, "sq4");
}

TEST_CASE("Half query BF FP16 D7 cosine sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "sq4");
}

TEST_CASE("Half query BF BF16 D7 L2 sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "sq4");
}

TEST_CASE("Half query BF BF16 D7 IP sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 7, "sq4");
}

TEST_CASE("Half query BF BF16 D7 cosine sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "sq4");
}

TEST_CASE("Half query BF FP16 D33 L2 sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "sq4");
}

TEST_CASE("Half query BF FP16 D33 IP sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 33, "sq4");
}

TEST_CASE("Half query BF FP16 D33 cosine sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "sq4");
}

TEST_CASE("Half query BF BF16 D33 L2 sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "sq4");
}

TEST_CASE("Half query BF BF16 D33 IP sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 33, "sq4");
}

TEST_CASE("Half query BF BF16 D33 cosine sq4", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "sq4");
}

TEST_CASE("Half query BF FP16 D7 L2 fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "fp16");
}

TEST_CASE("Half query BF FP16 D7 IP fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 7, "fp16");
}

TEST_CASE("Half query BF FP16 D7 cosine fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "fp16");
}

TEST_CASE("Half query BF BF16 D7 L2 fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "fp16");
}

TEST_CASE("Half query BF BF16 D7 IP fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 7, "fp16");
}

TEST_CASE("Half query BF BF16 D7 cosine fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "fp16");
}

TEST_CASE("Half query BF FP16 D33 L2 fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "fp16");
}

TEST_CASE("Half query BF FP16 D33 IP fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 33, "fp16");
}

TEST_CASE("Half query BF FP16 D33 cosine fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "fp16");
}

TEST_CASE("Half query BF BF16 D33 L2 fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "fp16");
}

TEST_CASE("Half query BF BF16 D33 IP fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 33, "fp16");
}

TEST_CASE("Half query BF BF16 D33 cosine fp16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "fp16");
}

TEST_CASE("Half query BF FP16 D7 L2 bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "bf16");
}

TEST_CASE("Half query BF FP16 D7 IP bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 7, "bf16");
}

TEST_CASE("Half query BF FP16 D7 cosine bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "bf16");
}

TEST_CASE("Half query BF BF16 D7 L2 bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "bf16");
}

TEST_CASE("Half query BF BF16 D7 IP bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 7, "bf16");
}

TEST_CASE("Half query BF BF16 D7 cosine bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "bf16");
}

TEST_CASE("Half query BF FP16 D33 L2 bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "bf16");
}

TEST_CASE("Half query BF FP16 D33 IP bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_IP, 33, "bf16");
}

TEST_CASE("Half query BF FP16 D33 cosine bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "bf16");
}

TEST_CASE("Half query BF BF16 D33 L2 bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "bf16");
}

TEST_CASE("Half query BF BF16 D33 IP bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_IP, 33, "bf16");
}

TEST_CASE("Half query BF BF16 D33 cosine bf16", "[half_query_matrix][half_query_bf]") {
    QueryExercise(false, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "bf16");
}

TEST_CASE("Half query lazy FP16 D7 L2 fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "fp32");
}

TEST_CASE("Half query lazy FP16 D7 IP fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_IP, 7, "fp32");
}

TEST_CASE("Half query lazy FP16 D7 cosine fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "fp32");
}

TEST_CASE("Half query lazy BF16 D7 L2 fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "fp32");
}

TEST_CASE("Half query lazy BF16 D7 IP fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_IP, 7, "fp32");
}

TEST_CASE("Half query lazy BF16 D7 cosine fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "fp32");
}

TEST_CASE("Half query lazy FP16 D33 L2 fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "fp32");
}

TEST_CASE("Half query lazy FP16 D33 IP fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_IP, 33, "fp32");
}

TEST_CASE("Half query lazy FP16 D33 cosine fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "fp32");
}

TEST_CASE("Half query lazy BF16 D33 L2 fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "fp32");
}

TEST_CASE("Half query lazy BF16 D33 IP fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_IP, 33, "fp32");
}

TEST_CASE("Half query lazy BF16 D33 cosine fp32", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "fp32");
}

TEST_CASE("Half query lazy FP16 D7 L2 sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "sq8");
}

TEST_CASE("Half query lazy FP16 D7 IP sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_IP, 7, "sq8");
}

TEST_CASE("Half query lazy FP16 D7 cosine sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_COSINE, 7, "sq8");
}

TEST_CASE("Half query lazy BF16 D7 L2 sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_L2SQR, 7, "sq8");
}

TEST_CASE("Half query lazy BF16 D7 IP sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_IP, 7, "sq8");
}

TEST_CASE("Half query lazy BF16 D7 cosine sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_COSINE, 7, "sq8");
}

TEST_CASE("Half query lazy FP16 D33 L2 sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "sq8");
}

TEST_CASE("Half query lazy FP16 D33 IP sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_IP, 33, "sq8");
}

TEST_CASE("Half query lazy FP16 D33 cosine sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, false, vsag::MetricType::METRIC_TYPE_COSINE, 33, "sq8");
}

TEST_CASE("Half query lazy BF16 D33 L2 sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_L2SQR, 33, "sq8");
}

TEST_CASE("Half query lazy BF16 D33 IP sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_IP, 33, "sq8");
}

TEST_CASE("Half query lazy BF16 D33 cosine sq8", "[half_query_matrix][half_query_lazy]") {
    QueryExercise(true, true, vsag::MetricType::METRIC_TYPE_COSINE, 33, "sq8");
}

TEST_CASE("Half raw input FP16 D7", "[half_raw_input_oracle]") {
    RawInputExercise(false, 7);
}

TEST_CASE("Half raw input BF16 D7", "[half_raw_input_oracle]") {
    RawInputExercise(true, 7);
}

TEST_CASE("Half raw input FP16 D33", "[half_raw_input_oracle]") {
    RawInputExercise(false, 33);
}

TEST_CASE("Half raw input BF16 D33", "[half_raw_input_oracle]") {
    RawInputExercise(true, 33);
}
