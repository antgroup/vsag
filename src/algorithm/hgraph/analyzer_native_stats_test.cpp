// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "algorithm/bruteforce/half_dense_test_fixture.h"
#include "algorithm/hgraph/hgraph.h"
#include "analyzer/hgraph_analyzer.h"

namespace {
vsag::IndexCommonParam
NativeStatsCommon(vsag::DataTypes dtype) {
    vsag::IndexCommonParam c;
    static const auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    c.allocator_ = allocator;
    c.dim_ = half_dense_test::DIM;
    c.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    c.data_type_ = dtype;
    return c;
}
}  // namespace

TEST_CASE("Analyzer native half and existing FLOAT INT8 sample contracts",
          "[AnalyzerNativeStats]") {
    for (auto dtype : {vsag::DataTypes::DATA_TYPE_FLOAT,
                       vsag::DataTypes::DATA_TYPE_FP16,
                       vsag::DataTypes::DATA_TYPE_BF16,
                       vsag::DataTypes::DATA_TYPE_INT8}) {
        INFO("dtype=" << static_cast<int>(dtype));
        auto common = NativeStatsCommon(dtype);
        auto params = vsag::JsonType::Parse(
            R"({"max_degree":16,"ef_construction":32,"build_thread_count":1})");
        params["base_quantization_type"].SetString(
            dtype == vsag::DataTypes::DATA_TYPE_INT8 ? "int8" : "fp32");
        auto mapped = vsag::HGraph::CheckAndMappingExternalParam(params, common);
        auto graph = std::make_shared<vsag::HGraph>(mapped, common);
        graph->InitFeatures();
        half_dense_test::Rows rows(128, 100, dtype == vsag::DataTypes::DATA_TYPE_BF16);
        std::vector<int8_t> bytes(128 * half_dense_test::DIM);
        for (uint64_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<int8_t>(i % 13 - 6);
        auto data = rows.Data(dtype != vsag::DataTypes::DATA_TYPE_FLOAT);
        if (dtype == vsag::DataTypes::DATA_TYPE_INT8) {
            data = vsag::Dataset::Make()
                       ->Dim(common.dim_)
                       ->NumElements(128)
                       ->Ids(rows.ids.data())
                       ->Int8Vectors(bytes.data())
                       ->Owner(false);
        }
        REQUIRE(graph->Build(data).empty());
        REQUIRE(graph->GetNumElements() == 128);
        REQUIRE_NOTHROW(graph->GetStats());
        vsag::AnalyzerParam ap(common.allocator_.get());
        ap.base_sample_size = 4;
        ap.topk = 5;
        ap.search_params = R"({"hgraph":{"ef_search":32}})";
        vsag::HGraphAnalyzer analyzer(graph.get(), ap);
        auto query = rows.Data(dtype != vsag::DataTypes::DATA_TYPE_FLOAT, 0, 2);
        if (dtype == vsag::DataTypes::DATA_TYPE_INT8) {
            query = vsag::Dataset::Make()
                        ->Dim(common.dim_)
                        ->NumElements(2)
                        ->Int8Vectors(bytes.data())
                        ->Owner(false);
        }
        REQUIRE(analyzer.SetQuery(query));
        const auto before = analyzer.GetQueryAvgDistance();
        REQUIRE(std::isfinite(before));
        REQUIRE(std::isfinite(analyzer.GetQuerySearchRecall(ap.search_params)));
        auto invalid = vsag::Dataset::Make()
                           ->Dim(common.dim_ + 1)
                           ->NumElements(2)
                           ->Float32Vectors(rows.fp32.data())
                           ->Owner(false);
        REQUIRE_FALSE(analyzer.SetQuery(invalid));
        REQUIRE(analyzer.GetQueryAvgDistance() == before);
        vsag::SearchRequest bad_request;
        bad_request.query_ = invalid;
        bad_request.params_str_ = ap.search_params;
        REQUIRE_THROWS_AS(analyzer.AnalyzeIndexBySearch(bad_request), vsag::VsagException);
        REQUIRE(analyzer.GetQueryAvgDistance() == before);
        if (dtype == vsag::DataTypes::DATA_TYPE_FP16 || dtype == vsag::DataTypes::DATA_TYPE_BF16) {
            REQUIRE_FALSE(analyzer.SetQuery(rows.Data(false, 0, 2)));
            REQUIRE(analyzer.GetQueryAvgDistance() == before);
            REQUIRE(analyzer.SetQuery(rows.Data(true, 1, 2)));
            REQUIRE(std::isfinite(analyzer.GetQueryAvgDistance()));
            auto lazy_params = vsag::JsonType::Parse(R"({"transition_threshold":4})");
            lazy_params["hgraph"].SetJson(params);
            auto lp = vsag::LazyHGraph::CheckAndMappingExternalParam(lazy_params, common);
            auto lazy = std::make_shared<vsag::LazyHGraph>(lp, common);
            lazy->InitFeatures();
            REQUIRE(lazy->Build(rows.Data(true)).empty());
            REQUIRE(lazy->GetNumElements() == 128);
            REQUIRE_NOTHROW(lazy->GetStats());
        }
    }
}
