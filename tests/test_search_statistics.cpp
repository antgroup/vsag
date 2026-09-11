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

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "functest.h"
#include "vsag/search_request.h"
#include "vsag/vsag.h"

namespace {

constexpr const char* kCommonParams = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 4,
    "index_param": {"base_quantization_type": "fp32"}
})";

vsag::IndexPtr
MakeBruteForce() {
    auto index = vsag::Factory::CreateIndex("brute_force", kCommonParams);
    REQUIRE(index.has_value());
    return index.value();
}

vsag::DatasetPtr
MakeQuery() {
    static const float vector[] = {0.1F, 0.2F, 0.3F, 0.4F};
    return vsag::Dataset::Make()->NumElements(1)->Dim(4)->Float32Vectors(vector)->Owner(false);
}

void
BuildBase(const vsag::IndexPtr& index) {
    static std::vector<float> vectors = []() {
        std::vector<float> result(48);
        for (uint64_t i = 0; i < 12; ++i) {
            result[4 * i] = static_cast<float>(i) * 0.01F;
            result[4 * i + 1] = static_cast<float>(i) * 0.02F;
            result[4 * i + 2] = 0.5F - static_cast<float>(i) * 0.03F;
            result[4 * i + 3] = 1.0F;
        }
        return result;
    }();
    static std::vector<int64_t> ids = []() {
        std::vector<int64_t> result(12);
        for (uint64_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<int64_t>(i);
        }
        return result;
    }();
    auto base = vsag::Dataset::Make()
                    ->NumElements(12)
                    ->Dim(4)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
}

vsag::SearchRequest
MakeRequest(bool want_statistics = true) {
    vsag::SearchRequest request;
    request.query_ = MakeQuery();
    request.topk_ = 1;
    request.params_str_ = want_statistics ? "{}" : R"({"want_statistics":false})";
    return request;
}

}  // namespace

TEST_CASE("BruteForce SearchWithRequest exposes typed search statistics",
          "[ft][search_statistics]") {
    auto index = MakeBruteForce();
    BuildBase(index);
    auto result = index->SearchWithRequest(MakeRequest());
    REQUIRE(result.has_value());

    auto metrics = result.value()->GetSearchMetrics();
    REQUIRE(metrics.has_value());
    CHECK(metrics->distance_evaluations == 12);
    CHECK(metrics->DistanceEvaluations(vsag::DistanceEvaluationPhase::APPROXIMATE) == 12);
    CHECK(metrics->DistanceEvaluations(vsag::DistanceEvaluationBackend::FP32) == 12);
    auto actual_json = vsag::JsonType::Parse(result.value()->GetStatistics());
    for (const auto& field : vsag::MetricsSchema::Base().Fields()) {
        CHECK(actual_json.Contains(field.name));
    }
    CHECK(actual_json["distance_evaluations"].GetUint64() == 12);
    CHECK(actual_json["distance_evaluations_by_phase"]["approximate"].GetUint64() == 12);
    CHECK(actual_json["distance_evaluations_by_backend"]["fp32"].GetUint64() == 12);
    CHECK(actual_json["complete"].GetBool());
}

TEST_CASE("BruteForce empty SearchWithRequest exposes a typed zero snapshot",
          "[ft][search_statistics]") {
    auto result = MakeBruteForce()->SearchWithRequest(MakeRequest());
    REQUIRE(result.has_value());

    auto metrics = result.value()->GetSearchMetrics();
    REQUIRE(metrics.has_value());
    CHECK(metrics->distance_evaluations == 0);
    CHECK(metrics->dist_cmp == 0);
    auto json = vsag::JsonType::Parse(result.value()->GetStatistics());
    CHECK(json["distance_evaluations"].GetUint64() == 0);
    CHECK(json["complete"].GetBool());
}

TEST_CASE("SearchWithRequest can disable result statistics", "[ft][search_statistics]") {
    SECTION("typed BruteForce path preserves search results") {
        auto index = MakeBruteForce();
        BuildBase(index);
        auto enabled = index->SearchWithRequest(MakeRequest());
        auto disabled = index->SearchWithRequest(MakeRequest(false));
        REQUIRE(enabled.has_value());
        REQUIRE(disabled.has_value());

        CHECK(disabled.value()->GetIds()[0] == enabled.value()->GetIds()[0]);
        CHECK(disabled.value()->GetDistances()[0] == enabled.value()->GetDistances()[0]);
        CHECK_FALSE(disabled.value()->GetSearchMetrics().has_value());
        CHECK(disabled.value()->GetStatistics() == "{}");
    }

    SECTION("typed BruteForce empty-index path") {
        auto result = MakeBruteForce()->SearchWithRequest(MakeRequest(false));
        REQUIRE(result.has_value());
        CHECK_FALSE(result.value()->GetSearchMetrics().has_value());
        CHECK(result.value()->GetStatistics() == "{}");
    }

    SECTION("legacy JSON-only HGraph path") {
        auto created = vsag::Factory::CreateIndex("hgraph", kCommonParams);
        REQUIRE(created.has_value());
        auto index = created.value();
        BuildBase(index);
        auto request = MakeRequest(false);
        request.params_str_ = R"({"want_statistics":false,"hgraph":{"ef_search":10}})";
        auto result = index->SearchWithRequest(request);
        REQUIRE(result.has_value());

        CHECK_FALSE(result.value()->GetSearchMetrics().has_value());
        CHECK(result.value()->GetStatistics() == "{}");
    }

    SECTION("legacy JSON-only HGraph empty-index path") {
        auto created = vsag::Factory::CreateIndex("hgraph", kCommonParams);
        REQUIRE(created.has_value());
        auto request = MakeRequest(false);
        request.params_str_ = R"({"want_statistics":false,"hgraph":{"ef_search":10}})";
        auto result = created.value()->SearchWithRequest(request);
        REQUIRE(result.has_value());

        CHECK_FALSE(result.value()->GetSearchMetrics().has_value());
        CHECK(result.value()->GetStatistics() == "{}");
    }

    SECTION("escaped top-level keys honor suppression") {
        auto index = MakeBruteForce();
        BuildBase(index);
        auto request = MakeRequest();
        request.params_str_ = R"({"want_\u0073tatistics":false})";
        auto result = index->SearchWithRequest(request);
        REQUIRE(result.has_value());
        CHECK_FALSE(result.value()->GetSearchMetrics().has_value());
        CHECK(result.value()->GetStatistics() == "{}");
    }

    SECTION("nested reserved keys do not suppress statistics") {
        auto index = MakeBruteForce();
        BuildBase(index);
        auto request = MakeRequest();
        request.params_str_ = R"({"metadata":{"want_statistics":false}})";
        auto result = index->SearchWithRequest(request);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetSearchMetrics().has_value());
        CHECK(result.value()->GetSearchMetrics()->distance_evaluations == 12);
    }

    SECTION("escaped reserved parameter type is validated") {
        auto request = MakeRequest();
        request.params_str_ = R"({"want_\u0073tatistics":"false"})";
        auto result = MakeBruteForce()->SearchWithRequest(request);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("reserved parameter type is validated") {
        auto request = MakeRequest();
        request.params_str_ = R"({"want_statistics":"false"})";
        auto result = MakeBruteForce()->SearchWithRequest(request);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("BruteForce legacy search keeps JSON statistics compatibility",
          "[ft][search_statistics]") {
    auto index = MakeBruteForce();
    BuildBase(index);
    auto result = index->KnnSearch(MakeQuery(), 1, "{}");
    REQUIRE(result.has_value());

    auto json = vsag::JsonType::Parse(result.value()->GetStatistics());
    CHECK(json["distance_evaluations"].GetUint64() == 12);
    REQUIRE(result.value()->GetSearchMetrics().has_value());
}
