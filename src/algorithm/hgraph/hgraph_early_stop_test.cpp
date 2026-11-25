
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

#include <vsag/dataset.h>
#include <vsag/factory.h>
#include <vsag/filter.h>
#include <vsag/index.h>

#include <array>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <future>
#include <numeric>

#include "hgraph_parameter.h"
#include "impl/logger/capturing_logger_test.h"
#include "json_wrapper.h"
#include "unittest.h"

using namespace vsag;

namespace {

// Every case searches a line: the points 1, 2, ..., count carry the labels 100, 101, ..., and the
// query sits at 0, so label 99 + i is its i-th nearest neighbour, at l2 distance i * i.

// A one-dimensional HGraph over the line. Entries of index_param replace the defaults: FP32 codes,
// no reordering, a single build thread.
IndexPtr
build_line_index(int64_t count,
                 const std::string& index_param = "{}",
                 const std::string& metric = "l2") {
    auto params = JsonType::Parse(R"({"dim":1,"dtype":"float32",
        "index_param":{"max_degree":8,"ef_construction":64,"build_thread_count":1,
        "base_quantization_type":"fp32","use_reorder":false}})");
    params["metric_type"].SetString(metric);
    params["index_param"].UpdateJson(JsonType::Parse(index_param));
    auto index = Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(index.has_value());
    std::vector<float> points(count);
    std::vector<int64_t> ids(count);
    std::iota(points.begin(), points.end(), 1.0F);
    std::iota(ids.begin(), ids.end(), 100);
    auto base = Dataset::Make()
                    ->NumElements(count)
                    ->Dim(1)
                    ->Ids(ids.data())
                    ->Float32Vectors(points.data())
                    ->Owner(false);
    REQUIRE(index.value()->Build(base).has_value());
    return index.value();
}

// One one-dimensional query per value; the values must outlive the dataset.
DatasetPtr
make_queries(const float* values, int64_t count = 1) {
    return Dataset::Make()->NumElements(count)->Dim(1)->Float32Vectors(values)->Owner(false);
}

DatasetPtr
origin_query() {
    static const float origin = 0.0F;
    return make_queries(&origin);
}

// With k = 2 the search from the origin holds the distances 1 and 4 while its nearest pending
// candidate is at 9: alpha 1.1 stops it there, and no candidate is ever 1e30 times as far.
const std::string kTightAlpha = R"({"type":"distance","alpha":1.1})";
const std::string kUnreachableAlpha = R"({"type":"distance","alpha":1e30})";
// alpha 1 is the tightest rule there is: wherever it runs, it ends the search from the origin.
const std::string kTightestAlpha = R"({"type":"distance","alpha":1.0})";

// Search parameters with the given early_stop object, or without the option when it is empty.
std::string
search_params(int64_t ef_search, const std::string& early_stop = "", int64_t parallelism = 1) {
    auto params = JsonType::Parse(R"({"hgraph":{}})");
    params["hgraph"]["ef_search"].SetInt(ef_search);
    params["hgraph"]["parallelism"].SetInt(parallelism);
    if (not early_stop.empty()) {
        params["hgraph"]["early_stop"].SetJson(JsonType::Parse(early_stop));
    }
    return params.Dump();
}

// A KNN search that must succeed.
DatasetPtr
knn_search(const IndexPtr& index, const DatasetPtr& query, int64_t k, const std::string& params) {
    auto result = index->KnnSearch(query, k, params);
    REQUIRE(result.has_value());
    return result.value();
}

std::vector<int64_t>
labels(const DatasetPtr& result, int64_t row = 0) {
    const auto* begin = result->GetIds() + row * result->GetDim();
    return {begin, begin + result->GetDim()};
}

std::vector<float>
distances(const DatasetPtr& result, int64_t row = 0) {
    const auto* begin = result->GetDistances() + row * result->GetDim();
    return {begin, begin + result->GetDim()};
}

// A statistic of the result; empty when the search did not report it.
std::string
stat(const DatasetPtr& result, const std::string& key) {
    return result->GetStatistics({key})[0];
}

// The traversal work a search reports.
const std::vector<std::string> kWork{"dist_cmp", "hops"};

// The request failed, and for the reason given.
template <typename T>
void
require_rejected(const tl::expected<T, Error>& result, const std::string& reason) {
    REQUIRE_FALSE(result.has_value());
    REQUIRE_THAT(result.error().message, Catch::Matchers::ContainsSubstring(reason));
}

}  // namespace

TEST_CASE("HGraph distance early stop ends a search and keeps the nearest results",
          "[ut][HGraph][distance_early_stop]") {
    auto index = build_line_index(8);
    const auto fixed = knn_search(index, origin_query(), 2, search_params(32));
    const auto params = search_params(32, kTightAlpha);
    const auto early = knn_search(index, origin_query(), 2, params);
    REQUIRE(labels(early) == std::vector<int64_t>{100, 101});
    REQUIRE(stat(early, "early_stop_count") == "1");

    // The option belongs to its request: the same search without it does the fixed work again.
    const auto again = knn_search(index, origin_query(), 2, search_params(32));
    REQUIRE(again->GetStatistics(kWork) == fixed->GetStatistics(kWork));

    // A result threshold applies after the traversal, so the rule still waits for two results:
    // the traversal is the same, and the threshold then drops label 101 at distance 4.
    SearchRequest request;
    request.query_ = origin_query();
    request.topk_ = 2;
    request.params_str_ = params;
    request.threshold_ = 2.0F;
    auto limited = index->SearchWithRequest(request);
    REQUIRE(limited.has_value());
    REQUIRE(labels(limited.value()) == std::vector<int64_t>{100});
    REQUIRE(limited.value()->GetStatistics(kWork) == early->GetStatistics(kWork));
}

TEST_CASE("HGraph distance early stop is rejected for searches it does not support",
          "[ut][HGraph][distance_early_stop]") {
    auto index = build_line_index(8);
    const auto query = origin_query();
    const auto params = search_params(32, kTightAlpha);

    SECTION("range search") {
        require_rejected(index->RangeSearch(query, 4, params), "requires built-in KNN distance");
    }
    SECTION("iterator search") {
        IteratorContext* iterator = nullptr;
        require_rejected(index->KnnSearch(query, 2, params, FilterPtr{}, iterator, false),
                         "does not support iterator search");
        REQUIRE(iterator == nullptr);
        // SearchWithRequest hands iterator requests to the same KnnSearch overload.
        SearchRequest request;
        request.query_ = query;
        request.topk_ = 2;
        request.params_str_ = params;
        request.enable_iterator_search_ = true;
        request.p_iter_ctx_ = &iterator;
        require_rejected(index->SearchWithRequest(request), "does not support iterator search");
        REQUIRE(iterator == nullptr);
    }
    SECTION("search session") {
        require_rejected(index->OpenSearchSession(query, 2, params), "early-stopped search");
    }
    SECTION("inner product") {
        auto ip = build_line_index(8, "{}", "ip");
        require_rejected(ip->KnnSearch(query, 2, params), "requires l2 or cosine");
    }
    SECTION("custom distance callback") {
        bool callback_invoked = false;
        SearchRequest request;
        request.query_ = query;
        request.topk_ = 2;
        request.params_str_ = params;
        request.distance_batch_func_ = [&](const int64_t*, uint64_t, float*) {
            callback_invoked = true;
        };
        require_rejected(index->SearchWithRequest(request), "requires built-in KNN distance");
        REQUIRE_FALSE(callback_invoked);
    }
}

TEST_CASE("HGraph distance early stop counts the stopped queries of each request",
          "[ut][HGraph][distance_early_stop]") {
    auto index = build_line_index(8);
    // Both queries stop early: the two nearest points are 1 and 2 from 0, and 8 and 7 from 10.
    const std::array<float, 2> values{0.0F, 10.0F};
    const auto batch_query = make_queries(values.data(), 2);
    const auto early_params = search_params(32, kTightAlpha);
    const auto fixed_params = search_params(32);

    SECTION("a batch returns each query's own results and the sum of the counts") {
        const auto batch = knn_search(index, batch_query, 2, early_params);
        for (int64_t row = 0; row < 2; ++row) {
            const auto single =
                knn_search(index, make_queries(values.data() + row), 2, early_params);
            REQUIRE(labels(batch, row) == labels(single));
            REQUIRE(distances(batch, row) == distances(single));
        }
        REQUIRE(stat(batch, "early_stop_count") == "2");
    }
    SECTION("concurrent requests each report their own count") {
        std::vector<std::future<tl::expected<DatasetPtr, Error>>> requests;
        for (uint64_t i = 0; i < 8; ++i) {
            requests.emplace_back(std::async(std::launch::async, [&, i] {
                return index->KnnSearch(batch_query, 2, i % 2 == 0 ? early_params : fixed_params);
            }));
        }
        for (uint64_t i = 0; i < requests.size(); ++i) {
            auto result = requests[i].get();
            REQUIRE(result.has_value());
            REQUIRE(stat(result.value(), "early_stop_count") == (i % 2 == 0 ? "2" : ""));
        }
    }
}

TEST_CASE(
    "HGraph distance early stop runs on quantized, reordered, conjugate and parallel searches",
    "[ut][HGraph][distance_early_stop]") {
    struct Variant {
        const char* quantizer;
        bool reorder;
        bool conjugate;
        int64_t parallelism;
    };
    const auto variant = GENERATE(Variant{"sq8", false, false, 1},
                                  Variant{"sq8", true, false, 1},
                                  Variant{"fp32", false, true, 1},
                                  Variant{"fp32", false, false, 2});
    CAPTURE(variant.quantizer, variant.reorder, variant.conjugate, variant.parallelism);
    // A build with two threads leaves the index the thread pool that parallelism 2 searches on.
    auto index_param =
        JsonType::Parse(R"({"build_thread_count":2,"precise_quantization_type":"fp32"})");
    index_param["base_quantization_type"].SetString(variant.quantizer);
    index_param["use_reorder"].SetBool(variant.reorder);
    index_param["use_conjugate_graph"].SetBool(variant.conjugate);
    auto index = build_line_index(64, index_param.Dump());
    const auto query = origin_query();
    auto params = [&](const std::string& early_stop = "") {
        return search_params(32, early_stop, variant.parallelism);
    };
    if (variant.conjugate) {
        // The conjugate enhancement returns at once while its graph is empty. Feedback links the
        // two nearest labels to the farthest one, 163, which every search below scores and drops.
        REQUIRE(index->Feedback(query, 2, params(), 163).value() == 2);
    }
    const auto fixed = knn_search(index, query, 2, params());
    const auto unreached = knn_search(index, query, 2, params(kUnreachableAlpha));
    const auto early = knn_search(index, query, 2, params(kTightAlpha));

    // A threshold that is never reached leaves the traversal and its results unchanged.
    REQUIRE(stat(unreached, "early_stop_count") == "0");
    REQUIRE(unreached->GetStatistics(kWork) == fixed->GetStatistics(kWork));
    REQUIRE(labels(unreached) == labels(fixed));
    REQUIRE(distances(unreached) == distances(fixed));

    // A tight threshold stops the traversal early and keeps the two nearest points.
    REQUIRE(stat(early, "early_stop_count") == "1");
    REQUIRE(labels(early) == std::vector<int64_t>{100, 101});
    REQUIRE(std::stoull(stat(early, "dist_cmp")) < std::stoull(stat(fixed, "dist_cmp")));
    if (variant.parallelism > 1) {
        // The parallel traversal answered: a fallback to the serial one would be counted.
        REQUIRE(stat(early, "parallel_search_fallback_count") == "0");
    }
}

TEST_CASE("HGraph distance early stop is named in the reasoning report",
          "[ut][HGraph][distance_early_stop][reasoning]") {
    // The serial and the parallel traversal each report the stop themselves. A build with two
    // threads leaves the index the thread pool that parallelism 2 searches on.
    const int64_t parallelism = GENERATE(1, 2);
    CAPTURE(parallelism);
    auto index = build_line_index(64, R"({"build_thread_count":2})");
    SearchRequest request;
    request.query_ = origin_query();
    request.topk_ = 2;
    request.params_str_ = search_params(32, kTightAlpha, parallelism);
    request.expected_labels_.push_back(100);
    auto result = index->SearchWithRequest(request);
    REQUIRE(result.has_value());
    REQUIRE(stat(result.value(), "early_stop_count") == "1");
    REQUIRE(stat(result.value(), "parallel_search_fallback_count") == "0");
    const auto report = JsonType::Parse(result.value()->GetReasoning());
    REQUIRE(report["meta"]["termination_reason"].GetString() == "distance_early_stop");
}

TEST_CASE("HGraph distance early stop picks alpha from k when alpha is omitted",
          "[ut][HGraph][distance_early_stop]") {
    auto index = build_line_index(64);
    const std::vector<std::string> keys{"dist_cmp", "hops", "early_stop_count"};
    uint64_t stopped = 0;
    for (const int64_t k : {2, 10, 40}) {
        CAPTURE(k);
        auto with_alpha = JsonType::Parse(R"({"type":"distance"})");
        with_alpha["alpha"].SetFloat(HGraphSearchParameters::DefaultDistanceEarlyStopAlpha(k));
        const auto omitted =
            knn_search(index, origin_query(), k, search_params(64, R"({"type":"distance"})"));
        const auto given =
            knn_search(index, origin_query(), k, search_params(64, with_alpha.Dump()));
        REQUIRE(omitted->GetStatistics(keys) == given->GetStatistics(keys));
        REQUIRE(labels(omitted) == labels(given));
        REQUIRE(distances(omitted) == distances(given));
        stopped += std::stoull(stat(omitted, "early_stop_count"));
    }
    REQUIRE(stopped > 0);
}

TEST_CASE("HGraph distance early stop is skipped when ef_search is at most 1.25 k",
          "[ut][HGraph][distance_early_stop]") {
    testing::CapturingLogger capture;
    const testing::ScopedLogger installed(&capture);
    auto skip_warnings = [&capture]() {
        return std::count_if(
            capture.lines.begin(), capture.lines.end(), [](const std::string& line) {
                return line.find("early_stop is skipped") != std::string::npos;
            });
    };
    auto index = build_line_index(64);
    constexpr int64_t k = 8;
    const std::vector<std::string> keys{"dist_cmp", "hops", "early_stop_count"};
    // 10 <= 1.25 * 8: the search runs as if early_stop were absent and reports no count.
    for (const int64_t ef_search : {4, 8, 10}) {
        CAPTURE(ef_search);
        const auto fixed = knn_search(index, origin_query(), k, search_params(ef_search));
        const auto skipped =
            knn_search(index, origin_query(), k, search_params(ef_search, kTightestAlpha));
        REQUIRE(stat(skipped, "early_stop_count").empty());
        REQUIRE(skipped->GetStatistics(keys) == fixed->GetStatistics(keys));
        REQUIRE(labels(skipped) == labels(fixed));
    }
    // 11 > 1.25 * 8: the rule runs.
    const auto applied = knn_search(index, origin_query(), k, search_params(11, kTightestAlpha));
    REQUIRE(stat(applied, "early_stop_count") == "1");
    // The index warned once; a second index warns for itself.
    REQUIRE(skip_warnings() == 1);
    knn_search(build_line_index(64), origin_query(), k, search_params(10, kTightestAlpha));
    REQUIRE(skip_warnings() == 2);
}
