
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
#include <future>

#include "inner_string_params.h"
#include "json_wrapper.h"
#include "unittest.h"

TEST_CASE("HGraph distance early stop is query local and rejects unsupported modes",
          "[ut][HGraph][distance_early_stop]") {
    using namespace vsag;
    std::vector<float> vectors{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ids{100, 101, 102, 103, 104, 105, 106, 107};
    auto base = Dataset::Make()
                    ->NumElements(ids.size())
                    ->Dim(1)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    auto make_index = [&](const char* metric = "l2") {
        auto params = JsonType::Parse(R"({"dim":1,"dtype":"float32","metric_type":"l2",
            "index_param":{"max_degree":8,"ef_construction":64,"build_thread_count":1,
            "base_quantization_type":"fp32","use_reorder":false}})");
        params["metric_type"].SetString(metric);
        auto index = Factory::CreateIndex("hgraph", params.Dump());
        REQUIRE(index.has_value());
        REQUIRE(index.value()->Build(base).has_value());
        return index.value();
    };
    float query_value = 0;
    auto query =
        Dataset::Make()->NumElements(1)->Dim(1)->Float32Vectors(&query_value)->Owner(false);
    const std::string fixed_params = R"({"hgraph":{"ef_search":32}})";
    const std::string early_params =
        R"({"hgraph":{"ef_search":32,"early_stop":{"type":"distance","alpha":1.1}}})";
    auto index = make_index();
    auto fixed = index->KnnSearch(query, 2, fixed_params);
    auto early = index->KnnSearch(query, 2, early_params);
    REQUIRE(fixed.has_value());
    REQUIRE(early.has_value());
    REQUIRE(early.value()->GetDim() == 2);
    REQUIRE(early.value()->GetIds()[0] == 100);
    REQUIRE(early.value()->GetIds()[1] == 101);
    REQUIRE(early.value()->GetStatistics({"early_stop_count"})[0] == "1");
    auto again = index->KnnSearch(query, 2, fixed_params);
    REQUIRE(again.has_value());
    REQUIRE(again.value()->GetStatistics({"dist_cmp", "hops"}) ==
            fixed.value()->GetStatistics({"dist_cmp", "hops"}));

    SearchRequest thresholded;
    thresholded.query_ = query;
    thresholded.topk_ = 2;
    thresholded.params_str_ = early_params;
    thresholded.threshold_ = 2.0F;
    auto limited = index->SearchWithRequest(thresholded);
    REQUIRE(limited.has_value());
    REQUIRE(limited.value()->GetDim() == 1);
    REQUIRE(limited.value()->GetIds()[0] == 100);
    REQUIRE(limited.value()->GetStatistics({"dist_cmp", "hops"}) ==
            early.value()->GetStatistics({"dist_cmp", "hops"}));

    class OddLabels : public Filter {
    public:
        bool
        CheckValid(int64_t id) const override {
            return id % 2 == 1;
        }
    };
    auto filter = std::make_shared<OddLabels>();
    auto filtered = index->KnnSearch(query, 2, early_params, FilterPtr(filter));
    REQUIRE(filtered.has_value());
    REQUIRE(filtered.value()->GetIds()[0] == 101);
    REQUIRE(filtered.value()->GetIds()[1] == 103);

    REQUIRE_FALSE(index->RangeSearch(query, 4, early_params).has_value());
    IteratorContext* iterator = nullptr;
    REQUIRE_FALSE(
        index->KnnSearch(query, 2, early_params, FilterPtr{}, iterator, false).has_value());
    REQUIRE(iterator == nullptr);
    REQUIRE_FALSE(index->OpenSearchSession(query, 2, early_params).has_value());

    auto ip = make_index("ip");
    REQUIRE_FALSE(ip->KnnSearch(query, 2, early_params).has_value());

    bool callback_invoked = false;
    SearchRequest request;
    request.query_ = query;
    request.topk_ = 2;
    request.params_str_ = early_params;
    request.distance_batch_func_ = [&](const int64_t*, uint64_t, float*) {
        callback_invoked = true;
    };
    REQUIRE_FALSE(index->SearchWithRequest(request).has_value());
    REQUIRE_FALSE(callback_invoked);
}

TEST_CASE("HGraph distance early stop is isolated across batches and serialization",
          "[ut][HGraph][distance_early_stop]") {
    using namespace vsag;
    const std::string create = R"({"dim":1,"dtype":"float32","metric_type":"l2",
        "index_param":{"max_degree":8,"ef_construction":64,"build_thread_count":1,
        "base_quantization_type":"fp32","use_reorder":false}})";
    auto built = Factory::CreateIndex("hgraph", create);
    REQUIRE(built.has_value());
    auto index = built.value();
    std::vector<float> vectors{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ids{100, 101, 102, 103, 104, 105, 106, 107};
    auto base = Dataset::Make()
                    ->NumElements(ids.size())
                    ->Dim(1)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    const std::string early =
        R"({"hgraph":{"ef_search":32,"early_stop":{"type":"distance","alpha":1.1}}})";
    const std::string fixed = R"({"hgraph":{"ef_search":32}})";
    std::array<float, 2> query_values{0.0F, 10.0F};
    auto count_early = [](const DatasetPtr& result) -> uint64_t {
        const auto value = result->GetStatistics({"early_stop_count"})[0];
        return value.empty() ? 0 : std::stoull(value);
    };
    auto verify = [&](const IndexPtr& target) {
        auto queries = Dataset::Make()
                           ->NumElements(query_values.size())
                           ->Dim(1)
                           ->Float32Vectors(query_values.data())
                           ->Owner(false);
        auto batch = target->KnnSearch(queries, 2, early);
        REQUIRE(batch.has_value());
        REQUIRE(batch.value()->GetNumElements() == 2);
        REQUIRE(batch.value()->GetDim() == 2);
        uint64_t expected_stops = 0;
        for (uint64_t row = 0; row < query_values.size(); ++row) {
            auto query = Dataset::Make()
                             ->NumElements(1)
                             ->Dim(1)
                             ->Float32Vectors(query_values.data() + row)
                             ->Owner(false);
            auto single = target->KnnSearch(query, 2, early);
            REQUIRE(single.has_value());
            expected_stops += count_early(single.value());
            for (uint64_t col = 0; col < 2; ++col) {
                REQUIRE(batch.value()->GetIds()[2 * row + col] == single.value()->GetIds()[col]);
                REQUIRE(batch.value()->GetDistances()[2 * row + col] ==
                        single.value()->GetDistances()[col]);
            }
        }
        REQUIRE(expected_stops == 2);
        REQUIRE(count_early(batch.value()) == expected_stops);
        auto disabled = target->KnnSearch(queries, 2, fixed);
        REQUIRE(disabled.has_value());
        REQUIRE(count_early(disabled.value()) == 0);
        REQUIRE(disabled.value()->GetStatistics({"early_stop_count"})[0].empty());

        std::vector<std::future<tl::expected<DatasetPtr, Error>>> concurrent;
        for (uint64_t i = 0; i < 8; ++i) {
            concurrent.emplace_back(std::async(std::launch::async, [&, target, i] {
                return target->KnnSearch(queries, 2, i % 2 == 0 ? early : fixed);
            }));
        }
        for (uint64_t i = 0; i < concurrent.size(); ++i) {
            auto result = concurrent[i].get();
            REQUIRE(result.has_value());
            REQUIRE(count_early(result.value()) == (i % 2 == 0 ? expected_stops : 0));
        }
    };
    verify(index);
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = Factory::CreateIndex("hgraph", create);
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(serialized.value()).has_value());
    verify(restored.value());
}

TEST_CASE(
    "HGraph distance early stop runs on quantized, reordered, conjugate and parallel searches",
    "[ut][HGraph][distance_early_stop]") {
    using namespace vsag;
    struct Variant {
        const char* quantizer;
        bool reorder;
        bool conjugate;
        int64_t threads;
    };
    const auto variant = GENERATE(Variant{"sq8", false, false, 1},
                                  Variant{"fp16", false, false, 1},
                                  Variant{"sq8", true, false, 1},
                                  Variant{"fp32", false, true, 1},
                                  Variant{"fp32", false, false, 2},
                                  Variant{"sq8", true, false, 2});
    CAPTURE(variant.quantizer, variant.reorder, variant.conjugate, variant.threads);
    constexpr int64_t count = 64;
    std::vector<float> vectors(count);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i] = static_cast<float>(i + 1);
        ids[i] = 100 + i;
    }
    auto base = Dataset::Make()
                    ->NumElements(count)
                    ->Dim(1)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    auto create = JsonType::Parse(R"({"dim":1,"dtype":"float32","metric_type":"l2",
        "index_param":{"max_degree":8,"ef_construction":64,"build_thread_count":2,
        "base_quantization_type":"fp32","use_reorder":false,"precise_quantization_type":"fp32"}})");
    create["index_param"]["base_quantization_type"].SetString(variant.quantizer);
    create["index_param"]["use_reorder"].SetBool(variant.reorder);
    create["index_param"]["use_conjugate_graph"].SetBool(variant.conjugate);
    auto built = Factory::CreateIndex("hgraph", create.Dump());
    REQUIRE(built.has_value());
    auto index = built.value();
    REQUIRE(index->Build(base).has_value());

    float query_value = 0;
    auto query =
        Dataset::Make()->NumElements(1)->Dim(1)->Float32Vectors(&query_value)->Owner(false);
    auto params = [&](const char* early_stop) {
        auto parsed = JsonType::Parse(R"({"hgraph":{"ef_search":32}})");
        parsed["hgraph"]["parallelism"].SetInt(variant.threads);
        if (early_stop != nullptr) {
            parsed["hgraph"]["early_stop"].SetJson(JsonType::Parse(early_stop));
        }
        return parsed.Dump();
    };
    auto fixed = index->KnnSearch(query, 2, params(nullptr));
    auto unreached = index->KnnSearch(query, 2, params(R"({"type":"distance","alpha":1e30})"));
    auto early = index->KnnSearch(query, 2, params(R"({"type":"distance","alpha":1.1})"));
    REQUIRE(fixed.has_value());
    REQUIRE(unreached.has_value());
    REQUIRE(early.has_value());

    // A threshold that is never reached leaves the traversal and its results unchanged.
    REQUIRE(unreached.value()->GetStatistics({"dist_cmp", "hops"}) ==
            fixed.value()->GetStatistics({"dist_cmp", "hops"}));
    REQUIRE(unreached.value()->GetStatistics({"early_stop_count"})[0] == "0");
    REQUIRE(unreached.value()->GetDim() == fixed.value()->GetDim());
    for (int64_t i = 0; i < fixed.value()->GetDim(); ++i) {
        REQUIRE(unreached.value()->GetIds()[i] == fixed.value()->GetIds()[i]);
        REQUIRE(unreached.value()->GetDistances()[i] == fixed.value()->GetDistances()[i]);
    }

    // A tight threshold stops the traversal early and keeps the two nearest points.
    REQUIRE(early.value()->GetStatistics({"early_stop_count"})[0] == "1");
    REQUIRE(early.value()->GetDim() == 2);
    REQUIRE(early.value()->GetIds()[0] == 100);
    REQUIRE(early.value()->GetIds()[1] == 101);
    if (variant.threads == 1) {
        REQUIRE(std::stoull(early.value()->GetStatistics({"dist_cmp"})[0]) <
                std::stoull(fixed.value()->GetStatistics({"dist_cmp"})[0]));
    } else {
        // The parallel traversal does not report its bottom-layer counters; check that it ran.
        REQUIRE(early.value()->GetStatistics({"parallel_search_fallback_count"})[0] == "0");
    }
}
