// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <atomic>
#include <thread>
#include <vector>

#include "json_types.h"
#include "unittest.h"
#include "vsag/resource.h"
#include "vsag/vsag.h"

TEST_CASE("HGraph result statistics remain independent and readable", "[ut][hgraph_statistics]") {
    std::vector<float> vectors(64 * 4);
    std::vector<int64_t> ids(64);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int64_t>(i);
        for (uint64_t j = 0; j < 4; ++j) {
            vectors[i * 4 + j] = static_cast<float>(i + j) / 64.0F;
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(64)->Dim(4)->Ids(ids.data())->Float32Vectors(vectors.data())->Owner(false);
    auto created = vsag::Factory::CreateIndex("hgraph",
                                              R"({"dtype":"float32","metric_type":"l2","dim":4,
             "index_param":{"base_quantization_type":"fp32","max_degree":16,
                            "ef_construction":64}})");
    REQUIRE(created.has_value());
    auto index = std::move(created.value());
    REQUIRE(index->Build(base).has_value());
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(vectors.data() + 7 * 4)->Owner(false);
    auto searched = index->KnnSearch(query, 3, R"({"hgraph":{"ef_search":16}})");
    REQUIRE(searched.has_value());
    auto result = std::move(searched.value());
    REQUIRE(result->GetIds()[0] == 7);
    const auto snapshot = result->GetStatistics();
    const auto parsed = vsag::JsonType::Parse(snapshot);
    REQUIRE(parsed["distance_evaluations"].GetUint64() > 0);
    const auto phases = parsed["distance_evaluations_by_phase"];
    REQUIRE(parsed["distance_evaluations"].GetUint64() == phases["routing"].GetUint64() +
                                                              phases["approximate"].GetUint64() +
                                                              phases["rerank"].GetUint64());
    const auto selected = result->GetStatistics({"distance_evaluations", "missing"});
    REQUIRE(selected[0] == parsed["distance_evaluations"].Dump());
    REQUIRE(selected[1].empty());
    {
        auto resource = std::make_shared<vsag::Resource>();
        auto allocator = resource->GetAllocator();
        vsag::SearchRequest request;
        request.query_ = query;
        request.topk_ = 3;
        request.params_str_ = R"({"hgraph":{"ef_search":16}})";
        request.search_allocator_ = allocator.get();
        auto custom = index->SearchWithRequest(request);
        REQUIRE(custom.has_value());
        REQUIRE(custom.value()->GetIds()[0] == 7);
        REQUIRE(vsag::JsonType::Parse(custom.value()->GetStatistics())["distance_evaluations"]
                    .GetUint64() > 0);
    }
    {
        query->NumElements(2);
        auto batch = index->KnnSearch(query, 3, R"({"hgraph":{"ef_search":16}})");
        REQUIRE(batch.has_value());
        REQUIRE(batch.value()->GetNumElements() == 2);
        REQUIRE(vsag::JsonType::Parse(batch.value()->GetStatistics()).Contains("batch_routes"));
        query->NumElements(1);
    }
    query->Float32Vectors(vectors.data() + 19 * 4);
    REQUIRE(index->KnnSearch(query, 3, R"({"hgraph":{"ef_search":16}})").has_value());
    index.reset();
    REQUIRE(result->GetStatistics() == snapshot);
    REQUIRE(result->GetIds()[0] == 7);
    std::atomic<bool> consistent{true};
    auto read = [&]() {
        for (uint64_t i = 0; i < 20; ++i) {
            if (result->GetStatistics() != snapshot) {
                consistent.store(false);
            }
        }
    };
    std::thread first(read), second(read);
    first.join();
    second.join();
    REQUIRE(consistent.load());
    result->Statistics(R"({"manual":7})");
    REQUIRE(result->GetStatistics() == R"({"manual":7})");
    REQUIRE(result->GetStatistics({"manual"})[0] == "7");
}
