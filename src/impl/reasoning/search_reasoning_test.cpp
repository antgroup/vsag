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

#include "search_reasoning.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/default_allocator.h"

namespace vsag {

TEST_CASE("ReasoningContext basic operations", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    SECTION("Initialization") {
        REQUIRE(ctx.topk_ == 0);
        REQUIRE(ctx.total_hops_ == 0);
        REQUIRE(ctx.total_dist_computations_ == 0);
    }

    SECTION("SetSearchParams") {
        ctx.SetSearchParams(10, "hgraph", true, false);
        REQUIRE(ctx.topk_ == 10);
        REQUIRE(ctx.index_type_ == "hgraph");
        REQUIRE(ctx.use_reorder_ == true);
        REQUIRE(ctx.filter_active_ == false);
    }

    SECTION("AddSearchHop and AddDistanceComputation") {
        ctx.AddSearchHop();
        ctx.AddSearchHop();
        REQUIRE(ctx.total_hops_ == 2);

        ctx.AddDistanceComputation(5);
        ctx.AddDistanceComputation(3);
        REQUIRE(ctx.total_dist_computations_ == 8);
    }
}

TEST_CASE("ReasoningContext with expected targets", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);
    labels.push_back(200);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;
    label_to_inner_id[200] = 1;

    float query[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    float vectors[2][4] = {{1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}};

    SECTION("InitializeExpectedTargets") {
        ctx.InitializeExpectedTargets(
            labels, label_to_inner_id, query, vectors, DataTypes::DATA_TYPE_FLOAT, 4);
        REQUIRE(ctx.expected_traces_.size() == 2);
        REQUIRE(ctx.expected_inner_ids_.size() == 2);

        auto it1 = ctx.expected_traces_.find(0);
        REQUIRE(it1 != ctx.expected_traces_.end());
        REQUIRE(it1.value().label == 100);
        REQUIRE(it1.value().true_distance == 0.0F);

        auto it2 = ctx.expected_traces_.find(1);
        REQUIRE(it2 != ctx.expected_traces_.end());
        REQUIRE(it2.value().label == 200);
        REQUIRE(it2.value().true_distance > 0.0F);
    }
}

TEST_CASE("ReasoningContext event recording", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;

    float query[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    float vectors[1][4] = {{1.0F, 0.0F, 0.0F, 0.0F}};
    ctx.InitializeExpectedTargets(
        labels, label_to_inner_id, query, vectors, DataTypes::DATA_TYPE_FLOAT, 4);

    SECTION("RecordVisit") {
        ctx.RecordVisit(0, 0.5F, 1);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().was_visited == true);
        REQUIRE(it.value().visited_at_hop == 1);
        REQUIRE(it.value().quantized_distance == 0.5F);
    }

    SECTION("RecordEviction") {
        ctx.RecordEviction(0, 2);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().was_evicted == true);
        REQUIRE(it.value().was_visited == true);
        REQUIRE(it.value().visited_at_hop == 2);
    }

    SECTION("RecordFilterReject") {
        ctx.RecordFilterReject(0);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().filter_rejected == true);
        REQUIRE(it.value().was_visited == true);
    }

    SECTION("RecordReorder") {
        ctx.RecordVisit(0, 0.3F, 1);
        ctx.RecordReorder(0, 0.3F, 0.5F);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().reorder_evicted == true);
        REQUIRE(it.value().quantized_distance == 0.3F);
        REQUIRE(it.value().true_distance == 0.5F);
        REQUIRE(ctx.reorder_changes_.size() == 1);
    }
}

TEST_CASE("ReasoningContext diagnosis logic", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);
    labels.push_back(200);
    labels.push_back(300);
    labels.push_back(400);
    labels.push_back(500);
    labels.push_back(600);
    labels.push_back(700);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;
    label_to_inner_id[200] = 1;
    label_to_inner_id[300] = 2;
    label_to_inner_id[400] = 3;
    label_to_inner_id[500] = 4;
    label_to_inner_id[600] = 5;
    label_to_inner_id[700] = 6;

    float query[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    float vectors[7][4] = {
        {1.0F, 0.0F, 0.0F, 0.0F},
        {0.5F, 0.5F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
        {0.5F, 0.5F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
    };
    ctx.InitializeExpectedTargets(
        labels, label_to_inner_id, query, vectors, DataTypes::DATA_TYPE_FLOAT, 4);

    SECTION("Diagnose: not_reachable") {
        ctx.DiagnoseExpectedTargets();
        auto it = ctx.expected_traces_.find(4);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().diagnosis == "not_reachable");
    }

    SECTION("Diagnose: filter_rejected") {
        ctx.RecordFilterReject(2);
        ctx.DiagnoseExpectedTargets();
        auto it = ctx.expected_traces_.find(2);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().diagnosis == "filter_rejected");
    }

    SECTION("Diagnose: ef_too_small") {
        ctx.RecordVisit(3, 0.5F, 1);
        ctx.RecordEviction(3, 3);
        ctx.DiagnoseExpectedTargets();
        auto it = ctx.expected_traces_.find(3);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().diagnosis == "ef_too_small");
    }

    SECTION("Diagnose: quantization_error") {
        auto it = ctx.expected_traces_.find(5);
        REQUIRE(it != ctx.expected_traces_.end());
        it.value().true_distance = 0.5F;
        it.value().quantized_distance = 1.0F;
        it.value().was_visited = true;
        ctx.DiagnoseExpectedTargets();
        REQUIRE(it.value().diagnosis == "quantization_error");
    }

    SECTION("Diagnose: reorder_evicted") {
        ctx.RecordVisit(6, 0.3F, 1);
        ctx.RecordReorder(6, 0.3F, 0.8F);
        ctx.DiagnoseExpectedTargets();
        auto it = ctx.expected_traces_.find(6);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().diagnosis == "reorder_evicted");
    }

    SECTION("Diagnose: success") {
        ctx.RecordVisit(0, 0.0F, 1);
        Vector<InnerIdType> result_ids(&allocator);
        result_ids.push_back(0);
        ctx.MarkResult(result_ids);
        ctx.DiagnoseExpectedTargets();
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().diagnosis == "success");
    }
}

TEST_CASE("ReasoningContext GenerateReport", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);
    labels.push_back(200);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;
    label_to_inner_id[200] = 1;

    float query[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    float vectors[2][4] = {{1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}};
    ctx.InitializeExpectedTargets(
        labels, label_to_inner_id, query, vectors, DataTypes::DATA_TYPE_FLOAT, 4);

    ctx.RecordVisit(0, 0.0F, 1);
    Vector<InnerIdType> result_ids(&allocator);
    result_ids.push_back(0);
    ctx.MarkResult(result_ids);

    ctx.DiagnoseExpectedTargets();

    std::string report = ctx.GenerateReport();
    REQUIRE(report.find("expected_analysis") != std::string::npos);
    REQUIRE(report.find("1/2") != std::string::npos);
    REQUIRE(report.find("1 missed") != std::string::npos);
}

}  // namespace vsag
