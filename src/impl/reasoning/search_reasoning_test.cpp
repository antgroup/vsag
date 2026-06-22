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
#include <string_view>

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

    SECTION("InitializeExpectedTargets") {
        ctx.InitializeExpectedTargets(labels, label_to_inner_id);
        REQUIRE(ctx.expected_traces_.size() == 2);
        REQUIRE(ctx.expected_inner_ids_.size() == 2);

        auto it1 = ctx.expected_traces_.find(0);
        REQUIRE(it1 != ctx.expected_traces_.end());
        REQUIRE(it1.value().label == 100);
        REQUIRE(it1.value().true_distance == 0.0F);

        auto it2 = ctx.expected_traces_.find(1);
        REQUIRE(it2 != ctx.expected_traces_.end());
        REQUIRE(it2.value().label == 200);
        REQUIRE(it2.value().true_distance == 0.0F);
    }
}

TEST_CASE("ReasoningContext event recording", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

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

    SECTION("RecordFilterReject keeps visited hop") {
        ctx.RecordVisit(0, 0.2F, 3);
        ctx.RecordFilterReject(0);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().visited_at_hop == 3);
    }

    SECTION("RecordReorder") {
        ctx.RecordVisit(0, 0.3F, 1);
        ctx.RecordReorder(0, 0.3F, 0.5F);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().reorder_evicted == false);
        REQUIRE(it.value().quantized_distance == 0.3F);
        REQUIRE(it.value().true_distance == 0.5F);
        REQUIRE(ctx.reorder_changes_.size() == 1);
    }

    SECTION("RecordReorder ignores non-expected target") {
        ctx.RecordReorder(99, 0.3F, 0.5F);
        REQUIRE(ctx.reorder_changes_.empty());
    }

    SECTION("RecordReorderEviction") {
        ctx.RecordReorderEviction(0, 2);
        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it != ctx.expected_traces_.end());
        REQUIRE(it.value().reorder_evicted == true);
        REQUIRE(it.value().was_visited == true);
        REQUIRE(it.value().visited_at_hop == 2);
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

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

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
        ctx.RecordReorderEviction(6, 2);
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

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);
    ctx.SetTrueDistance(0, 0.0F);
    ctx.SetTrueDistance(1, 1.4142135F);

    ctx.RecordVisit(0, 0.0F, 1);
    Vector<InnerIdType> result_ids(&allocator);
    result_ids.push_back(0);
    ctx.MarkResult(result_ids);

    ctx.DiagnoseExpectedTargets();

    std::string report = ctx.GenerateReport();
    REQUIRE(report.find("expected_analysis") != std::string::npos);
    REQUIRE(report.find("1/2") != std::string::npos);
    REQUIRE(report.find("1 missed") != std::string::npos);
    REQUIRE(report.find("missed_targets") != std::string::npos);
    REQUIRE(report.find("not_reachable") != std::string::npos);
    REQUIRE(report.find("1.4142135") != std::string::npos);
}

TEST_CASE("ReasoningContext termination reasons are centralized", "[reasoning]") {
    REQUIRE(std::string_view(ReasoningContext::kTerminationLowerBoundReached) ==
            "lower_bound_reached");
    REQUIRE(std::string_view(ReasoningContext::kTerminationHopsLimitReached) ==
            "hops_limit_reached");
    REQUIRE(std::string_view(ReasoningContext::kTerminationTimeout) == "timeout");
}

TEST_CASE("ReasoningContext SetTermination", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    SECTION("SetTermination stores reason") {
        REQUIRE(ctx.termination_reason_.empty());
        ctx.SetTermination(ReasoningContext::kTerminationLowerBoundReached);
        REQUIRE(ctx.termination_reason_ == "lower_bound_reached");
    }

    SECTION("SetTermination can be overwritten") {
        ctx.SetTermination(ReasoningContext::kTerminationHopsLimitReached);
        REQUIRE(ctx.termination_reason_ == "hops_limit_reached");
        ctx.SetTermination(ReasoningContext::kTerminationTimeout);
        REQUIRE(ctx.termination_reason_ == "timeout");
    }
}

TEST_CASE("ReasoningContext MarkResult", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);
    labels.push_back(200);
    labels.push_back(300);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;
    label_to_inner_id[200] = 1;
    label_to_inner_id[300] = 2;

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

    SECTION("MarkResult marks matching targets") {
        Vector<InnerIdType> result_ids(&allocator);
        result_ids.push_back(0);
        result_ids.push_back(2);
        ctx.MarkResult(result_ids);

        REQUIRE(ctx.expected_traces_.find(0).value().was_in_result_set == true);
        REQUIRE(ctx.expected_traces_.find(1).value().was_in_result_set == false);
        REQUIRE(ctx.expected_traces_.find(2).value().was_in_result_set == true);
    }

    SECTION("MarkResult ignores non-expected IDs") {
        Vector<InnerIdType> result_ids(&allocator);
        result_ids.push_back(99);
        result_ids.push_back(0);
        ctx.MarkResult(result_ids);

        REQUIRE(ctx.expected_traces_.find(0).value().was_in_result_set == true);
        REQUIRE(ctx.expected_traces_.find(1).value().was_in_result_set == false);
    }
}

TEST_CASE("ReasoningContext SetTrueDistance", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

    SECTION("SetTrueDistance updates expected target") {
        ctx.SetTrueDistance(0, 1.5F);
        REQUIRE(ctx.expected_traces_.find(0).value().true_distance == 1.5F);
    }

    SECTION("SetTrueDistance ignores non-expected ID") {
        ctx.SetTrueDistance(99, 2.0F);
        REQUIRE(ctx.expected_traces_.find(0).value().true_distance == 0.0F);
    }
}

TEST_CASE("ReasoningContext RecordVisit preserves first distance", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

    SECTION("Preserves non-zero first distance") {
        ctx.RecordVisit(0, 0.3F, 1);
        ctx.RecordVisit(0, 0.5F, 2);

        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it.value().quantized_distance == 0.3F);
        REQUIRE(it.value().visited_at_hop == 2);
    }

    SECTION("Zero distance sentinel: second visit overwrites zero first distance") {
        // Known limitation: RecordVisit uses 0.0F as sentinel for "not yet set",
        // so a true zero distance gets overwritten by the next visit.
        ctx.RecordVisit(0, 0.0F, 1);
        ctx.RecordVisit(0, 0.5F, 2);

        auto it = ctx.expected_traces_.find(0);
        REQUIRE(it.value().quantized_distance == 0.5F);
        REQUIRE(it.value().visited_at_hop == 2);
    }
}

TEST_CASE("ReasoningContext RecordEviction preserves earlier visit", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

    ctx.RecordVisit(0, 0.5F, 1);
    ctx.RecordEviction(0, 3);

    auto it = ctx.expected_traces_.find(0);
    REQUIRE(it.value().was_evicted == true);
    REQUIRE(it.value().visited_at_hop == 1);
}

TEST_CASE("ReasoningContext GenerateReport found count", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);
    labels.push_back(200);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;
    label_to_inner_id[200] = 1;

    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

    ctx.RecordVisit(0, 0.0F, 1);
    ctx.RecordVisit(1, 0.1F, 2);

    Vector<InnerIdType> result_ids(&allocator);
    result_ids.push_back(0);
    result_ids.push_back(1);
    ctx.MarkResult(result_ids);
    ctx.DiagnoseExpectedTargets();

    std::string report = ctx.GenerateReport();
    REQUIRE(report.find("2/2") != std::string::npos);
    REQUIRE(report.find("0 missed") != std::string::npos);
}

}  // namespace vsag
