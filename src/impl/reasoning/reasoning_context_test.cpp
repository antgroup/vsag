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

#include "reasoning_context.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/default_allocator.h"

namespace vsag {

TEST_CASE("ReasoningContext basic operations", "[reasoning]") {
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
    REQUIRE(ctx.expected_traces_.size() == 2);
}

TEST_CASE("ReasoningContext GenerateReport meta section", "[reasoning]") {
    DefaultAllocator allocator;
    ReasoningContext ctx(&allocator);

    Vector<int64_t> labels(&allocator);
    labels.push_back(100);

    UnorderedMap<int64_t, InnerIdType> label_to_inner_id(&allocator);
    label_to_inner_id[100] = 0;
    ctx.InitializeExpectedTargets(labels, label_to_inner_id);

    Vector<InnerIdType> result_ids(&allocator);
    result_ids.push_back(0);
    ctx.MarkResult(result_ids);
    ctx.DiagnoseExpectedTargets();

    ctx.SetSearchParams(10, "HGraph", true, false, false);
    ctx.SetTermination(ReasoningTermination::kHopsLimitReached);

    std::string report = ctx.GenerateReport();
    REQUIRE(report.find("1/1") != std::string::npos);
    REQUIRE(report.find("0 missed") != std::string::npos);
    REQUIRE(report.find("hops_limit_reached") != std::string::npos);
    REQUIRE(report.find("HGraph") != std::string::npos);
}

TEST_CASE("ReasoningContext MakeStatusReport", "[reasoning]") {
    std::string report =
        ReasoningContext::MakeStatusReport(ReasoningReportStatus::kEmptyIndex, "SINDI_V2");
    REQUIRE(report.find("empty_index") != std::string::npos);
    REQUIRE(report.find("SINDI_V2") != std::string::npos);
}

}  // namespace vsag