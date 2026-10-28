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

#include "test_budget.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifndef VSAG_TEST_BUDGET_STANDALONE
#include <catch2/catch_test_macros.hpp>
#endif

namespace {

using namespace fixtures::budget;

void
Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception = std::invalid_argument>
void
CheckThrows(const std::function<void()>& function) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    Check(caught, "expected budget exception was not thrown");
}

CaseIdentity
Identity() {
    return {"budget core fixture", "[ut][budget]", "tests/fixtures/core/test_budget_test.cpp", 1};
}

RunSpec
Run(const std::string& key, uint64_t num) {
    RunSpec result;
    result.key = key;
    result.workload.num = num;
    return result;
}

void
TestPhases() {
    Workload work;
    work.dim = 2;
    work.num = 3;
    work.query_num = 2;
    work.train_num = 5;
    work.metric = "cosine";
    work.train_runs = work.build_runs = work.search_runs = work.oracle_runs = 2;
    work.serialize_runs = 2;
    work.train_iterations = 3;
    work.train_centroids = 2;
    work.build_candidates = 4;
    work.search_candidates = 3;
    work.update_num = 4;
    work.repetitions = 2;
    const auto cost = EstimateCost(work);
    Check(cost.data == 20 && cost.train == 240 && cost.build == 288 && cost.search == 144 &&
              cost.oracle == 144 && cost.serialize == 24 && cost.update == 192,
          "phase arithmetic differs from the declared work");
    Check(cost.Total() == 1052, "phase ledger did not sum all work");
    work.threads = 128;
    Check(EstimateCost(work).Total() == cost.Total(), "threads must not discount work");
    work.quantizer = "pq";
    Check(EstimateCost(work).train == cost.train * 8, "quantizer training weight not charged");
    Check(QuantizerWeight("rabitq,fp32,block_memory_io,32,8") >
              QuantizerWeight("rabitq,fp32,block_memory_io,4,1"),
          "RaBitQ bit widths not charged");
    Check(QuantizerWeight("sq4->sq4,fp32") == 5, "Tune transition must charge both formats");
    Check(QuantizerWeight("pqfs,fp16") == 9, "PQ fast-scan configuration should retain PQ work");
}

void
TestCaseCap() {
    Policy policy{"small", 25, 47};
    const auto plan =
        PlanCase(Identity(), policy, {{Run("a", 10), Run("b", 10)}, {Run("a", 20), Run("b", 20)}});
    Check(plan.TotalCost().Total() == 20 && plan.SelectedCandidate() == 0,
          "cap must apply to the full case");
    Check(plan.Runs().size() == 2 && plan.Runs()[1].workload.num == 10,
          "required variant lost during budget planning");
    policy.cap_points = 40;
    Check(PlanCase(Identity(), policy, {{Run("a", 10), Run("b", 10)}, {Run("a", 20), Run("b", 20)}})
                  .SelectedCandidate() == 1,
          "largest complete candidate fitting the cap should be selected");
    policy.cap_points = 15;
    CheckThrows([&] { PlanCase(Identity(), policy, {{Run("a", 10), Run("b", 10)}}); });
}

void
TestMinimums() {
    Workload work;
    work.dim = 959;
    work.min_dim = 960;
    CheckThrows([&] { EstimateCost(work); });
    work.dim = 960;
    work.num = 255;
    work.min_num = 256;
    CheckThrows([&] { EstimateCost(work); });
    work.num = 256;
    work.train_num = 255;
    work.min_train_num = 256;
    CheckThrows([&] { EstimateCost(work); });
    work.train_num = 256;
    Check(EstimateCost(work).Total() == 245760, "legal minimum should be accepted unchanged");
    // Low-dimensional RaBitQ is legal outside recall tests; the core must not invent a 960 rule.
    work = Workload{};
    work.quantizer = "rabitq";
    Check(EstimateCost(work).Total() == 1, "model-specific minimum must remain caller-defined");
    work.train_runs = 1;
    CheckThrows([&] { EstimateCost(work); });
    work.train_runs = 0;
    work.search_runs = 1;
    CheckThrows([&] { EstimateCost(work); });
}

void
TestCoverage() {
    const auto first = Run("required", 10);
    auto changed = Run("required", 20);
    changed.workload.metric = "cosine";
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{first}, {changed}}); });
    changed = Run("required", 20);
    changed.workload.quantizer = "sq8";
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{first}, {changed}}); });
    changed = Run("required", 20);
    changed.workload.data_runs = 2;
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{first}, {changed}}); });
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{first}, {}}); });
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{first, first}}); });
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{Run("a", 20)}, {Run("a", 10)}}); });
    auto larger = first;
    larger.workload.num = 100;
    larger.workload.dim = 2;
    auto traded = larger;
    traded.workload.num = 1000;
    traded.workload.dim = 1;
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{larger}, {traded}}); });
    auto invalid = first;
    invalid.workload.num = 0;
    // Invalid expensive/unselected candidates are not hidden by an earlier fitting candidate.
    CheckThrows([&] { PlanCase(Identity(), Policy{}, {{first}, {invalid}}); });
}

void
TestOverflow() {
    Cost total;
    total.data = std::numeric_limits<uint64_t>::max();
    Cost increment;
    increment.train = 1;
    CheckThrows<std::overflow_error>([&] { total += increment; });
    Check(total.train == 0, "failed ledger update must be atomic");
    total.train = 1;
    CheckThrows<std::overflow_error>([&] { total.Total(); });
    Workload work;
    work.dim = std::numeric_limits<uint64_t>::max();
    work.num = 2;
    CheckThrows<std::overflow_error>([&] { EstimateCost(work); });
    auto first = Run("a", 1);
    auto overflow = first;
    overflow.workload.num = std::numeric_limits<uint64_t>::max();
    overflow.workload.dim = 2;
    CheckThrows<std::overflow_error>([&] {
        PlanCase(Identity(), Policy{}, {{first}, {overflow}});
    });
}

void
TestStableSelection() {
    const auto seed = CaseSeed("case-a", 424242);
    Check(seed == 8558749683471766968ULL, "stable seed contract changed");
    Check(seed == CaseSeed("case-a", 424242), "case seed is not stable");
    Check(seed != CaseSeed("case-b", 424242) && seed != CaseSeed("case-a", 424243),
          "case/global seed must both affect selection");
    const auto indices = ShuffledIndices(20, seed);
    Check(indices == ShuffledIndices(20, seed), "shuffle depends on process order");
    auto sorted = indices;
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t index = 0; index < sorted.size(); ++index) {
        Check(sorted[index] == index, "shuffle lost or duplicated an index");
    }
    Check(ShuffledIndices(0, seed).empty(), "empty shuffle is invalid");
    Check(SampleInclusive(seed, 9, 9) == 9, "singleton sample should be exact");
    for (uint64_t index = 0; index < 100; ++index) {
        const auto value = SampleInclusive(index, 100, 200);
        Check(value >= 100 && value <= 200, "sample escaped its inclusive interval");
    }
    SampleInclusive(seed, 0, std::numeric_limits<uint64_t>::max());
    CheckThrows([&] { SampleInclusive(seed, 2, 1); });
}

void
TestInvalidInputs() {
    CheckThrows([] { MetricWeight("unknown"); });
    CheckThrows([] { QuantizerWeight(""); });
    CheckThrows([] { QuantizerWeight("fp32,"); });
    CheckThrows([] { QuantizerWeight("fp32,unknown"); });
    CheckThrows([] { QuantizerWeight("rabitq,fp32,block_memory_io,0,1"); });
    CheckThrows([] { QuantizerWeight("rabitq,fp32,block_memory_io,x,1"); });
    CheckThrows([] { QuantizerWeight("sq8,fp32,unsupported"); });
    CheckThrows([] { PlanCase(Identity(), Policy{}, {}); });
    CheckThrows([] { PlanCase(CaseIdentity{}, Policy{}, {{Run("a", 1)}}); });
    auto policy = Policy{};
    policy.cap_points = 0;
    CheckThrows([&] { PlanCase(Identity(), policy, {{Run("a", 1)}}); });
    auto work = Workload{};
    work.threads = 0;
    CheckThrows([&] { EstimateCost(work); });
    work = Workload{};
    work.data_runs = 0;
    CheckThrows([&] { EstimateCost(work); });
}

void
TestLog() {
    auto identity = Identity();
    identity.name = "name\"with\nnewline";
    auto policy = Policy::Daily();
    auto plan = PlanCase(identity, policy, {{Run("a", 10), Run("b", 20)}});
    std::ostringstream output;
    plan.Log(output);
    const auto line = output.str();
    Check(line.find("[test-budget] {\"schema_version\":1") == 0, "missing log schema");
    Check(line.find("\"unit\":\"logical_points\"") != std::string::npos,
          "log must not claim seconds");
    Check(line.find("\"total_points\":30") != std::string::npos, "log lost cumulative cost");
    Check(line.find("name\\\"with\\u000anewline") != std::string::npos, "invalid JSON escaping");
    Check(std::count(line.begin(), line.end(), '\n') == 1, "one plan should be one log record");
    Check(plan.Cap() == policy.cap_points && plan.Seed() == CaseSeed(identity.name, policy.seed),
          "plan policy identity not retained");
}

}  // namespace

#ifdef VSAG_TEST_BUDGET_STANDALONE
int
main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests{
        {"phases", TestPhases},
        {"case cap", TestCaseCap},
        {"minimums", TestMinimums},
        {"coverage", TestCoverage},
        {"overflow", TestOverflow},
        {"stable selection", TestStableSelection},
        {"invalid inputs", TestInvalidInputs},
        {"log", TestLog}};
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
#else
TEST_CASE("Test budget accounts for every work phase", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestPhases());
}

TEST_CASE("Test budget cap covers the complete testcase", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestCaseCap());
}

TEST_CASE("Test budget preserves model-specific legal minimums", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestMinimums());
}

TEST_CASE("Test budget candidates preserve required coverage", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestCoverage());
}

TEST_CASE("Test budget ledger rejects overflow atomically", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestOverflow());
}

TEST_CASE("Test budget selection has reproducible case seeds", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestStableSelection());
}

TEST_CASE("Test budget rejects invalid inputs", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestInvalidInputs());
}

TEST_CASE("Test budget emits one complete logical-cost record", "[ut][test_budget]") {
    REQUIRE_NOTHROW(TestLog());
}
#endif
