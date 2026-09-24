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

#include "graph_build_helper.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace vsag {
namespace {
struct ReplayFixture {
    GraphBuildProgress progress;
    std::vector<int> events;
    int fail_step{-1};
    int duplicate{-1};
    int entry{-1};
    bool update_entry{false};

    void
    Step(int step, int id) {
        if (step == fail_step) {
            throw std::runtime_error("injected candidate failure");
        }
        events.push_back(10 * id + step);
    }

    struct State {
        int id;
        uint64_t processed;
        State(int id, uint64_t processed) : id(id), processed(processed) {
        }
        State(const State&) = delete;
        State&
        operator=(const State&) = delete;
    };

    State
    MakeState(int id, uint64_t processed) {
        return State(id, processed);
    }
    bool
    Prepare(State& state) {
        Step(0, state.id);
        if (state.processed == 0) {
            entry = state.id;
            return true;
        }
        return false;
    }
    int
    Search(State& state) {
        Step(1, state.id);
        return state.id;
    }
    bool
    AcceptDuplicate(State& state, int& candidates) {
        REQUIRE(candidates == state.id);
        Step(2, state.id);
        return state.id == duplicate;
    }
    void
    Connect(State& state, int& candidates) {
        REQUIRE(candidates == state.id);
        Step(3, state.id);
    }
    void
    Finish(State& state) {
        Step(4, state.id);
        if (update_entry) {
            entry = state.id;
        }
    }
    void
    Append(const std::vector<int>& ids) {
        GraphBuildHelper::AppendIds(ids, progress, *this);
    }
};

struct CandidateIndex {
    bool initialized{false};
    bool fail{false};
    bool reject{false};
    const int* input{nullptr};

    void
    InitFeatures() {
        initialized = true;
    }

    std::vector<int64_t>
    Build(const int& dataset) {
        REQUIRE(initialized);
        input = &dataset;
        if (fail) {
            throw std::runtime_error("build failure");
        }
        return reject ? std::vector<int64_t>{17} : std::vector<int64_t>{};
    }
};
}  // namespace

TEST_CASE("GraphBuildHelper batch replay preserves order and sparse IDs", "[graph_build_helper]") {
    ReplayFixture whole;
    ReplayFixture batches;
    whole.Append({});
    REQUIRE(whole.progress.processed_members == 0);
    whole.Append({3, 90, 7, 42});
    batches.Append({3});
    batches.Append({90, 7});
    batches.Append({});
    batches.Append({42});
    REQUIRE(whole.events == batches.events);
    REQUIRE(
        whole.events ==
        std::vector<int>{30, 900, 901, 902, 903, 904, 70, 71, 72, 73, 74, 420, 421, 422, 423, 424});
    REQUIRE(whole.progress.processed_members == 4);
    REQUIRE(batches.progress.processed_members == 4);
    REQUIRE(whole.entry == 3);
}

TEST_CASE("GraphBuildHelper prepared insertion matches candidate replay", "[graph_build_helper]") {
    ReplayFixture replay;
    ReplayFixture online;
    replay.update_entry = online.update_entry = true;
    replay.Append({4, 19});
    auto first = online.MakeState(4, 0);
    REQUIRE(online.Prepare(first));
    auto next = online.MakeState(19, 1);
    REQUIRE_FALSE(online.Prepare(next));
    // The owner could release its graph lock here without rerunning preparation.
    GraphBuildHelper::RunPrepared(online, next);
    REQUIRE(online.events == replay.events);
    REQUIRE(online.entry == replay.entry);
    REQUIRE(online.progress.processed_members == 0);
}

TEST_CASE("GraphBuildHelper duplicates count as completed members", "[graph_build_helper]") {
    ReplayFixture fixture;
    fixture.duplicate = 1;
    fixture.update_entry = true;
    fixture.Append({0, 1});
    REQUIRE(fixture.progress.processed_members == 2);
    REQUIRE(fixture.entry == 0);
    REQUIRE(fixture.events == std::vector<int>{0, 10, 11, 12});
    fixture.Append({2});
    REQUIRE(fixture.entry == 2);
    REQUIRE(fixture.progress.processed_members == 3);
}

TEST_CASE("GraphBuildHelper does not finish failed members", "[graph_build_helper]") {
    for (int fail_step = 0; fail_step <= 4; ++fail_step) {
        ReplayFixture fixture;
        fixture.Append({3});
        fixture.fail_step = fail_step;
        bool published = false;
        REQUIRE_THROWS_AS(([&] {
                              fixture.Append({9, 17});
                              published = true;
                          }()),
                          std::runtime_error);
        REQUIRE_FALSE(published);
        REQUIRE(fixture.progress.processed_members == 1);
        REQUIRE(fixture.events.size() == static_cast<uint64_t>(1 + fail_step));
    }
}

TEST_CASE("GraphBuildHelper complete index validates before publication", "[graph_build_helper]") {
    const int dataset = 42;
    std::weak_ptr<CandidateIndex> candidate;
    for (int mode = 0; mode < 3; ++mode) {
        auto factory = [&] {
            auto index = std::make_shared<CandidateIndex>();
            index->fail = mode == 1;
            index->reject = mode == 2;
            candidate = index;
            return index;
        };
        if (mode == 0) {
            auto result = GraphBuildHelper::BuildIndexCandidate(factory, dataset, "rejected ids");
            REQUIRE(result->initialized);
            REQUIRE(result->input == &dataset);
            REQUIRE_FALSE(candidate.expired());
        } else {
            REQUIRE_THROWS(GraphBuildHelper::BuildIndexCandidate(factory, dataset, "rejected ids"));
        }
        REQUIRE(candidate.expired());
    }
    auto throwing_factory = []() -> std::shared_ptr<CandidateIndex> { throw std::bad_alloc(); };
    REQUIRE_THROWS_AS(
        GraphBuildHelper::BuildIndexCandidate(throwing_factory, dataset, "rejected ids"),
        std::bad_alloc);
}

}  // namespace vsag
