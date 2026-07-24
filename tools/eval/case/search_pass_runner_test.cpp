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

#include "search_pass_runner.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vsag::eval {

namespace {

struct trace_state {
    std::vector<std::string> events;
    std::vector<std::string> query_passes;
    std::vector<bool> memory_active_during_query;
    bool memory_active{false};
    bool memory_active_during_statistics{false};
    uint64_t statistics_pass_count{0};
};

class TracingMonitor : public Monitor {
public:
    TracingMonitor(std::string name, trace_state& trace, bool is_memory = false)
        : Monitor(std::move(name)), trace_(trace), is_memory_(is_memory) {
    }

    void
    Start() override {
        trace_.events.emplace_back(name_ + ".start");
        if (is_memory_) {
            trace_.memory_active = true;
        }
    }

    void
    Stop() override {
        trace_.events.emplace_back(name_ + ".stop");
        if (is_memory_) {
            trace_.memory_active = false;
        }
    }

    JsonType
    GetResult() override {
        return {};
    }

    void
    Record(void* input = nullptr) override {
        static_cast<void>(input);
        trace_.events.emplace_back(name_ + ".record");
    }

private:
    trace_state& trace_;
    bool is_memory_;
};

trace_state
run_standard_schedule(bool enable_memory) {
    trace_state trace;
    auto latency = std::make_shared<TracingMonitor>("latency", trace);
    auto recall = std::make_shared<TracingMonitor>("recall", trace);
    MonitorPtr memory = nullptr;
    if (enable_memory) {
        memory = std::make_shared<TracingMonitor>("memory", trace, true);
    }

    SearchPassRunner::Run(
        {latency, recall},
        latency,
        memory,
        [&trace](const MonitorPtr& monitor) {
            trace.query_passes.emplace_back(monitor->GetName());
            trace.memory_active_during_query.emplace_back(trace.memory_active);
            trace.events.emplace_back("query." + monitor->GetName());
            monitor->Record();
        },
        [&trace]() {
            trace.events.emplace_back("statistics");
            trace.memory_active_during_statistics = trace.memory_active;
            ++trace.statistics_pass_count;
        });
    return trace;
}

}  // namespace

TEST_CASE("SearchPassRunner keeps ordinary passes stable when memory is toggled",
          "[ut][eval][search_pass]") {
    const auto without_memory = run_standard_schedule(false);
    const auto with_memory = run_standard_schedule(true);

    const std::vector<std::string> expected_query_passes = {"latency", "recall"};
    const std::vector<bool> expected_without_memory_activity = {false, false};
    const std::vector<bool> expected_with_memory_activity = {true, false};
    CHECK(without_memory.query_passes == expected_query_passes);
    CHECK(with_memory.query_passes == expected_query_passes);
    CHECK(without_memory.statistics_pass_count == 1);
    CHECK(with_memory.statistics_pass_count == 1);

    CHECK(without_memory.memory_active_during_query == expected_without_memory_activity);
    CHECK(with_memory.memory_active_during_query == expected_with_memory_activity);
    CHECK_FALSE(without_memory.memory_active_during_statistics);
    CHECK_FALSE(with_memory.memory_active_during_statistics);

    const std::vector<std::string> expected_without_memory = {
        "latency.start",
        "query.latency",
        "latency.record",
        "latency.stop",
        "recall.start",
        "query.recall",
        "recall.record",
        "recall.stop",
        "statistics",
    };
    const std::vector<std::string> expected_with_memory = {
        "latency.start",
        "memory.start",
        "query.latency",
        "latency.record",
        "memory.stop",
        "latency.stop",
        "recall.start",
        "query.recall",
        "recall.record",
        "recall.stop",
        "statistics",
    };
    CHECK(without_memory.events == expected_without_memory);
    CHECK(with_memory.events == expected_with_memory);
}

TEST_CASE("SearchPassRunner uses a memory-only pass when latency is disabled",
          "[ut][eval][search_pass]") {
    trace_state trace;
    auto recall = std::make_shared<TracingMonitor>("recall", trace);
    auto memory = std::make_shared<TracingMonitor>("memory", trace, true);

    SearchPassRunner::Run(
        {recall},
        nullptr,
        memory,
        [&trace](const MonitorPtr& monitor) {
            const std::string pass_name = monitor == nullptr ? "memory-only" : monitor->GetName();
            trace.query_passes.emplace_back(pass_name);
            trace.memory_active_during_query.emplace_back(trace.memory_active);
            trace.events.emplace_back("query." + pass_name);
            if (monitor != nullptr) {
                monitor->Record();
            }
        },
        [&trace]() {
            trace.events.emplace_back("statistics");
            trace.memory_active_during_statistics = trace.memory_active;
            ++trace.statistics_pass_count;
        });

    const std::vector<std::string> expected_events = {
        "memory.start",
        "query.memory-only",
        "memory.stop",
        "recall.start",
        "query.recall",
        "recall.record",
        "recall.stop",
        "statistics",
    };
    const std::vector<std::string> expected_query_passes = {"memory-only", "recall"};
    const std::vector<bool> expected_memory_activity = {true, false};
    CHECK(trace.events == expected_events);
    CHECK(trace.query_passes == expected_query_passes);
    CHECK(trace.memory_active_during_query == expected_memory_activity);
    CHECK_FALSE(trace.memory_active_during_statistics);
    CHECK(trace.statistics_pass_count == 1);
}

TEST_CASE("SearchPassRunner unwinds memory before the ordinary monitor on failure",
          "[ut][eval][search_pass]") {
    trace_state trace;
    auto latency = std::make_shared<TracingMonitor>("latency", trace);
    auto recall = std::make_shared<TracingMonitor>("recall", trace);
    auto memory = std::make_shared<TracingMonitor>("memory", trace, true);

    CHECK_THROWS_AS(SearchPassRunner::Run(
                        {latency, recall},
                        latency,
                        memory,
                        [&trace](const MonitorPtr& monitor) {
                            trace.events.emplace_back("query." + monitor->GetName());
                            throw std::runtime_error("injected search failure");
                        },
                        [&trace]() {
                            trace.events.emplace_back("statistics");
                            ++trace.statistics_pass_count;
                        }),
                    std::runtime_error);

    const std::vector<std::string> expected_events = {
        "latency.start",
        "memory.start",
        "query.latency",
        "memory.stop",
        "latency.stop",
    };
    CHECK(trace.events == expected_events);
    CHECK_FALSE(trace.memory_active);
    CHECK(trace.statistics_pass_count == 0);
}

TEST_CASE("SearchPassRunner skips statistics when there is no query pass",
          "[ut][eval][search_pass]") {
    uint64_t search_pass_count = 0;
    uint64_t statistics_pass_count = 0;

    SearchPassRunner::Run(
        {},
        nullptr,
        nullptr,
        [&search_pass_count](const MonitorPtr&) { ++search_pass_count; },
        [&statistics_pass_count]() { ++statistics_pass_count; });

    CHECK(search_pass_count == 0);
    CHECK(statistics_pass_count == 0);
}

}  // namespace vsag::eval
