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

#include "latency_monitor.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../case/search_timing.h"

namespace vsag::eval {
namespace {

class ManualClock {
public:
    using rep = int64_t;
    using period = std::milli;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<ManualClock>;
    static constexpr bool is_steady = true;

    static time_point
    now() noexcept {
        return time_point(duration(now_ms_));
    }

    static void
    Set(int64_t now_ms) {
        now_ms_ = now_ms;
    }

    static void
    Advance(int64_t duration_ms) {
        now_ms_ += duration_ms;
    }

private:
    inline static int64_t now_ms_{0};
};

void
EnableAllMetrics(LatencyMonitor& monitor) {
    monitor.SetMetrics("qps");
    monitor.SetMetrics("avg_latency");
    monitor.SetMetrics("percent_latency");
}

void
RequireNear(double actual, double expected) {
    REQUIRE(std::abs(actual - expected) < 1e-9);
}

void
RequireAllPercentiles(const Monitor::JsonType& result, double expected) {
    for (const auto* percentile : {"p50", "p80", "p90", "p95", "p99"}) {
        RequireNear(result["latency_detail(ms)"][percentile].get<double>(), expected);
    }
}

}  // namespace

TEST_CASE("MeasureSearch excludes query preparation and statistics",
          "[ut][eval][latency_monitor]") {
    ManualClock::Set(0);
    ManualClock::Advance(11);

    auto [result, latency_ms] = MeasureSearch<ManualClock>([]() {
        ManualClock::Advance(7);
        return 42;
    });

    ManualClock::Advance(13);

    REQUIRE(result == 42);
    RequireNear(latency_ms, 7.0);
    REQUIRE(ManualClock::now().time_since_epoch().count() == 31);
}

TEST_CASE("LatencyMonitor retains the first query sample", "[ut][eval][latency_monitor]") {
    LatencyMonitor monitor;
    EnableAllMetrics(monitor);

    monitor.Start();
    monitor.SetTimingBatch(LatencyTimingBatch{{7.0}, 1, 0.25});
    monitor.Stop();
    const auto result = monitor.GetResult();

    RequireNear(result["qps"].get<double>(), 4.0);
    RequireNear(result["latency_avg(ms)"].get<double>(), 7.0);
    RequireAllPercentiles(result, 7.0);
    REQUIRE(result["measurement_sample_count"].get<uint64_t>() == 1);
    REQUIRE(result["measurement_successful_query_count"].get<uint64_t>() == 1);
    RequireNear(result["measurement_duration(s)"].get<double>(), 0.25);
    REQUIRE(result["measurement_method"]["latency"].get<std::string>() ==
            "steady_clock_around_knn_search");
    REQUIRE(result["measurement_method"]["qps"].get<std::string>() ==
            "successful_queries_per_wall_time");
}

TEST_CASE("LatencyMonitor uses batch wall time for imbalanced workers",
          "[ut][eval][latency_monitor]") {
    std::vector<double> latency_ms{100.0};
    latency_ms.insert(latency_ms.end(), 10, 1.0);

    LatencyMonitor monitor;
    EnableAllMetrics(monitor);
    monitor.Start();
    monitor.SetTimingBatch(LatencyTimingBatch{std::move(latency_ms), 11, 0.1});
    monitor.Stop();
    const auto result = monitor.GetResult();

    RequireNear(result["qps"].get<double>(), 110.0);
    RequireNear(result["latency_avg(ms)"].get<double>(), 10.0);
    REQUIRE(result["measurement_sample_count"].get<uint64_t>() == 11);
    REQUIRE(result["measurement_successful_query_count"].get<uint64_t>() == 11);
    RequireNear(result["measurement_duration(s)"].get<double>(), 0.1);
}

TEST_CASE("LatencyMonitor computes QPS from successful queries", "[ut][eval][latency_monitor]") {
    LatencyMonitor monitor;
    monitor.SetMetrics("qps");
    monitor.Start();
    monitor.SetTimingBatch(LatencyTimingBatch{{1.0, 2.0, 3.0}, 2, 0.5});
    monitor.Stop();

    const auto result = monitor.GetResult();
    RequireNear(result["qps"].get<double>(), 4.0);
    REQUIRE(result["measurement_sample_count"].get<uint64_t>() == 3);
    REQUIRE(result["measurement_successful_query_count"].get<uint64_t>() == 2);
}

TEST_CASE("LatencyMonitor handles an empty timing batch", "[ut][eval][latency_monitor]") {
    LatencyMonitor monitor;
    EnableAllMetrics(monitor);
    monitor.Start();
    monitor.SetTimingBatch(LatencyTimingBatch{{}, 0, 0.0});
    monitor.Stop();

    const auto result = monitor.GetResult();
    RequireNear(result["qps"].get<double>(), 0.0);
    RequireNear(result["latency_avg(ms)"].get<double>(), 0.0);
    RequireAllPercentiles(result, 0.0);
    REQUIRE(result["measurement_sample_count"].get<uint64_t>() == 0);
    REQUIRE(result["measurement_successful_query_count"].get<uint64_t>() == 0);
    RequireNear(result["measurement_duration(s)"].get<double>(), 0.0);
}

}  // namespace vsag::eval
