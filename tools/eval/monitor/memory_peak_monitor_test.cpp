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

#include "memory_peak_monitor.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

namespace vsag::eval {

namespace {

constexpr std::chrono::milliseconds K_LONG_SAMPLE_INTERVAL{60000};

std::string
metric_name(const std::string& metric) {
    return metric + "(test)";
}

std::string
format_delta(uint64_t delta_bytes) {
    std::atomic<uint64_t> read_count{0};
    MemoryPeakMonitor monitor(
        "test",
        [&read_count, delta_bytes]() {
            const auto call = read_count.fetch_add(1);
            return MemoryPeakMonitor::MemoryReadResult{true, call == 0 ? 0 : delta_bytes, {}};
        },
        K_LONG_SAMPLE_INTERVAL);
    monitor.Start();
    monitor.Stop();
    return monitor.GetResult()[metric_name("memory_peak")].get<std::string>();
}

}  // namespace

TEST_CASE("MemoryPeakMonitor does not read memory before Start", "[ut][eval][memory_peak]") {
    std::atomic<uint64_t> read_count{0};
    MemoryPeakMonitor monitor(
        "test",
        [&read_count]() {
            ++read_count;
            return MemoryPeakMonitor::MemoryReadResult{true, 1024, {}};
        },
        K_LONG_SAMPLE_INTERVAL);

    const auto result = monitor.GetResult();
    CHECK(read_count == 0);
    CHECK(result[metric_name("memory_peak")] == "N/A");
    CHECK(result[metric_name("memory_peak_available")] == false);
    CHECK(result[metric_name("memory_peak_error")] == "memory monitor has not been started");
}

TEST_CASE("MemoryPeakMonitor clamps a sample below its baseline", "[ut][eval][memory_peak]") {
    std::atomic<uint64_t> read_count{0};
    MemoryPeakMonitor monitor(
        "test",
        [&read_count]() {
            const auto call = read_count.fetch_add(1);
            const uint64_t bytes = call == 0 ? 4096 : 1024;
            return MemoryPeakMonitor::MemoryReadResult{true, bytes, {}};
        },
        K_LONG_SAMPLE_INTERVAL);

    monitor.Start();
    monitor.Stop();

    const auto result = monitor.GetResult();
    CHECK(result[metric_name("memory_peak")] == "0.00 B");
    CHECK(result[metric_name("memory_rss_baseline_bytes")] == 4096);
    CHECK(result[metric_name("memory_rss_peak_bytes")] == 4096);
    CHECK(result[metric_name("memory_peak_delta_bytes")] == 0);
    CHECK(result[metric_name("memory_peak_sample_count")] == 2);
    CHECK(result[metric_name("memory_peak_failed_sample_count")] == 0);
    CHECK(result[metric_name("memory_peak_available")] == true);
}

TEST_CASE("MemoryPeakMonitor samples a transient peak in the background",
          "[ut][eval][memory_peak]") {
    std::mutex reader_mutex;
    std::condition_variable reader_condition;
    uint64_t read_count = 0;
    MemoryPeakMonitor monitor(
        "test",
        [&]() {
            std::lock_guard<std::mutex> lock(reader_mutex);
            const auto call = read_count++;
            if (read_count >= 2) {
                reader_condition.notify_all();
            }
            const uint64_t bytes = call == 0 ? 100 : (call == 1 ? 300 : 150);
            return MemoryPeakMonitor::MemoryReadResult{true, bytes, {}};
        },
        std::chrono::milliseconds{1});

    monitor.Start();
    {
        std::unique_lock<std::mutex> lock(reader_mutex);
        REQUIRE(reader_condition.wait_for(
            lock, std::chrono::seconds{1}, [&]() { return read_count >= 2; }));
    }
    monitor.Stop();

    const auto result = monitor.GetResult();
    CHECK(result[metric_name("memory_peak")] == "200.00 B");
    CHECK(result[metric_name("memory_rss_peak_bytes")] == 300);
    CHECK(result[metric_name("memory_peak_delta_bytes")] == 200);
    CHECK(result[metric_name("memory_peak_sample_count")].get<uint64_t>() >= 3);
}

TEST_CASE("MemoryPeakMonitor reports a baseline read failure", "[ut][eval][memory_peak]") {
    std::atomic<uint64_t> read_count{0};
    MemoryPeakMonitor monitor(
        "test",
        [&read_count]() {
            ++read_count;
            return MemoryPeakMonitor::MemoryReadResult{false, 0, "baseline unavailable"};
        },
        std::chrono::milliseconds{1});

    monitor.Start();
    monitor.Stop();

    const auto result = monitor.GetResult();
    CHECK(read_count == 1);
    CHECK(result[metric_name("memory_peak")] == "N/A");
    CHECK(result[metric_name("memory_rss_baseline_bytes")] == 0);
    CHECK(result[metric_name("memory_rss_peak_bytes")] == 0);
    CHECK(result[metric_name("memory_peak_delta_bytes")] == 0);
    CHECK(result[metric_name("memory_peak_sample_count")] == 0);
    CHECK(result[metric_name("memory_peak_failed_sample_count")] == 1);
    CHECK(result[metric_name("memory_peak_available")] == false);
    CHECK(result[metric_name("memory_peak_error")] == "baseline unavailable");
}

TEST_CASE("MemoryPeakMonitor can restart and Stop is idempotent", "[ut][eval][memory_peak]") {
    std::atomic<uint64_t> read_count{0};
    MemoryPeakMonitor monitor(
        "test",
        [&read_count]() {
            const auto call = read_count.fetch_add(1);
            const uint64_t values[] = {100, 200, 500, 550};
            const uint64_t index = std::min<uint64_t>(call, 3);
            return MemoryPeakMonitor::MemoryReadResult{true, values[index], {}};
        },
        K_LONG_SAMPLE_INTERVAL);

    monitor.Start();
    for (uint64_t i = 0; i < 10; ++i) {
        monitor.Record();
    }
    CHECK(read_count == 1);
    monitor.Stop();
    monitor.Stop();
    CHECK(read_count == 2);

    monitor.Start();
    monitor.Stop();
    const auto result = monitor.GetResult();
    CHECK(read_count == 4);
    CHECK(result[metric_name("memory_rss_baseline_bytes")] == 500);
    CHECK(result[metric_name("memory_rss_peak_bytes")] == 550);
    CHECK(result[metric_name("memory_peak_delta_bytes")] == 50);
    CHECK(result[metric_name("memory_peak_sample_count")] == 2);
}

TEST_CASE("MemoryPeakMonitor formats binary unit boundaries", "[ut][eval][memory_peak]") {
    CHECK(format_delta(1023) == "1023.00 B");
    CHECK(format_delta(1024) == "1.00 KB");
    CHECK(format_delta(uint64_t{1024} * 1024) == "1.00 MB");
    CHECK(format_delta(uint64_t{1024} * 1024 * 1024) == "1.00 GB");
    CHECK(format_delta(uint64_t{1024} * 1024 * 1024 * 1024) == "1.00 TB");
}

TEST_CASE("MemoryPeakMonitor default reader reports platform availability",
          "[ut][eval][memory_peak]") {
    MemoryPeakMonitor monitor("test");
    monitor.Start();
    monitor.Stop();

    const auto result = monitor.GetResult();
#if defined(__linux__) || defined(__APPLE__)
    CHECK(result[metric_name("memory_peak_available")] == true);
    CHECK(result[metric_name("memory_rss_baseline_bytes")].get<uint64_t>() > 0);
    CHECK(result[metric_name("memory_rss_peak_bytes")].get<uint64_t>() > 0);
    CHECK(result[metric_name("memory_peak_sample_count")].get<uint64_t>() >= 2);
    CHECK(result[metric_name("memory_peak_failed_sample_count")] == 0);
    CHECK(result[metric_name("memory_peak_error")].get<std::string>().empty());
#else
    CHECK(result[metric_name("memory_peak")] == "N/A");
    CHECK(result[metric_name("memory_peak_available")] == false);
    CHECK(result[metric_name("memory_peak_sample_count")] == 0);
    CHECK(result[metric_name("memory_peak_failed_sample_count")] == 1);
    CHECK_FALSE(result[metric_name("memory_peak_error")].get<std::string>().empty());
#endif
}

}  // namespace vsag::eval
