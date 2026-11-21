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

#pragma once

#include <atomic>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace vsag {

/**
 * @brief Minimal request-owned profiling for single-query IVF custom-distance KNN.
 * All stored durations are nanoseconds. Worker start/end are relative to scan start.
 * scan includes worker submission, execution and drain; total includes the public IVF call.
 * Worker CPU is measured on its own thread only, excluding callback-created threads.
 * Attribute and callback timings are wall durations and may include blocking.
 * Worker times overlap: never add them and interpret the sum as wall latency.
 * Read/serialize only after search returns or throws. Do not mutate fields during search.
 */
class SearchProfile {
public:
    struct Wall {
        uint64_t total_ns{0};
        uint64_t scan_ns{0};
    };

    struct alignas(64) Worker {
        uint64_t start_ns{0};
        uint64_t end_ns{0};
        uint64_t execution_ns{0};
        uint64_t cpu_ns{0};
        uint64_t attribute_exec_ns{0};
        uint64_t distance_callback_ns{0};
        uint64_t buckets{0};
        uint64_t distance_candidates{0};
        uint64_t distance_batches{0};
        bool cpu_available{false};
    };

    Wall wall;
    std::vector<Worker> workers;
    // Normal return, including a partial timeout result; false when search throws.
    bool complete{false};

    /** @brief Internal lifecycle API: reject concurrent reuse before resetting output. */
    bool
    Begin() noexcept {
        bool expected = false;
        if (not active_.compare_exchange_strong(expected, true)) {
            return false;
        }
        wall = Wall{};
        workers.clear();
        complete = false;
        return true;
    }

    void
    Finish() noexcept {
        active_.store(false);
    }

    /**
     * @brief Caller-thread report: one summary line followed by one line per worker.
     * scan_avg_cores = sum(worker CPU) / scan wall; unavailable CPU is printed as NA,
     * never as zero cores. CPU validity is explicit for each worker and the summary.
     * Counts describe submitted callback candidates (including a batch that throws),
     * not distinct IDs; replicated bucket entries can be counted multiple times.
     */
    std::string
    ToString(uint64_t query_id = 0) const {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::fixed << std::setprecision(3);
        uint64_t cpu_ns = 0;
        bool cpu_available = not workers.empty();
        for (const auto& worker : workers) {
            cpu_available = cpu_available and worker.cpu_available;
            cpu_ns += worker.cpu_ns;
        }
        out << "[IVF_PROFILE] query=" << query_id << " parallelism=" << workers.size()
            << " complete=" << (complete ? "true" : "false")
            << " total_wall_ms=" << wall.total_ns / 1e6 << " scan_wall_ms=" << wall.scan_ns / 1e6
            << " cpu_available=" << (cpu_available ? "true" : "false") << " worker_cpu_ms=";
        if (cpu_available) {
            out << cpu_ns / 1e6;
        } else {
            out << "NA";
        }
        out << " scan_avg_cores=";
        if (cpu_available and wall.scan_ns > 0) {
            out << static_cast<double>(cpu_ns) / wall.scan_ns;
        } else {
            out << "NA";
        }
        out << '\n';
        uint64_t id = 0;
        for (const auto& worker : workers) {
            out << "[IVF_WORKER] query=" << query_id << " worker=" << id++
                << " start_ms=" << worker.start_ns / 1e6 << " end_ms=" << worker.end_ns / 1e6
                << " wall_ms=" << worker.execution_ns / 1e6
                << " cpu_available=" << (worker.cpu_available ? "true" : "false") << " cpu_ms=";
            if (worker.cpu_available) {
                out << worker.cpu_ns / 1e6;
            } else {
                out << "NA";
            }
            out << " buckets=" << worker.buckets << " candidates=" << worker.distance_candidates
                << " batches=" << worker.distance_batches
                << " attr_ms=" << worker.attribute_exec_ns / 1e6
                << " dist_ms=" << worker.distance_callback_ns / 1e6 << '\n';
        }
        return out.str();
    }

private:
    std::atomic<bool> active_{false};
};

}  // namespace vsag
