// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

namespace vsag {

struct alignas(64) SindiRerankWorkerProfile {
    double cost_estimation_us{0};
    double cost_start_delay_us{0};
    uint64_t candidates{0};
    uint64_t nonzeros{0};
    uint64_t stolen_candidates{0};
    uint64_t steal_attempts{0};
    uint64_t cursor_fetches{0};
    double start_delay_us{0};
    double elapsed_us{0};
};

struct SindiRerankProfile {
    double cost_estimation_us{0};
    double partition_us{0};
    double dispatch_join_us{0};
    std::vector<uint64_t> boundaries;
    std::vector<uint64_t> initial_costs;
    std::vector<SindiRerankWorkerProfile> workers;
};

// Caller-thread profile; worker records are written independently and consumed after joining.
SindiRerankProfile&
SindiLastRerankProfile();

struct SindiQueryStageProfile {
    bool enabled{false};
    bool parallel_rerank{false};
    double prepare_us{0};
    double windows_us{0};
    double merge_us{0};
    double rerank_us{0};
    double statistics_counter_us{0};
    double statistics_json_us{0};
    double result_us{0};
    double total_us{0};
};

// Query-local diagnostics, available on the calling thread after the API returns.
SindiQueryStageProfile&
SindiLastQueryProfile();

class SindiStageTimer {
public:
    using Clock = std::chrono::steady_clock;

    SindiStageTimer(bool enabled, double& destination)
        : enabled_(enabled),
          destination_(destination),
          begin_(enabled ? Clock::now() : Clock::time_point{}) {
    }

    ~SindiStageTimer() {
        Stop();
    }

    void
    Stop() {
        if (enabled_) {
            destination_ +=
                std::chrono::duration<double, std::micro>(Clock::now() - begin_).count();
            enabled_ = false;
        }
    }

private:
    bool enabled_;
    double& destination_;
    Clock::time_point begin_;
};

class SindiQueryProfileScope {
public:
    SindiQueryProfileScope()
        : profile_(SindiLastQueryProfile()), begin_(SindiStageTimer::Clock::now()) {
        profile_ = {};
        SindiLastRerankProfile() = {};
    }

    ~SindiQueryProfileScope() {
        if (profile_.enabled) {
            profile_.rerank_us = std::max(0.0, profile_.rerank_us - profile_.statistics_counter_us);
            profile_.total_us =
                std::chrono::duration<double, std::micro>(SindiStageTimer::Clock::now() - begin_)
                    .count();
        }
    }

    void
    FinishPreparation() {
        if (profile_.enabled) {
            profile_.prepare_us +=
                std::chrono::duration<double, std::micro>(SindiStageTimer::Clock::now() - begin_)
                    .count();
        }
    }

private:
    SindiQueryStageProfile& profile_;
    SindiStageTimer::Clock::time_point begin_;
};
}  // namespace vsag
