// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "query_context.h"

#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "unittest.h"

TEST_CASE("SearchStatistics distance contract", "[ut][search_statistics]") {
    vsag::SearchStatistics stats;
    stats.AddDistance(vsag::SearchStatistics::DistancePhase::ROUTING, "fp32", 2);
    stats.AddDistance(vsag::SearchStatistics::DistancePhase::APPROXIMATE, "sq8", 3);
    stats.AddDistance(vsag::SearchStatistics::DistancePhase::RERANK, "fp32", 1);

    auto json = vsag::JsonType::Parse(stats.Dump());
    CHECK(json["distance_evaluations"].GetUint64() == 6);
    CHECK(json["distance_evaluations_by_phase"]["routing"].GetUint64() == 2);
    CHECK(json["distance_evaluations_by_phase"]["approximate"].GetUint64() == 3);
    CHECK(json["distance_evaluations_by_phase"]["rerank"].GetUint64() == 1);
    CHECK(json["distance_evaluations_by_backend"]["fp32"].GetUint64() == 3);
    CHECK(json["distance_evaluations_by_backend"]["sq8"].GetUint64() == 3);
    CHECK(json["distance_evaluations_by_phase"]["routing"].GetUint64() +
              json["distance_evaluations_by_phase"]["approximate"].GetUint64() +
              json["distance_evaluations_by_phase"]["rerank"].GetUint64() ==
          json["distance_evaluations"].GetUint64());
    CHECK(json["distance_evaluations_by_backend"]["fp32"].GetUint64() +
              json["distance_evaluations_by_backend"]["sq8"].GetUint64() ==
          json["distance_evaluations"].GetUint64());
    CHECK(json["complete"].GetBool());
}

TEST_CASE("SearchStatistics unknown backend is incomplete", "[ut][search_statistics]") {
    vsag::SearchStatistics stats;
    stats.AddDistance(vsag::SearchStatistics::DistancePhase::APPROXIMATE, "future_backend", 0);
    auto json = vsag::JsonType::Parse(stats.Dump());
    CHECK(json["distance_evaluations"].GetUint64() == 0);
    CHECK(json["distance_evaluations_by_backend"]["unknown"].GetUint64() == 0);
    CHECK(json["complete"].GetBool());

    stats.AddDistance(vsag::SearchStatistics::DistancePhase::APPROXIMATE, "future_backend", 4);
    json = vsag::JsonType::Parse(stats.Dump());
    CHECK(json["distance_evaluations"].GetUint64() == 4);
    CHECK(json["distance_evaluations_by_backend"]["unknown"].GetUint64() == 4);
    CHECK_FALSE(json["complete"].GetBool());
}

TEST_CASE("SearchStatistics addition saturates", "[ut][search_statistics]") {
    vsag::SearchStatistics stats;
    stats.AddDistance(vsag::SearchStatistics::DistancePhase::APPROXIMATE,
                      "fp32",
                      std::numeric_limits<uint64_t>::max());
    CHECK(vsag::SearchStatistics::SaturatingAdd(stats.distance_evaluations, 0) == false);
    CHECK(stats.complete.load());
    stats.AddDistance(vsag::SearchStatistics::DistancePhase::APPROXIMATE, "fp32", 1);
    auto json = vsag::JsonType::Parse(stats.Dump());
    CHECK(json["distance_evaluations"].GetUint64() == std::numeric_limits<uint64_t>::max());
    CHECK(json["distance_evaluations_by_phase"]["approximate"].GetUint64() ==
          std::numeric_limits<uint64_t>::max());
    CHECK(json["distance_evaluations_by_backend"]["fp32"].GetUint64() ==
          std::numeric_limits<uint64_t>::max());
    CHECK_FALSE(json["complete"].GetBool());

    stats.AddDistance(vsag::SearchStatistics::DistancePhase::ROUTING, "sq8", 1);
    json = vsag::JsonType::Parse(stats.Dump());
    CHECK(json["distance_evaluations_by_phase"]["routing"].GetUint64() == 0);
    CHECK(json["distance_evaluations_by_backend"]["sq8"].GetUint64() == 0);
}

TEST_CASE("SearchStatistics parallel path stays exact", "[ut][search_statistics]") {
    vsag::SearchStatistics stats;
    stats.parallel_.store(true);
    constexpr uint32_t kThreads = 8;
    constexpr uint64_t kPerThread = 10000;
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&stats]() {
            for (uint64_t i = 0; i < kPerThread; ++i) {
                stats.AddDistance(
                    vsag::SearchStatistics::DistancePhase::APPROXIMATE, "fp32", 1);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto json = vsag::JsonType::Parse(stats.Dump());
    CHECK(json["distance_evaluations"].GetUint64() == kThreads * kPerThread);
    CHECK(json["distance_evaluations_by_phase"]["approximate"].GetUint64() ==
          kThreads * kPerThread);
    CHECK(json["distance_evaluations_by_backend"]["fp32"].GetUint64() == kThreads * kPerThread);
    CHECK(json["complete"].GetBool());
}

TEST_CASE("SearchStatistics classifies stable backend names", "[ut][search_statistics]") {
    CHECK(vsag::SearchStatistics::BackendFromName("pqfs") ==
          vsag::DistanceEvaluationBackend::PQ_FASTSCAN);
    CHECK(vsag::SearchStatistics::BackendFromName("pq_fastscan") ==
          vsag::DistanceEvaluationBackend::PQ_FASTSCAN);
    CHECK(vsag::SearchStatistics::BackendName(vsag::DistanceEvaluationBackend::PQ_FASTSCAN) ==
          std::string("pq_fastscan"));
    CHECK(vsag::SearchStatistics::BackendFromName("int8") == vsag::DistanceEvaluationBackend::INT8);
    CHECK(vsag::SearchStatistics::BackendFromName("QUANTIZATION_ADAPTER_sq8_uniform") ==
          vsag::DistanceEvaluationBackend::SQ8_UNIFORM);
    CHECK(vsag::SearchStatistics::BackendFromName("QUANTIZATION_ADAPTER_pq_fastscan") ==
          vsag::DistanceEvaluationBackend::PQ_FASTSCAN);
    CHECK(vsag::SearchStatistics::BackendFromName("float_custom") ==
          vsag::DistanceEvaluationBackend::UNKNOWN);
}

TEST_CASE("ScopedDistancePhase restores after exceptions", "[ut][search_statistics]") {
    vsag::QueryContext context;
    context.distance_phase = vsag::DistanceEvaluationPhase::ROUTING;
    try {
        vsag::ScopedDistancePhase scoped(context, vsag::DistanceEvaluationPhase::RERANK);
        CHECK(context.distance_phase == vsag::DistanceEvaluationPhase::RERANK);
        throw std::runtime_error("injected failure");
    } catch (const std::runtime_error&) {
    }
    CHECK(context.distance_phase == vsag::DistanceEvaluationPhase::ROUTING);
}
