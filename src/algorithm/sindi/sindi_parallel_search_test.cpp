// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "sindi_parallel_search.h"

#include <atomic>
#include <cstdlib>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "impl/allocator/safe_allocator.h"
#include "sindi.h"
#include "sindi_cpu_timer.h"
#include "sindi_query_resources.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("SINDI fixed partitions conserve capacity independently of slots",
          "[ut][SINDIParallel]") {
    const auto plan = SindiParallelPlan::Make(37, 500, 2.0);
    REQUIRE(plan.boundaries.front() == 0);
    REQUIRE(plan.boundaries.back() == 37);
    REQUIRE(plan.quotas.size() == 16);
    REQUIRE(std::accumulate(plan.quotas.begin(), plan.quotas.end(), uint64_t{0}) == 1000);
    REQUIRE(plan.quotas[0] == 63);
    REQUIRE(plan.quotas[15] == 62);
    REQUIRE(SindiParallelPlan::Make(0, 500, 1).quotas.empty());
    REQUIRE(SindiParallelPlan::Make(1, 500, 1).quotas[0] == 500);
    REQUIRE_THROWS(SindiParallelPlan::Make(1, 500, std::numeric_limits<double>::infinity()));
    REQUIRE_THROWS(SindiParallelPlan::Make(1, 500, std::numeric_limits<double>::quiet_NaN()));
    REQUIRE_THROWS(SindiParallelPlan::Make(1, 500, 1e100));
    REQUIRE_THROWS(SindiParallelPlan::Make(1, 500, 0.9));
}

namespace {
struct ConcurrencyProbe {
    std::atomic<uint32_t> active{0};
    std::atomic<uint32_t> peak{0};
    std::atomic<uint32_t> live{0};
    std::atomic<uint32_t> calls{0};
    std::atomic<bool> fail{false};
};
class SlotFilter final : public Filter {
public:
    explicit SlotFilter(std::shared_ptr<ConcurrencyProbe> probe = nullptr)
        : probe(std::move(probe)) {
        if (this->probe)
            ++this->probe->live;
    }
    ~SlotFilter() override {
        if (probe)
            --probe->live;
    }
    bool
    CheckValid(int64_t label) const override {
        if (probe && probe->fail)
            throw std::runtime_error("injected evaluator failure");
        if (busy.exchange(true)) {
            throw std::runtime_error("one evaluator entered concurrently");
        }
        if (probe) {
            ++probe->calls;
            const auto current = probe->active.fetch_add(1) + 1;
            auto peak = probe->peak.load();
            while (peak < current && !probe->peak.compare_exchange_weak(peak, current)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            --probe->active;
        }
        const bool result = label % 2 == 0;
        busy.store(false);
        return result;
    }
    mutable std::atomic<bool> busy{false};
    std::shared_ptr<ConcurrencyProbe> probe;
};

class IndependentFilters final : public Filter, public ConcurrentFilterProvider {
public:
    bool
    CheckValid(int64_t) const override {
        throw std::runtime_error("shared filter must not run");
    }
    std::unique_ptr<ConcurrentFilterLease>
    TryLease(uint32_t count) const override {
        auto lease = std::make_unique<ConcurrentFilterLease>();
        for (uint32_t i = 0; i < count; ++i) {
            lease->evaluators.emplace_back(std::make_shared<SlotFilter>(probe));
        }
        return lease;
    }
    std::shared_ptr<ConcurrencyProbe> probe = std::make_shared<ConcurrencyProbe>();
};
}  // namespace

TEST_CASE("SINDI parallel search preserves filtering and deterministic ties",
          "[ut][SINDIParallel]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    auto params = std::make_shared<SINDIParameter>();
    params->immutable = true;
    params->term_id_limit = 4;
    params->window_size = 4;
    SINDI index(params, common);
    uint32_t term = 1;
    float value = 1;
    std::vector<SparseVector> vectors(80, SparseVector{1, &term, &value});
    std::vector<int64_t> labels(80);
    std::iota(labels.begin(), labels.end(), int64_t{100});
    auto data = Dataset::Make()
                    ->NumElements(80)
                    ->SparseVectors(vectors.data())
                    ->Ids(labels.data())
                    ->Owner(false);
    REQUIRE(index.Build(data).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors.data())->Owner(false);
    auto filter = std::make_shared<IndependentFilters>();
    const std::vector<int64_t> expected{100, 102, 104, 106, 108, 110, 112, 114};
    for (auto slots : {2, 4, 8, 16}) {
        auto json = nlohmann::json{{"sindi",
                                    {{"window_parallelism", slots},
                                     {"parallel_n_candidate_ratio", 4},
                                     {"n_candidate", 16},
                                     {"query_prune_ratio", 0},
                                     {"timeout_ms", 10000}}}};
        auto result = index.KnnSearch(query, 8, json.dump(), filter);
        REQUIRE(result->GetDim() == 8);
        std::vector<int64_t> ids(result->GetIds(), result->GetIds() + result->GetDim());
        for (auto id : ids) {
            REQUIRE(id >= 100);
            REQUIRE(id % 2 == 0);
        }
        REQUIRE(ids == expected);
        auto stats = nlohmann::json::parse(result->GetStatistics());
        REQUIRE(stats["sindi_candidate_capacity"] == 64);
        REQUIRE(stats["sindi_partitions"] == 16);
        REQUIRE(stats["sindi_granted_slots"].get<uint32_t>() <= slots);
        REQUIRE(stats["sindi_execution_complete"] == true);
        REQUIRE(stats["sindi_windows_processed"] == 20);
        REQUIRE(stats["complete"] == false);
        REQUIRE(stats["sindi_cpu_time_enabled"] == false);
        REQUIRE_FALSE(stats.contains("sindi_search_cpu_us"));
        const auto* configured = std::getenv("VSAG_SINDI_EXTRA_WORKERS");
        if (configured && std::string(configured) == "0") {
            REQUIRE(stats["sindi_granted_slots"] == 1);
        }
        if (stats["sindi_granted_slots"].get<uint32_t>() > 1) {
            REQUIRE(filter->probe->peak.load() > 1);
        }
        REQUIRE(filter->probe->live == 0);
    }
    const auto normal = R"({"sindi":{"window_parallelism":4,
        "parallel_n_candidate_ratio":4,"n_candidate":16,"timeout_ms":10000}})";
    filter->probe->fail = true;
    REQUIRE_THROWS(index.KnnSearch(query, 8, normal, filter));
    REQUIRE(filter->probe->live == 0);
    filter->probe->fail = false;
    auto profiled = index.KnnSearch(query,
                                    8,
                                    R"({"sindi":{"window_parallelism":4,
        "parallel_n_candidate_ratio":4,"n_candidate":16,"profile_cpu":true}})",
                                    filter);
    auto cpu_stats = nlohmann::json::parse(profiled->GetStatistics());
    REQUIRE(cpu_stats["sindi_cpu_time_enabled"] == true);
#ifdef __linux__
    REQUIRE(cpu_stats["sindi_cpu_time_complete"] == true);
    REQUIRE(cpu_stats["sindi_search_cpu_us"].get<double>() > 0);
    REQUIRE(cpu_stats["sindi_search_cpu_us"].get<double>() >=
            cpu_stats["sindi_worker_cpu_us"].get<double>());
#endif
    auto serial =
        index.KnnSearch(query, 8, R"({"sindi":{"n_candidate":16,"profile_cpu":true}})", nullptr);
    REQUIRE(serial->GetDim() == 8);
    auto serial_cpu = nlohmann::json::parse(serial->GetStatistics());
    REQUIRE(serial_cpu["sindi_strategy"] == "serial");
    REQUIRE(serial_cpu["sindi_cpu_time_enabled"] == true);
#ifdef __linux__
    REQUIRE(serial_cpu["sindi_cpu_time_complete"] == true);
    REQUIRE(serial_cpu["sindi_search_cpu_us"].get<double>() > 0);
    REQUIRE(serial_cpu["sindi_worker_cpu_us"] == 0);
#endif
    REQUIRE(index.KnnSearch(query, 8, normal, filter)->GetDim() == 8);
    REQUIRE(filter->probe->live == 0);
    filter->probe->calls = 0;
    auto limited = index.KnnSearch(query,
                                   8,
                                   R"({"sindi":{"window_parallelism":4,
        "parallel_n_candidate_ratio":4,"n_candidate":16,"filter_callback_limit":3}})",
                                   filter);
    REQUIRE(filter->probe->calls <= 3);
    REQUIRE(limited->GetDim() <= 3);
    REQUIRE(nlohmann::json::parse(limited->GetStatistics())["sindi_stop_reason"] ==
            "callback_limit");
    REQUIRE(filter->probe->live == 0);
    REQUIRE_THROWS(index.KnnSearch(query,
                                   8,
                                   R"({"sindi":{"window_parallelism":4,
        "parallel_n_candidate_ratio":4,"n_candidate":16,"timeout_ms":1}})",
                                   filter));
    REQUIRE(filter->probe->live == 0);
    auto generous = index.KnnSearch(query,
                                    8,
                                    R"({"sindi":{"window_parallelism":2,
        "parallel_n_candidate_ratio":4,"n_candidate":16,"timeout_ms":9223372036854775807}})",
                                    nullptr);
    REQUIRE(generous->GetDim() == 8);
    REQUIRE_THROWS(index.KnnSearch(query,
                                   8,
                                   R"({"sindi":{"window_parallelism":2,
        "parallel_n_candidate_ratio":4,"n_candidate":16,"timeout_ms":-1}})",
                                   nullptr));
    REQUIRE_THROWS(
        index.KnnSearch(query,
                        8,
                        R"({"sindi":{"window_parallelism":2,"parallel_n_candidate_ratio":1}})",
                        std::make_shared<SlotFilter>()));
    REQUIRE_THROWS(index.RangeSearch(
        query, 1, R"({"sindi":{"window_parallelism":2,"parallel_n_candidate_ratio":1}})", nullptr));
}

TEST_CASE("SINDI callback permits never exceed the invocation budget", "[ut][SINDIParallel]") {
    SindiParallelState state;
    state.callback_limit = 7;
    std::atomic<uint64_t> accepted{0};
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            for (uint32_t j = 0; j < 20; ++j) {
                if (state.ReserveCallback()) {
                    ++accepted;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(accepted == 7);
    REQUIRE(state.stop.load() == SindiParallelState::Stop::CALLBACK_LIMIT);
    SindiParallelState expired;
    expired.deadline = SindiParallelState::Clock::now();
    REQUIRE(expired.Stopped());
    REQUIRE(expired.stop.load() == SindiParallelState::Stop::DEADLINE);
}

TEST_CASE("SINDI CPU admission resolves nested and namespaced quotas", "[ut][SINDIParallel]") {
    std::unordered_map<std::string, std::string> files;
    const auto read = [&](const std::string& path) {
        const auto found = files.find(path);
        return found == files.end() ? std::string{} : found->second;
    };
    const std::string v2 = "29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
    files["/cg/pod/task/cpu.max"] = "800000 100000";
    files["/cg/pod/cpu.max"] = "400000 100000";
    files["/cg/cpu.max"] = "max 100000";
    REQUIRE(SindiCgroupCpuSlots("0::/pod/task\n", v2, read, 16) == 4);
    REQUIRE(SindiCgroupCpuSlots("0::/pod/task\n", v2, read, 2) == 2);
    files["/cg/pod/cpu.max"] = "50000 100000";
    REQUIRE(SindiCgroupCpuSlots("0::/pod/task\n", v2, read, 16) == 1);

    const std::string v1 = "30 23 0:27 / /cpu rw - cgroup cgroup rw,cpu,cpuacct\n";
    files["/cpu/pod/cpu.cfs_quota_us"] = "200000";
    files["/cpu/pod/cpu.cfs_period_us"] = "100000";
    REQUIRE(SindiCgroupCpuSlots("3:cpu,cpuacct:/pod\n", v1, read, 16) == 2);
    REQUIRE(SindiCgroupCpuSlots("3:cpuacct:/pod\n", v1, read, 16) == 16);

    const std::string namespaced = "29 23 0:26 /pod /bound rw - cgroup2 cgroup rw\n";
    files["/bound/cpu.max"] = "300000 100000";
    REQUIRE(SindiCgroupCpuSlots("0::/\n", namespaced, read, 16) == 3);
    REQUIRE(SindiCgroupCpuSlots("0::/pod\n", namespaced, read, 16) == 3);
    REQUIRE(SindiCgroupCpuSlots("0::/../../pod\n", namespaced, read, 16) == 1);
    REQUIRE(SindiCgroupCpuSlots("invalid", v2, read, 2) == 2);
}

TEST_CASE("SINDI CPU timers are opt in and stop once", "[ut][SINDIParallel]") {
    SindiCpuTime disabled;
    {
        SindiCpuTimer timer(false, disabled);
    }
    REQUIRE_FALSE(disabled.available);
    REQUIRE(disabled.nanoseconds == 0);
    SindiCpuTime cpu;
    SindiCpuTimer timer(true, cpu);
    volatile uint64_t work = 0;
    for (uint64_t i = 0; i < 10000; ++i) {
        work = work + i;
    }
    timer.Stop();
    const auto measured = cpu.nanoseconds;
    timer.Stop();
    REQUIRE(work == 49995000);
    REQUIRE(cpu.nanoseconds == measured);
#ifdef __linux__
    REQUIRE(cpu.available);
    REQUIRE(cpu.nanoseconds > 0);
#endif
}
