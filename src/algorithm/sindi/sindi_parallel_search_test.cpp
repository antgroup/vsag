// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "sindi_parallel_search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <mutex>

#ifdef __linux__
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/searcher/search_execution_context.h"
#include "sindi.h"
#include "sindi_cpu_timer.h"
#include "unittest.h"
#include "vsag/engine.h"

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
    common.search_resources_ = std::make_shared<SearchExecutionContext>(15);
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
        if (stats["sindi_granted_slots"].get<uint32_t>() > 1) {
            REQUIRE(filter->probe->peak.load() > 1);
        }
        REQUIRE(filter->probe->live == 0);
    }
    // Omitted capacity uses the same plan and result as an explicit ratio of two.
    auto defaults = nlohmann::json{
        {"sindi", {{"window_parallelism", 4}, {"n_candidate", 16}, {"query_prune_ratio", 0}}}};
    auto implicit = index.KnnSearch(query, 8, defaults.dump(), filter);
    auto implicit_stats = nlohmann::json::parse(implicit->GetStatistics());
    REQUIRE(implicit_stats["sindi_candidate_capacity"] == 32);
    defaults["sindi"]["parallel_n_candidate_ratio"] = 2;
    auto explicit_ratio = index.KnnSearch(query, 8, defaults.dump(), filter);
    REQUIRE(implicit->GetDim() == explicit_ratio->GetDim());
    for (int64_t i = 0; i < implicit->GetDim(); ++i) {
        REQUIRE(implicit->GetIds()[i] == explicit_ratio->GetIds()[i]);
        REQUIRE(implicit->GetDistances()[i] == explicit_ratio->GetDistances()[i]);
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

TEST_CASE("SINDI empty search retains injected resource readiness",
          "[ut][SINDIParallel][SINDIEmptyResources]") {
    IndexCommonParam common;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.search_resources_ = std::make_shared<SearchExecutionContext>(0);
    auto params = std::make_shared<SINDIParameter>();
    params->immutable = true;
    params->term_id_limit = 4;
    params->window_size = 10000;
    SINDI index(params, common);
    uint32_t term = 1;
    float value = 1;
    SparseVector sparse{1, &term, &value};
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(&sparse)->Owner(false);
    auto result = index.KnnSearch(
        query, 8, R"({"sindi":{"window_parallelism":4,"n_candidate":16}})", nullptr);
    REQUIRE(result->GetDim() == 0);
    auto stats = nlohmann::json::parse(result->GetStatistics());
    REQUIRE(stats["sindi_resources_ready"] == true);
    REQUIRE(stats["sindi_requested_slots"] == 4);
    REQUIRE(stats["sindi_windows_processed"] == 0);
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

TEST_CASE("SINDI global candidate merge uses one complete ordering", "[ut][SINDIParallel]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    using Candidate = vsag::SindiCandidate;
    const std::array<Candidate, 6> candidates{Candidate{1.0F, 7},
                                              Candidate{1.0F, 2},
                                              Candidate{1.0F, 5},
                                              Candidate{0.0F, 9},
                                              Candidate{1.0F, 1},
                                              Candidate{2.0F, 0}};
    std::vector<Candidate> sorted(candidates.begin(), candidates.end());
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t capacity : {0U, 1U, 3U, 6U}) {
        auto order = sorted;
        auto expected = sorted;
        expected.resize(capacity);
        do {
            vsag::SindiCandidateHeap heap(CompareByFirst{true},
                                          vsag::Vector<Candidate>(allocator.get()));
            for (const auto& candidate : order) {
                vsag::SindiMergeCandidate(heap, candidate, capacity);
            }
            // Exercise the same transfer type as the production partition merge.
            MaxHeap output(CompareByFirst{true}, vsag::Vector<Candidate>(allocator.get()));
            output = std::move(heap);
            std::vector<Candidate> actual;
            while (not output.empty()) {
                actual.push_back(output.top());
                output.pop();
            }
            std::sort(actual.begin(), actual.end());
            REQUIRE(actual == expected);
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

#ifdef __linux__
TEST_CASE("SINDI cold resource contract child", "[.][SindiResourceCold]") {
    const auto* mode = std::getenv("VSAG_TEST_SINDI_RESOURCE_MODE");
    REQUIRE(mode != nullptr);
    const bool zero = std::string(mode) == "zero";
    REQUIRE(setenv("VSAG_SINDI_MAX_CANDIDATES", "1024", 1) == 0);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    auto params = std::make_shared<SINDIParameter>();
    params->immutable = true;
    params->term_id_limit = 4;
    params->window_size = 4;
    auto index = std::make_unique<SINDI>(params, common);
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
    REQUIRE(index->Build(data).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors.data())->Owner(false);
    const std::string parameters =
        R"({"sindi":{"window_parallelism":4,"n_candidate":16,"query_prune_ratio":0}})";
    const auto thread_count = [] {
        uint64_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
            (void)entry;
            ++count;
        }
        return count;
    };
    const auto before = thread_count();
    const auto require_same_result = [](const DatasetPtr& actual, const DatasetPtr& expected) {
        REQUIRE(actual->GetDim() == expected->GetDim());
        for (int64_t i = 0; i < expected->GetDim(); ++i) {
            REQUIRE(actual->GetIds()[i] == expected->GetIds()[i]);
            REQUIRE(actual->GetDistances()[i] == expected->GetDistances()[i]);
        }
    };
    const auto serial_parameters = [&](const std::string& text) {
        auto json = nlohmann::json::parse(text);
        json["sindi"]["window_parallelism"] = 1;
        return json.dump();
    };
    const auto baseline = index->KnnSearch(query, 8, serial_parameters(parameters), nullptr);
    const auto require_serial_fallback = [&] {
        auto result = index->KnnSearch(query, 8, parameters, nullptr);
        require_same_result(result, baseline);
        REQUIRE(result->GetStatistics({"sindi_strategy"}).at(0) == "\"serial\"");
        REQUIRE(result->GetStatistics({"sindi_resources_ready"}).at(0) == "false");
        REQUIRE(result->GetStatistics({"sindi_granted_slots"}).at(0) == "1");
        REQUIRE(result->GetStatistics({"sindi_worker_limit"}).at(0) == "0");
        REQUIRE(thread_count() == before);
    };
    require_serial_fallback();
    // A huge ratio would fail the partition capacity limit. Original serial
    // search ignores it, so this detects accidentally using a one-slot plan.
    auto large_ratio = nlohmann::json::parse(parameters);
    large_ratio["sindi"]["parallel_n_candidate_ratio"] = 1000000;
    require_same_result(index->KnnSearch(query, 8, large_ratio.dump(), nullptr), baseline);
    auto ordinary_filter = std::make_shared<SlotFilter>();
    const auto filtered_baseline =
        index->KnnSearch(query, 8, serial_parameters(parameters), ordinary_filter);
    auto filtered = index->KnnSearch(query, 8, parameters, ordinary_filter);
    require_same_result(filtered, filtered_baseline);
    REQUIRE(thread_count() == before);
    auto provider = std::make_shared<IndependentFilters>();
    auto leased = index->KnnSearch(query, 8, parameters, provider);
    require_same_result(leased, filtered_baseline);
    REQUIRE(provider->probe->live == 0);
    auto budget = nlohmann::json::parse(parameters);
    budget["sindi"]["filter_callback_limit"] = 3;
    require_same_result(
        index->KnnSearch(query, 8, budget.dump(), ordinary_filter),
        index->KnnSearch(query, 8, serial_parameters(budget.dump()), ordinary_filter));
    auto timeout = nlohmann::json::parse(parameters);
    timeout["sindi"]["timeout_ms"] = 0;
    auto timed_fallback = index->KnnSearch(query, 8, timeout.dump(), nullptr);
    auto timed_baseline = index->KnnSearch(query, 8, serial_parameters(timeout.dump()), nullptr);
    require_same_result(timed_fallback, timed_baseline);
    REQUIRE(timed_fallback->GetStatistics({"is_timeout"}) ==
            timed_baseline->GetStatistics({"is_timeout"}));
    SearchRequest request;
    request.query_ = query;
    request.topk_ = 8;
    request.params_str_ = parameters;
    require_same_result(index->SearchWithRequest(request), baseline);
    auto failed = std::make_shared<ConcurrencyProbe>();
    failed->fail = true;
    REQUIRE_THROWS(index->KnnSearch(query, 8, parameters, std::make_shared<SlotFilter>(failed)));
    REQUIRE_THROWS(index->KnnSearch(query, 8, R"({"sindi":{"window_parallelism":0}})", nullptr));
    REQUIRE_THROWS(std::make_shared<SearchExecutionContext>(513));
    require_serial_fallback();
    common.search_resources_ = std::make_shared<SearchExecutionContext>(zero ? 0 : 3);
    // Bind the already-created resource while constructing an index, never in a query.
    auto old_index = std::move(index);
    index = std::make_unique<SINDI>(params, common);
    REQUIRE(index->Build(data).empty());
    const auto prepared_threads = thread_count();
    auto result = index->KnnSearch(query, 8, parameters, nullptr);
    REQUIRE(result->GetDim() == 8);
    REQUIRE(result->GetStatistics({"sindi_resources_ready"}).at(0) == "true");
    REQUIRE(result->GetStatistics({"sindi_candidate_capacity"}).at(0) == "32");
    for (double ratio : {1., 2., 4., 1000000.}) {
        auto explicit_serial = nlohmann::json::parse(serial_parameters(parameters));
        explicit_serial["sindi"]["parallel_n_candidate_ratio"] = ratio;
        auto serial_result = index->KnnSearch(query, 8, explicit_serial.dump(), nullptr);
        require_same_result(serial_result, baseline);
        REQUIRE(serial_result->GetStatistics({"dist_cmp", "distance_evaluations"}) ==
                baseline->GetStatistics({"dist_cmp", "distance_evaluations"}));
    }
    const auto limit = std::stoull(result->GetStatistics({"sindi_worker_limit"}).at(0));
    REQUIRE(limit <= 3);
    if (zero)
        REQUIRE(limit == 0);
    REQUIRE(prepared_threads == before + limit);
    REQUIRE(thread_count() == prepared_threads);
    std::vector<std::future<DatasetPtr>> queries;
    for (uint32_t i = 0; i < 8; ++i) {
        queries.push_back(std::async(
            std::launch::async, [&] { return index->KnnSearch(query, 8, parameters, nullptr); }));
    }
    for (auto& future : queries) REQUIRE(future.get()->GetDim() == 8);
    queries.clear();
    REQUIRE(thread_count() == prepared_threads);
    // An unrelated index without the capability keeps serial behavior even now.
    require_same_result(old_index->KnnSearch(query, 8, parameters, nullptr), baseline);
    index.reset();
    REQUIRE(thread_count() == prepared_threads);
}

TEST_CASE("SINDI resource contract is checked in fresh processes", "[ut][SINDIParallel]") {
    for (const std::string mode : {"zero", "workers"}) {
        std::vector<std::string> environment;
        for (char** entry = environ; *entry != nullptr; ++entry) {
            const std::string value(*entry);
            if (value.find("VSAG_TEST_SINDI_RESOURCE_MODE=") != 0)
                environment.push_back(value);
        }
        environment.push_back("VSAG_TEST_SINDI_RESOURCE_MODE=" + mode);
        std::vector<char*> env;
        for (auto& value : environment) env.push_back(value.data());
        env.push_back(nullptr);
        char executable[] = "/proc/self/exe";
        char filter[] = "[SindiResourceCold]";
        char* arguments[]{executable, filter, nullptr};
        pid_t child = 0;
        REQUIRE(posix_spawn(&child, executable, nullptr, nullptr, arguments, env.data()) == 0);
        int status = 0;
        REQUIRE(waitpid(child, &status, 0) == child);
        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) == 0);
    }
}
#endif
