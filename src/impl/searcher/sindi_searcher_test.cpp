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

#include "sindi_searcher.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

#include "algorithm/sindi/sindi.h"
#include "datacell/immutable_sindi_term_datacell.h"
#include "datacell/mutable_sindi_term_datacell.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/default_thread_pool.h"
#include "unittest.h"

using namespace vsag;

namespace {
class TracedSindiDataCell final : public MutableSindiTermDataCell {
public:
    using MutableSindiTermDataCell::MutableSindiTermDataCell;
    mutable std::array<std::atomic<uint64_t>, 64> visits{};
    bool throw_on_first{false};

    void
    QueryWindow(float* dists,
                uint32_t window,
                const SparseTermComputerPtr& computer,
                bool use_term_lists,
                SindiQueryContext& context) const override {
        visits.at(window).fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
        if (throw_on_first and window == 0) {
            throw std::runtime_error("injected window failure");
        }
        MutableSindiTermDataCell::QueryWindow(dists, window, computer, use_term_lists, context);
    }
};

class RejectingPool final : public DefaultThreadPool {
public:
    RejectingPool() : DefaultThreadPool(4) {
    }
    uint32_t enqueues{0};

    std::future<void>
    Enqueue(std::function<void()> task) override {
        if (++enqueues == 3) {
            throw std::runtime_error("injected enqueue failure");
        }
        return DefaultThreadPool::Enqueue(std::move(task));
    }
};

// A caller-supplied pool opts into identity without inheriting DefaultThreadPool.
class IdentifiedCustomPool final : public ThreadPool, public ThreadPoolWorkerIdentity {
public:
    std::atomic<uint32_t> enqueues{0};

    std::future<void>
    Enqueue(std::function<void()> task) override {
        enqueues.fetch_add(1);
        return pool_.Enqueue(std::move(task));
    }
    bool
    IsCurrentThread() const override {
        return pool_.IsCurrentThread();
    }
    void
    WaitUntilEmpty() override {
        pool_.WaitUntilEmpty();
    }
    void
    SetQueueSizeLimit(uint64_t limit) override {
        pool_.SetQueueSizeLimit(limit);
    }
    void
    SetPoolSize(uint64_t limit) override {
        pool_.SetPoolSize(limit);
    }

private:
    DefaultThreadPool pool_{4};
};

// Models a pool whose workers are occupied by queries. The test can release queued
// work after a timeout, so the old blocking-join bug fails without hanging the suite.
class DeferredPool final : public ThreadPool {
public:
    std::future<void>
    Enqueue(std::function<void()> task) override {
        auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
        auto future = packaged->get_future();
        std::unique_lock<std::mutex> lock(mutex_);
        if (recovery_mode_) {
            lock.unlock();
            (*packaged)();
        } else {
            tasks_.emplace_back([packaged]() { (*packaged)(); });
        }
        return future;
    }

    void
    RecoverQueuedWork() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            recovery_mode_ = true;
        }
        WaitUntilEmpty();
    }

    bool
    RunOne() {
        std::function<void()> task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) {
                return false;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
        return true;
    }

    void
    WaitUntilEmpty() override {
        while (RunOne()) {
        }
    }
    void
    SetQueueSizeLimit(uint64_t) override {
    }
    void
    SetPoolSize(uint64_t) override {
    }

private:
    std::mutex mutex_;
    bool recovery_mode_{false};
    std::deque<std::function<void()>> tasks_;
};
// Actual default-pool workers run the queries. Inner submissions remain recoverable
// from the test thread if a regression reintroduces a blocking join.
class RecoverableDefaultPool final : public DefaultThreadPool {
public:
    explicit RecoverableDefaultPool(uint32_t workers) : DefaultThreadPool(workers) {
        SetQueueSizeLimit(1);
    }

    std::future<void>
    RunQuery(std::function<void()> task) {
        return DefaultThreadPool::Enqueue(std::move(task));
    }

    std::future<void>
    Enqueue(std::function<void()> task) override {
        return deferred_.Enqueue(std::move(task));
    }

    void
    RecoverQueuedWork() {
        deferred_.RecoverQueuedWork();
    }

private:
    DeferredPool deferred_;
};
}  // namespace

TEST_CASE("SINDI weighted partition covers empty and skewed work", "[ut][SindiSearcher]") {
    REQUIRE_THROWS(SindiSearcher::Partition({}, 0, true));
    for (uint32_t workers : {1U, 2U, 4U, 16U}) {
        for (const auto& costs : std::vector<std::vector<uint64_t>>{{},
                                                                    {0},
                                                                    {0, 0, 0},
                                                                    {1, 2, 3},
                                                                    {0, 0, 100, 0, 0, 1},
                                                                    {10, 10, 10, 10, 1, 1, 1, 1},
                                                                    std::vector<uint64_t>(67, 0)}) {
            for (bool weighted : {false, true}) {
                const auto bounds = SindiSearcher::Partition(costs, workers, weighted);
                REQUIRE(bounds.size() == workers + 1);
                REQUIRE(bounds.front() == 0);
                REQUIRE(bounds.back() == costs.size());
                REQUIRE(std::is_sorted(bounds.begin(), bounds.end()));
                if (weighted and costs.size() >= workers and
                    std::accumulate(costs.begin(), costs.end(), uint64_t{0}) > 0) {
                    for (uint32_t i = 0; i < workers; ++i) {
                        REQUIRE(bounds[i] < bounds[i + 1]);
                    }
                }
            }
        }
    }
    const std::vector<uint64_t> costs{10, 10, 10, 10, 1, 1, 1, 1};
    REQUIRE(SindiSearcher::Partition(costs, 2, true)[1] == 2);
    REQUIRE(SindiSearcher::Partition(costs, 2, false)[1] == 4);
}

TEST_CASE("SINDI window options reject invalid modes", "[ut][SindiSearcher]") {
    REQUIRE(SindiSearchOptions::Parse(JsonType::Parse(R"({"sindi":{}})")).scheduler ==
            SindiWindowScheduler::COUNT_DEQUE);
    REQUIRE(SindiSearchOptions::Parse(JsonType::Parse(R"({"sindi":{}})")).rerank_scheduler ==
            SindiRerankScheduler::COUNT_STATIC);
    REQUIRE_THROWS(
        SindiSearchOptions::Parse(JsonType::Parse(R"({"sindi":{"window_scheduler":"typo"}})")));
    for (const auto* mode : {"count_deque", "count_atomic", "cost_atomic"}) {
        auto json =
            JsonType::Parse(std::string(R"({"sindi":{"window_scheduler":")") + mode + R"("}})");
        auto options = SindiSearchOptions::Parse(json);
        REQUIRE(std::string(SindiSearcher::SchedulerName(options.scheduler)) == mode);
        REQUIRE_FALSE(options.profile);
    }
}

TEST_CASE("SINDI rerank batches preserve distances and aggregate statistics",
          "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.dim_ = 16;
    auto parameter = std::make_shared<SparseVectorDataCellParameter>();
    parameter->FromJson(JsonType::Parse(R"({"io_params":{"type":"block_memory_io"},
        "quantization_params":{"type":"sparse"}})"));
    auto cell = FlattenInterface::MakeInstance(parameter, common);
    uint32_t term = 3;
    for (uint32_t id = 0; id < 37; ++id) {
        float value = static_cast<float>(id) * 0.25F;
        SparseVector vector{1, &term, &value};
        cell->InsertVector(&vector, id);
    }
    float value = 2;
    SparseVector query{1, &term, &value};
    Vector<InnerIdType> ids(allocator.get());
    for (uint32_t id = 0; id < 37; ++id) {
        ids.push_back(36 - id);
    }
    auto pool = std::make_shared<SafeThreadPool>(std::make_shared<DefaultThreadPool>(8));
    for (bool statistics_enabled : {false, true}) {
        for (uint32_t workers : {1U, 4U, 16U, 64U}) {
            SearchStatistics statistics;
            const auto distances =
                SindiSearcher::RerankDistances(*cell,
                                               query,
                                               ids,
                                               workers,
                                               pool,
                                               allocator.get(),
                                               statistics_enabled ? &statistics : nullptr,
                                               true);
            REQUIRE(distances.size() == ids.size());
            for (uint64_t i = 0; i < ids.size(); ++i) {
                REQUIRE(distances[i] == 1.0F - static_cast<float>(ids[i]) * 0.5F);
            }
            REQUIRE(statistics.distance_evaluations.load() ==
                    (statistics_enabled ? ids.size() : 0));
            REQUIRE(statistics
                        .distance_evaluations_by_phase[static_cast<uint64_t>(
                            DistanceEvaluationPhase::RERANK)]
                        .load() == (statistics_enabled ? ids.size() : 0));
        }
    }
    REQUIRE(SindiSearcher::RerankDistances(
                *cell, query, ids, 16, nullptr, allocator.get(), nullptr, false)
                .size() == ids.size());
    auto rejecting = std::make_shared<RejectingPool>();
    auto rejecting_pool = std::make_shared<SafeThreadPool>(rejecting);
    REQUIRE_THROWS(SindiSearcher::RerankDistances(
        *cell, query, ids, 8, rejecting_pool, allocator.get(), nullptr, false));
    REQUIRE(rejecting->enqueues == 3);
    for (const auto scheduler : {SindiRerankScheduler::COUNT_STATIC,
                                 SindiRerankScheduler::COUNT_ATOMIC,
                                 SindiRerankScheduler::NNZ_ATOMIC}) {
        auto deferred = std::make_shared<DeferredPool>();
        auto shared_pool = std::make_shared<SafeThreadPool>(deferred);
        Vector<float> actual(allocator.get());
        auto query_task = deferred->Enqueue([&]() {
            Vector<InnerIdType> local_ids(ids);
            actual = SindiSearcher::RerankDistances(
                *cell, query, local_ids, 4, shared_pool, allocator.get(), nullptr, true, scheduler);
        });
        auto occupied_worker = std::async(std::launch::async, [&]() { deferred->RunOne(); });
        const auto completed = query_task.wait_for(std::chrono::seconds(5));
        deferred->WaitUntilEmpty();
        query_task.get();
        occupied_worker.get();
        // Verify that draining deferred callbacks after the query is safe.
        deferred->WaitUntilEmpty();
        REQUIRE(completed == std::future_status::ready);
        for (uint64_t i = 0; i < ids.size(); ++i) {
            REQUIRE(actual[i] == 1.0F - static_cast<float>(ids[i]) * 0.5F);
        }
    }
    ids[0] = 99999;
    REQUIRE_THROWS(SindiSearcher::RerankDistances(
        *cell, query, ids, 4, pool, allocator.get(), nullptr, false));
    ids[10] = 99999;
    auto custom_pool = std::make_shared<SafeThreadPool>(std::make_shared<DeferredPool>());
    REQUIRE_THROWS(SindiSearcher::RerankDistances(
        *cell, query, ids, 4, custom_pool, allocator.get(), nullptr, false));
    ids.clear();
    REQUIRE(
        SindiSearcher::RerankDistances(*cell, query, ids, 4, pool, allocator.get(), nullptr, false)
            .empty());
    REQUIRE_THROWS(SindiSearcher::RerankDistances(
        *cell, query, ids, 0, pool, allocator.get(), nullptr, false));
}

TEST_CASE("SINDI rerank atomic ranges balance nonzeros and preserve candidate positions",
          "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.dim_ = 128;
    const auto* value_type = GENERATE("fp32", "fp16");
    auto parameter = std::make_shared<SparseVectorDataCellParameter>();
    parameter->FromJson(JsonType::Parse(
        nlohmann::json{{"io_params", {{"type", "block_memory_io"}}},
                       {"quantization_params", {{"type", "sparse"}, {"value_type", value_type}}}}
            .dump()));
    auto cell = FlattenInterface::MakeInstance(parameter, common);
    std::vector<uint32_t> terms(128);
    std::iota(terms.begin(), terms.end(), 0);
    std::vector<float> values(128, 0.5F);
    Vector<InnerIdType> ids(allocator.get());
    std::vector<uint64_t> costs;
    for (uint32_t id = 0; id < 64; ++id) {
        const uint32_t length = id < 16 ? 96 : (id % 7 == 0 ? 0 : 2);
        SparseVector vector{length, terms.data(), values.data()};
        cell->InsertVector(&vector, id);
        ids.push_back(id);
        costs.push_back(length);
    }
    // Duplicate IDs are independent candidate positions, with independent output writes.
    ids.push_back(0);
    costs.push_back(costs[0]);
    SparseVector query{128, terms.data(), values.data()};
    auto pool = std::make_shared<SafeThreadPool>(std::make_shared<DefaultThreadPool>(4));
    for (const auto scheduler : {SindiRerankScheduler::COUNT_STATIC,
                                 SindiRerankScheduler::COUNT_ATOMIC,
                                 SindiRerankScheduler::NNZ_ATOMIC}) {
        for (uint32_t workers : {1U, 4U, 80U}) {
            const auto actual = SindiSearcher::RerankDistances(
                *cell, query, ids, workers, pool, allocator.get(), nullptr, true, scheduler);
            const auto& profile = SindiLastRerankProfile();
            const uint64_t count = std::min<uint64_t>(workers, ids.size());
            REQUIRE(profile.workers.size() == count);
            uint64_t completed = 0, nonzeros = 0, fetches = 0, attempts = 0;
            for (uint64_t i = 0; i < ids.size(); ++i) {
                REQUIRE(actual[i] == 1.0F - costs[i] * 0.25F);
            }
            for (const auto& worker : profile.workers) {
                completed += worker.candidates;
                nonzeros += worker.nonzeros;
                fetches += worker.cursor_fetches;
                attempts += worker.steal_attempts;
                REQUIRE(worker.stolen_candidates <= worker.candidates);
                REQUIRE(worker.elapsed_us >= 0);
                REQUIRE(worker.start_delay_us >= 0);
            }
            REQUIRE(completed == ids.size());
            REQUIRE(nonzeros == std::accumulate(costs.begin(), costs.end(), uint64_t{0}));
            const bool atomic = workers > 1 and scheduler != SindiRerankScheduler::COUNT_STATIC;
            REQUIRE(fetches == (atomic ? ids.size() + count * count : 0));
            REQUIRE(attempts == (atomic ? count * (count - 1) : 0));
            if (scheduler == SindiRerankScheduler::NNZ_ATOMIC and workers == 4) {
                REQUIRE(profile.boundaries == SindiSearcher::Partition(costs, 4, true));
                const auto max_cost =
                    *std::max_element(profile.initial_costs.begin(), profile.initial_costs.end());
                REQUIRE(max_cost < 600);
            }
        }
        // No worker may leave a dangling reference when a future fails.
        ids[0] = 99999;
        REQUIRE_THROWS(SindiSearcher::RerankDistances(
            *cell, query, ids, 4, pool, allocator.get(), nullptr, false, scheduler));
        ids[0] = 0;
        SindiSearcher::RerankDistances(
            *cell, query, ids, 4, pool, allocator.get(), nullptr, false, scheduler);
        REQUIRE(SindiLastRerankProfile().workers.empty());
    }
    // Parallel block prefixes must yield the same weighted boundaries for uneven chunks.
    for (uint64_t size : {1U, 2U, 3U, 7U, 13U, 37U, 63U, 65U}) {
        Vector<InnerIdType> subset(allocator.get());
        subset.insert(subset.end(), ids.begin(), ids.begin() + size);
        std::vector<uint64_t> weights(costs.begin(), costs.begin() + size);
        for (uint32_t workers : {1U, 2U, 4U, 16U, 80U}) {
            const uint32_t count = std::min<uint64_t>(size, workers);
            const auto actual = SindiSearcher::RerankDistances(*cell,
                                                               query,
                                                               subset,
                                                               workers,
                                                               pool,
                                                               allocator.get(),
                                                               nullptr,
                                                               true,
                                                               SindiRerankScheduler::NNZ_ATOMIC);
            REQUIRE(SindiLastRerankProfile().boundaries ==
                    SindiSearcher::Partition(weights, count, true));
            for (uint64_t i = 0; i < size; ++i) {
                REQUIRE(actual[i] == 1.0F - weights[i] * 0.25F);
            }
            for (const auto& worker : SindiLastRerankProfile().workers) {
                REQUIRE(worker.cost_estimation_us > 0);
                REQUIRE(worker.cost_start_delay_us >= 0);
            }
        }
    }
    auto small_pool = std::make_shared<SafeThreadPool>(std::make_shared<DefaultThreadPool>(1));
    const auto single_pool_result = SindiSearcher::RerankDistances(
        *cell, query, ids, 16, small_pool, allocator.get(), nullptr, false);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        REQUIRE(single_pool_result[i] == 1.0F - costs[i] * 0.25F);
    }
    // Independent queries share the pool but keep counters and caller-thread profiles separate.
    for (const auto scheduler : {SindiRerankScheduler::COUNT_STATIC,
                                 SindiRerankScheduler::COUNT_ATOMIC,
                                 SindiRerankScheduler::NNZ_ATOMIC}) {
        const auto concurrent = [&](uint32_t size) {
            Vector<InnerIdType> subset(allocator.get());
            subset.insert(subset.end(), ids.begin(), ids.begin() + size);
            SearchStatistics statistics;
            auto actual = SindiSearcher::RerankDistances(
                *cell, query, subset, 16, pool, allocator.get(), &statistics, true, scheduler);
            return std::make_tuple(std::move(actual),
                                   SindiLastRerankProfile(),
                                   statistics.distance_evaluations.load());
        };
        auto first = std::async(std::launch::async, concurrent, 17);
        auto second = std::async(std::launch::async, concurrent, 37);
        for (auto* future : {&first, &second}) {
            auto [actual, profile, evaluations] = future->get();
            REQUIRE(evaluations == actual.size());
            REQUIRE(profile.boundaries.back() == actual.size());
            uint64_t completed = 0;
            for (const auto& worker : profile.workers) {
                completed += worker.candidates;
            }
            REQUIRE(completed == actual.size());
            for (uint64_t i = 0; i < actual.size(); ++i) {
                REQUIRE(actual[i] == 1.0F - costs[i] * 0.25F);
            }
        }
    }
    // All-zero NNZ must still cover every position exactly once, without dividing by zero.
    Vector<InnerIdType> zeros(allocator.get());
    for (InnerIdType id : {21U, 28U, 35U, 42U, 49U, 56U, 63U}) {
        zeros.push_back(id);
    }
    const auto zero_distances = SindiSearcher::RerankDistances(*cell,
                                                               query,
                                                               zeros,
                                                               4,
                                                               pool,
                                                               allocator.get(),
                                                               nullptr,
                                                               true,
                                                               SindiRerankScheduler::NNZ_ATOMIC);
    for (const auto distance : zero_distances) {
        REQUIRE(distance == 1.0F);
    }
    uint64_t completed = 0, fetches = 0;
    for (const auto& worker : SindiLastRerankProfile().workers) {
        REQUIRE(worker.nonzeros == 0);
        completed += worker.candidates;
        fetches += worker.cursor_fetches;
    }
    REQUIRE(completed == zeros.size());
    REQUIRE(fetches == zeros.size() + 16);
}

TEST_CASE("SINDI bulk merge preserves legacy candidate sets", "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::mt19937 random(424242);
    for (uint32_t workers : {1U, 4U, 16U}) {
        for (uint64_t limit : {1U, 7U, 500U, 1000U}) {
            for (uint32_t pattern = 0; pattern < 6; ++pattern) {
                for (bool prefilled : {false, true}) {
                    CAPTURE(workers, limit, pattern, prefilled);
                    std::vector<SindiWindowHeap> heaps;
                    for (uint32_t worker = 0; worker < workers; ++worker) {
                        heaps.emplace_back(allocator.get());
                    }
                    std::vector<uint32_t> ids(37 * workers);
                    std::iota(ids.begin(), ids.end(), 0);
                    std::shuffle(ids.begin(), ids.end(), random);
                    for (const auto id : ids) {
                        float score = static_cast<float>(id) * 0.25F - 100.0F;
                        if (pattern == 1) {
                            score = static_cast<float>(id % 11);
                        } else if (pattern == 2) {
                            score = id % 2 == 0 ? 0.0F : -0.0F;
                        } else if (pattern == 3 and id == 0) {
                            score = std::numeric_limits<float>::infinity();
                        } else if (pattern == 4 and id == 0) {
                            score = -std::numeric_limits<float>::infinity();
                        } else if (pattern == 5 and id == 0) {
                            score = std::numeric_limits<float>::quiet_NaN();
                        }
                        heaps[random() % workers].emplace(score, id);
                    }
                    auto reference_heaps = heaps;
                    MaxHeap actual(allocator.get());
                    if (prefilled) {
                        actual.emplace(-50.125F, ids.size());
                    }
                    auto expected = actual;
                    // Preserve the previous implementation as the differential oracle.
                    for (auto& local : reference_heaps) {
                        while (not local.empty()) {
                            const auto candidate = local.top();
                            if (expected.size() < limit or candidate < expected.top()) {
                                expected.push(candidate);
                                if (expected.size() > limit) {
                                    expected.pop();
                                }
                            }
                            local.pop();
                        }
                    }
                    SindiSearcher::MergeHeaps(heaps, actual, limit, allocator.get());
                    REQUIRE(actual.size() == expected.size());
                    std::set<std::pair<uint32_t, InnerIdType>> actual_candidates;
                    std::set<std::pair<uint32_t, InnerIdType>> expected_candidates;
                    while (not expected.empty()) {
                        if (pattern == 0 or pattern >= 3 or prefilled) {
                            REQUIRE(actual.top().second == expected.top().second);
                        }
                        uint32_t actual_bits = 0;
                        uint32_t expected_bits = 0;
                        std::memcpy(&actual_bits, &actual.top().first, sizeof(actual_bits));
                        std::memcpy(&expected_bits, &expected.top().first, sizeof(expected_bits));
                        if (pattern == 0 or pattern >= 3 or prefilled) {
                            REQUIRE(actual_bits == expected_bits);
                        }
                        actual_candidates.emplace(actual_bits, actual.top().second);
                        expected_candidates.emplace(expected_bits, expected.top().second);
                        actual.pop();
                        expected.pop();
                    }
                    REQUIRE(actual_candidates == expected_candidates);
                }
            }
        }
    }
    std::vector<SindiWindowHeap> empty;
    MaxHeap heap(allocator.get());
    REQUIRE(SindiSearcher::MergeHeaps(empty, heap, 500, allocator.get()));
    REQUIRE(heap.empty());
    empty.emplace_back(allocator.get());
    empty.back().emplace(1.0F, 1);
    REQUIRE(SindiSearcher::MergeHeaps(empty, heap, 0, allocator.get()));
    REQUIRE(heap.empty());
    // A tie only across the selection boundary must also preserve legacy tie-breaking.
    empty.back().emplace(1.0F, 2);
    REQUIRE_FALSE(SindiSearcher::MergeHeaps(empty, heap, 1, allocator.get()));
    REQUIRE(heap.size() == 1);
    REQUIRE(heap.top().second == 1);
}

TEST_CASE("SINDI schedulers preserve all selected windows and candidates", "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<SafeThreadPool>(std::make_shared<DefaultThreadPool>(8));
    for (bool compact : {false, true}) {
        auto data = std::make_shared<TracedSindiDataCell>(16,
                                                          8,
                                                          allocator.get(),
                                                          SparseValueQuantizationType::FP32,
                                                          std::make_shared<QuantizationParams>());
        uint32_t terms[2]{3, 5};
        for (uint32_t doc = 0; doc < 512; ++doc) {
            float values[2]{0.25F + doc, 0.5F};
            SparseVector vector{doc % 4 == 0 ? 2U : 1U, terms, values};
            data->InsertVector(vector, doc);
        }
        for (uint32_t w = 0; w < 64; ++w) {
            data->SortByValue(w);
        }
        SindiSearchTermDataCellPtr cell = data;
        if (compact) {
            auto immutable =
                std::make_shared<ImmutableSindiTermDataCell>(16,
                                                             8,
                                                             true,
                                                             SparseValueQuantizationType::FP32,
                                                             std::make_shared<QuantizationParams>(),
                                                             allocator.get());
            for (uint32_t w = 0; w < 64; ++w) {
                immutable->AppendWindow(data->GetWindow(w));
            }
            cell = immutable;
        }
        float values[2]{1, 2};
        SparseVector query{2, terms, values};
        SINDISearchParameter search;
        auto computer = std::make_shared<SparseTermComputer>(query, search, allocator.get(), 64);
        REQUIRE(SindiSearcher::EstimateWindowCost(*cell, *computer, 0, false) == 10);
        uint32_t missing_term = 99;
        float missing_value = 1;
        SparseVector missing_query{1, &missing_term, &missing_value};
        SparseTermComputer missing(missing_query, search, allocator.get());
        REQUIRE(SindiSearcher::EstimateWindowCost(*cell, missing, 0, false) == 0);
        REQUIRE_THROWS(SindiSearcher::EstimateWindowCost(*cell, missing, 99, false));
        SINDISearchParameter pruned;
        pruned.term_prune_ratio = 0.5;
        pruned.term_retain_threshold = 128;
        SparseTermComputer pruning_computer(query, pruned, allocator.get(), 64);
        REQUIRE(SindiSearcher::EstimateWindowCost(*cell, pruning_computer, 0, false) == 3);
        REQUIRE(SindiSearcher::EstimateWindowCost(*cell, pruning_computer, 0, true) == 10);
        InnerSearchParam param;
        param.ef = 20;
        SindiMetadataFilter metadata(allocator.get());
        SindiMetadataSearchRoute route;
        std::vector<uint32_t> windows;
        for (uint32_t w = 0; w < 64; w += 2) {
            windows.push_back(w);
        }
        std::set<InnerIdType> reference;
        for (bool term_lists : {false, true}) {
            for (auto mode : {SindiWindowScheduler::COUNT_DEQUE,
                              SindiWindowScheduler::COUNT_ATOMIC,
                              SindiWindowScheduler::COST_ATOMIC}) {
                for (uint32_t workers : {1U, 4U, 8U, 16U}) {
                    param.parallel_search_thread_count = workers;
                    SearchStatistics statistics;
                    SindiWindowSearchRequest request{windows,
                                                     cell,
                                                     computer,
                                                     param,
                                                     metadata,
                                                     route,
                                                     pool,
                                                     allocator.get(),
                                                     8,
                                                     512,
                                                     20,
                                                     term_lists,
                                                     &statistics};
                    SindiWindowSearchProfile profile;
                    MaxHeap heap(allocator.get());
                    SindiSearchOptions options{mode, true};
                    for (auto& visits : data->visits) {
                        visits.store(0, std::memory_order_relaxed);
                    }
                    SindiSearcher::SearchWindows(request, options, profile, heap);
                    if (not compact) {
                        for (uint32_t w = 0; w < 64; ++w) {
                            REQUIRE(data->visits[w].load() == (w % 2 == 0 ? 1 : 0));
                        }
                    }
                    REQUIRE(heap.size() == 20);
                    std::set<InnerIdType> result;
                    while (not heap.empty()) {
                        result.insert(heap.top().second);
                        heap.pop();
                    }
                    if (reference.empty()) {
                        reference = result;
                    }
                    REQUIRE(result == reference);
                    uint64_t processed = 0;
                    uint64_t cost = 0;
                    for (const auto& worker : profile.workers) {
                        processed += worker.windows;
                        cost += worker.estimated_cost;
                        REQUIRE(worker.stolen_windows <= worker.windows);
                        REQUIRE(worker.stolen_windows <= worker.steal_attempts);
                        REQUIRE(worker.elapsed_us >= 0);
                    }
                    REQUIRE(processed == windows.size());
                    if (mode == SindiWindowScheduler::COST_ATOMIC) {
                        REQUIRE(cost == 320);
                    }
                    REQUIRE(computer->term_iterator_ == 0);
                    auto json = nlohmann::json::parse(
                        SindiSearcher::DumpStatistics(statistics, options, profile));
                    REQUIRE(json["sindi_scheduler"]["workers"].size() == workers);
                }
            }
        }
        // Independent query-local ranges, heaps, and profiling on one shared pool.
        auto concurrent = [&]() {
            auto local_param = param;
            local_param.parallel_search_thread_count = 4;
            SindiWindowSearchProfile profile;
            SearchStatistics statistics;
            SindiWindowSearchRequest request{windows,
                                             cell,
                                             computer,
                                             local_param,
                                             metadata,
                                             route,
                                             pool,
                                             allocator.get(),
                                             8,
                                             512,
                                             20,
                                             false,
                                             &statistics};
            MaxHeap heap(allocator.get());
            SindiSearcher::SearchWindows(
                request, {SindiWindowScheduler::COST_ATOMIC, false}, profile, heap);
            return heap.top().first;
        };
        auto first = std::async(std::launch::async, concurrent);
        auto second = std::async(std::launch::async, concurrent);
        REQUIRE(first.get() == second.get());
        for (const auto scheduler : {SindiWindowScheduler::COUNT_DEQUE,
                                     SindiWindowScheduler::COUNT_ATOMIC,
                                     SindiWindowScheduler::COST_ATOMIC}) {
            auto deferred = std::make_shared<DeferredPool>();
            auto shared_pool = std::make_shared<SafeThreadPool>(deferred);
            std::set<InnerIdType> result;
            auto query_task = deferred->Enqueue([&]() {
                SearchStatistics local_stats;
                SindiWindowSearchProfile local_profile;
                SindiWindowSearchRequest request{windows,
                                                 cell,
                                                 computer,
                                                 param,
                                                 metadata,
                                                 route,
                                                 shared_pool,
                                                 allocator.get(),
                                                 8,
                                                 512,
                                                 20,
                                                 false,
                                                 &local_stats};
                MaxHeap local_heap(allocator.get());
                SindiSearcher::SearchWindows(request, {scheduler, true}, local_profile, local_heap);
                while (not local_heap.empty()) {
                    result.insert(local_heap.top().second);
                    local_heap.pop();
                }
            });
            auto occupied_worker = std::async(std::launch::async, [&]() { deferred->RunOne(); });
            const auto completed = query_task.wait_for(std::chrono::seconds(5));
            deferred->WaitUntilEmpty();
            query_task.get();
            occupied_worker.get();
            deferred->WaitUntilEmpty();
            REQUIRE(completed == std::future_status::ready);
            REQUIRE(result == reference);
        }
        if (not compact) {
            data->throw_on_first = true;
            SearchStatistics failure_stats;
            SindiWindowSearchProfile failure_profile;
            MaxHeap failure_heap(allocator.get());
            SindiWindowSearchRequest failure{windows,
                                             cell,
                                             computer,
                                             param,
                                             metadata,
                                             route,
                                             pool,
                                             allocator.get(),
                                             8,
                                             512,
                                             20,
                                             false,
                                             &failure_stats};
            REQUIRE_THROWS(SindiSearcher::SearchWindows(failure,
                                                        {SindiWindowScheduler::COUNT_ATOMIC, true},
                                                        failure_profile,
                                                        failure_heap));
            data->throw_on_first = false;
            auto rejecting = std::make_shared<RejectingPool>();
            failure.pool = std::make_shared<SafeThreadPool>(rejecting);
            REQUIRE_THROWS(SindiSearcher::SearchWindows(failure,
                                                        {SindiWindowScheduler::COUNT_ATOMIC, true},
                                                        failure_profile,
                                                        failure_heap));
            REQUIRE(rejecting->enqueues == 3);
            REQUIRE(nlohmann::json::parse(
                        SindiSearcher::DumpStatistics(failure_stats, {}, failure_profile))
                        .count("sindi_scheduler") == 0);
        }
        windows.clear();
        SearchStatistics statistics;
        SindiWindowSearchProfile profile;
        SindiWindowSearchRequest empty{windows,
                                       cell,
                                       computer,
                                       param,
                                       metadata,
                                       route,
                                       pool,
                                       allocator.get(),
                                       8,
                                       512,
                                       20,
                                       false,
                                       &statistics};
        MaxHeap heap(allocator.get());
        SindiSearcher::SearchWindows(
            empty, {SindiWindowScheduler::COST_ATOMIC, true}, profile, heap);
        REQUIRE(heap.empty());
    }
}

TEST_CASE("SINDI scheduler options integrate with both KNN entrypoints", "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    auto param = std::make_shared<SINDIParameter>();
    param->FromJson(JsonType::Parse(R"({"term_id_limit":16,"window_size":10000,
        "use_quantization":false,"use_reorder":false,"doc_prune_ratio":0})"));
    SINDI index(param, common);
    std::vector<int64_t> ids(20050);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> values(ids.size());
    std::vector<SparseVector> vectors(ids.size());
    uint32_t term = 3;
    for (uint64_t i = 0; i < ids.size(); ++i) {
        values[i] = 1 + i * 0.01F;
        vectors[i] = SparseVector{1, &term, &values[i]};
    }
    auto base = Dataset::Make()
                    ->NumElements(ids.size())
                    ->Ids(ids.data())
                    ->SparseVectors(vectors.data())
                    ->Owner(false);
    REQUIRE(index.Build(base).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors.data())->Owner(false);
    for (const auto* scheduler : {"count_deque", "count_atomic", "cost_atomic"}) {
        const auto options = std::string(R"({"sindi":{"n_candidate":20,"window_parallelism":4,
            "window_scheduler":")") +
                             scheduler + R"("}})";
        auto result = index.KnnSearch(query, 5, options, nullptr);
        REQUIRE(result->GetDim() == 5);
        REQUIRE(result->GetIds()[0] == 20049);
        auto stats = nlohmann::json::parse(result->GetStatistics());
        REQUIRE_FALSE(stats.empty());
        REQUIRE(stats.count("sindi_scheduler") == 0);
        SearchRequest request;
        request.query_ = query;
        request.topk_ = 5;
        request.params_str_ = options;
        auto requested = index.SearchWithRequest(request);
        REQUIRE(requested->GetIds()[0] == result->GetIds()[0]);
    }
}

TEST_CASE("SINDI automatic static reranking follows window parallelism", "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    auto parameter = std::make_shared<SINDIParameter>();
    parameter->FromJson(JsonType::Parse(R"({"term_id_limit":16,"window_size":10000,
        "use_quantization":false,"use_reorder":true,"rerank_type":"fp32","doc_prune_ratio":0})"));
    parameter->rerank_type = GENERATE("fp32", "fp16");
    SINDI index(parameter, common);
    std::vector<int64_t> labels(20050);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<float> values(labels.size());
    std::vector<SparseVector> vectors(labels.size());
    uint32_t term = 3;
    for (uint64_t i = 0; i < labels.size(); ++i) {
        // Keep the selected 500 scores exact in FP16, avoiding ambiguous rounded ties.
        values[i] = i < labels.size() - 500
                        ? 1.0F
                        : 2.0F + static_cast<float>(i - (labels.size() - 500)) * 0.25F;
        vectors[i] = SparseVector{1, &term, &values[i]};
    }
    auto base = Dataset::Make()
                    ->NumElements(labels.size())
                    ->Ids(labels.data())
                    ->SparseVectors(vectors.data())
                    ->Owner(false);
    REQUIRE(index.Build(base).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors.data())->Owner(false);
    for (uint32_t workers : {1U, 2U, 4U, 16U}) {
        const auto options = nlohmann::json{{"sindi",
                                             {{"n_candidate", 500},
                                              {"worker_n_candidate", 500},
                                              {"window_parallelism", workers},
                                              {"window_scheduler", "count_atomic"}}}}
                                 .dump();
        auto result = index.KnnSearch(query, 10, options, nullptr);
        for (uint32_t i = 0; i < 10; ++i) {
            REQUIRE(result->GetIds()[i] == 20049 - i);
        }
        const auto output = nlohmann::json::parse(result->GetStatistics());
        REQUIRE(output["distance_evaluations_by_phase"]["rerank"] == 500);
        REQUIRE_FALSE(SindiLastQueryProfile().enabled);
        REQUIRE(SindiLastRerankProfile().workers.empty());
        SearchRequest request;
        request.query_ = query;
        request.topk_ = 10;
        request.params_str_ = options;
        auto requested = index.SearchWithRequest(request);
        for (uint32_t i = 0; i < 10; ++i) {
            REQUIRE(requested->GetIds()[i] == result->GetIds()[i]);
            REQUIRE(requested->GetDistances()[i] == result->GetDistances()[i]);
        }
        REQUIRE(nlohmann::json::parse(
                    requested->GetStatistics())["distance_evaluations_by_phase"]["rerank"] == 500);
    }
}

TEST_CASE("SINDI range entrypoints retain statistics and scalar reranking", "[ut][SindiSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    auto parameter = std::make_shared<SINDIParameter>();
    parameter->FromJson(JsonType::Parse(R"({"term_id_limit":16,"window_size":10000,
        "use_quantization":false,"use_reorder":true,"rerank_type":"fp32","doc_prune_ratio":0})"));
    SINDI index(parameter, common);
    uint32_t term = 3;
    float values[] = {1, 2, 3};
    int64_t labels[] = {0, 1, 2};
    SparseVector vectors[] = {{1, &term, values}, {1, &term, values + 1}, {1, &term, values + 2}};
    auto base = Dataset::Make()->NumElements(3)->Ids(labels)->SparseVectors(vectors)->Owner(false);
    REQUIRE(index.Build(base).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors)->Owner(false);
    const std::string options = R"({"sindi":{"n_candidate":20,"window_parallelism":16}})";
    auto result = index.RangeSearch(query, -0.5F, options, nullptr, 2);
    REQUIRE(result->GetDim() == 2);
    REQUIRE(result->GetIds()[0] == 2);
    REQUIRE(result->GetIds()[1] == 1);
    REQUIRE_FALSE(nlohmann::json::parse(result->GetStatistics()).empty());
    REQUIRE_FALSE(SindiLastQueryProfile().enabled);
    REQUIRE(SindiLastRerankProfile().workers.empty());
    SearchRequest request;
    request.query_ = query;
    request.mode_ = SearchMode::RANGE_SEARCH;
    request.radius_ = -0.5F;
    request.limited_size_ = 2;
    request.params_str_ = options;
    const auto requested = index.SearchWithRequest(request);
    REQUIRE(requested->GetDim() == result->GetDim());
    REQUIRE(requested->GetIds()[0] == result->GetIds()[0]);
    REQUIRE_FALSE(nlohmann::json::parse(requested->GetStatistics()).empty());
    REQUIRE_FALSE(SindiLastQueryProfile().enabled);
    REQUIRE(SindiLastRerankProfile().workers.empty());
}

TEST_CASE("SINDI execution options serialize only public controls", "[ut][SindiSearcher]") {
    for (const auto* window : {"count_deque", "count_atomic", "cost_atomic"}) {
        // Removed knobs have no effect, like other unrecognized JSON fields.
        const auto json = nlohmann::json{{"sindi",
                                          {{"window_parallelism", 4},
                                           {"worker_n_candidate", 500},
                                           {"n_candidate", 500},
                                           {"window_scheduler", window},
                                           {"window_scheduler_profile", true},
                                           {"rerank_scheduler", "nnz_atomic"},
                                           {"rerank_parallelism", 8},
                                           {"rerank_batch", false},
                                           {"collect_statistics", false},
                                           {"stage_profile", true}}}};
        SINDISearchParameter original;
        original.FromJson(JsonType::Parse(json.dump()));
        REQUIRE(original.execution_options.rerank_parallelism == 4);
        REQUIRE(original.execution_options.rerank_scheduler == SindiRerankScheduler::COUNT_STATIC);
        REQUIRE(original.execution_options.rerank_batch);
        REQUIRE(original.execution_options.collect_statistics);
        REQUIRE_FALSE(original.execution_options.profile);
        REQUIRE_FALSE(original.execution_options.stage_profile);
        // Internal experimental controls remain usable without becoming user parameters.
        original.execution_options.rerank_parallelism = 8;
        original.execution_options.rerank_scheduler = SindiRerankScheduler::NNZ_ATOMIC;
        original.execution_options.profile = true;
        original.execution_options.stage_profile = true;
        original.execution_options.collect_statistics = false;
        original.execution_options.rerank_batch = false;
        const auto serialized = original.ToJson();
        for (const auto* removed : {"window_scheduler_profile",
                                    "rerank_scheduler",
                                    "rerank_parallelism",
                                    "rerank_batch",
                                    "collect_statistics",
                                    "stage_profile"}) {
            REQUIRE_FALSE(serialized[INDEX_SINDI].Contains(removed));
        }
        SINDISearchParameter restored;
        restored.FromJson(serialized);
        REQUIRE(restored.window_parallelism == 4);
        REQUIRE(restored.worker_n_candidate == 500);
        REQUIRE(restored.execution_options.scheduler == original.execution_options.scheduler);
        REQUIRE(restored.execution_options.rerank_parallelism == 4);
        REQUIRE(restored.execution_options.rerank_batch);
        REQUIRE(restored.execution_options.collect_statistics);
        REQUIRE_FALSE(restored.execution_options.profile);
        REQUIRE_FALSE(restored.execution_options.stage_profile);
        restored.FromJson(JsonType::Parse(R"({"sindi":{}})"));
        REQUIRE(restored.execution_options.scheduler == SindiWindowScheduler::COUNT_DEQUE);
        REQUIRE(restored.execution_options.rerank_parallelism == 1);
    }
}

TEST_CASE("SINDI stage accounting keeps rerank time nonnegative", "[ut][SindiSearcher]") {
    for (const double rerank : {0.0, 1.0, 3.0}) {
        {
            SindiQueryProfileScope scope;
            auto& profile = SindiLastQueryProfile();
            profile.enabled = true;
            profile.rerank_us = rerank;
            profile.statistics_counter_us = 2.0;
        }
        REQUIRE(SindiLastQueryProfile().rerank_us == std::max(0.0, rerank - 2.0));
    }
}

TEST_CASE("SINDI nested queries finish with every supplied pool worker occupied",
          "[ut][SindiSearcher]") {
    const auto workers = GENERATE(1U, 4U);
    const bool reorder = GENERATE(false, true);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<RecoverableDefaultPool>(workers);
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    // Nested SafeThreadPool wrappers must preserve the underlying pool identity.
    common.thread_pool_ = std::make_shared<SafeThreadPool>(std::make_shared<SafeThreadPool>(pool));
    auto parameter = std::make_shared<SINDIParameter>();
    parameter->FromJson(JsonType::Parse(nlohmann::json{
        {"term_id_limit", 16},
        {"window_size", 10000},
        {"use_quantization", false},
        {"use_reorder", reorder},
        {"doc_prune_ratio", 0}}.dump()));
    SINDI index(parameter, common);
    uint32_t term = 3;
    float values[]{1, 2, 3};
    int64_t labels[]{0, 1, 2};
    SparseVector vectors[]{{1, &term, values}, {1, &term, values + 1}, {1, &term, values + 2}};
    auto base = Dataset::Make()->NumElements(3)->Ids(labels)->SparseVectors(vectors)->Owner(false);
    REQUIRE(index.Build(base).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors)->Owner(false);
    const std::string options = R"({"sindi":{"n_candidate":20,"window_parallelism":4}})";
    for (const bool request_api : {false, true}) {
        std::atomic<uint32_t> entered{0};
        std::vector<DatasetPtr> results(workers);
        std::vector<std::future<void>> queries;
        for (uint32_t i = 0; i < workers; ++i) {
            queries.push_back(pool->RunQuery([&, i]() {
                entered.fetch_add(1);
                while (entered.load() < workers) {
                    std::this_thread::yield();
                }
                if (request_api) {
                    SearchRequest request;
                    request.query_ = query;
                    request.topk_ = 2;
                    request.params_str_ = options;
                    results[i] = index.SearchWithRequest(request);
                } else {
                    results[i] = index.KnnSearch(query, 2, options, nullptr);
                }
            }));
        }
        bool completed = true;
        for (auto& future : queries) {
            if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                completed = false;
                pool->RecoverQueuedWork();
            }
        }
        for (auto& future : queries) {
            future.get();
        }
        INFO("all nested queries must finish without recovery");
        REQUIRE(completed);
        for (const auto& result : results) {
            REQUIRE(result->GetDim() == 2);
            REQUIRE(result->GetIds()[0] == 2);
            REQUIRE(result->GetIds()[1] == 1);
            REQUIRE_FALSE(nlohmann::json::parse(result->GetStatistics()).empty());
        }
    }
}

TEST_CASE("SINDI inline window workers preserve reduced candidate budgets", "[ut][SindiSearcher]") {
    const bool reorder = GENERATE(false, true);
    const bool immutable = GENERATE(false, true);
    const bool custom_pool = GENERATE(false, true);
    const auto* scheduler = GENERATE("count_deque", "count_atomic", "cost_atomic");
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto deferred = std::make_shared<DeferredPool>();
    auto nested = std::make_shared<RecoverableDefaultPool>(1);
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.thread_pool_ = custom_pool ? std::make_shared<SafeThreadPool>(deferred)
                                      : std::make_shared<SafeThreadPool>(nested);
    auto parameter = std::make_shared<SINDIParameter>();
    parameter->FromJson(JsonType::Parse(nlohmann::json{
        {"term_id_limit", 16},
        {"window_size", 10000},
        {"use_quantization", false},
        {"use_reorder", reorder},
        {"immutable", immutable},
        {"doc_prune_ratio", 0}}.dump()));
    SINDI index(parameter, common);
    std::vector<int64_t> labels(40000);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<float> values(labels.size());
    std::vector<SparseVector> vectors(labels.size());
    uint32_t term = 3;
    for (uint64_t i = 0; i < labels.size(); ++i) {
        // Interleave the score ranks across all four windows. Each partition's
        // five candidates suffice to retain its share of the true global top-20.
        values[i] = static_cast<float>((i % 10000) * 4 + i / 10000);
        vectors[i] = SparseVector{1, &term, &values[i]};
    }
    auto base = Dataset::Make()
                    ->NumElements(labels.size())
                    ->Ids(labels.data())
                    ->SparseVectors(vectors.data())
                    ->Owner(false);
    REQUIRE(index.Build(base).empty());
    float query_value = 1;
    SparseVector query_vector{1, &term, &query_value};
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(&query_vector)->Owner(false);
    const auto options = nlohmann::json{{"sindi",
                                         {{"n_candidate", 20},
                                          {"window_parallelism", 4},
                                          {"worker_n_candidate", 5},
                                          {"window_scheduler", scheduler}}}}
                             .dump();
    for (const bool request_api : {false, true}) {
        DatasetPtr result;
        auto search = [&]() {
            if (request_api) {
                SearchRequest request;
                request.query_ = query;
                request.topk_ = 10;
                request.params_str_ = options;
                result = index.SearchWithRequest(request);
            } else {
                result = index.KnnSearch(query, 10, options, nullptr);
            }
        };
        auto future = custom_pool ? deferred->Enqueue(search) : nested->RunQuery(search);
        auto custom_worker = std::async(std::launch::async, [&]() {
            if (custom_pool) {
                deferred->RunOne();
            }
        });
        const auto completed = future.wait_for(std::chrono::seconds(5));
        if (completed != std::future_status::ready) {
            deferred->RecoverQueuedWork();
            nested->RecoverQueuedWork();
        }
        future.get();
        custom_worker.get();
        REQUIRE(completed == std::future_status::ready);
        REQUIRE(result->GetDim() == 10);
        for (int64_t i = 0; i < 10; ++i) {
            REQUIRE(result->GetIds()[i] == (3 - i % 4) * 10000 + 9999 - i / 4);
            REQUIRE(result->GetDistances()[i] == 1.0F - (39999.0F - static_cast<float>(i)));
        }
    }
}

TEST_CASE("SINDI custom pool identity enables external dispatch and nested inline work",
          "[ut][SindiSearcher]") {
    const bool reorder = GENERATE(false, true);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<IdentifiedCustomPool>();
    auto safe = std::make_shared<SafeThreadPool>(std::make_shared<SafeThreadPool>(pool));
    REQUIRE_FALSE(safe->RequiresInlineExecution());
    SafeThreadPool empty(static_cast<ThreadPool*>(nullptr), false);
    REQUIRE(empty.RequiresInlineExecution());
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.thread_pool_ = safe;
    auto parameter = std::make_shared<SINDIParameter>();
    parameter->FromJson(JsonType::Parse(nlohmann::json{
        {"term_id_limit", 16},
        {"window_size", 10000},
        {"use_quantization", false},
        {"use_reorder", reorder},
        {"doc_prune_ratio", 0}}.dump()));
    SINDI index(parameter, common);
    uint32_t term = 3;
    float values[]{1, 2, 3};
    int64_t labels[]{0, 1, 2};
    SparseVector vectors[]{{1, &term, values}, {1, &term, values + 1}, {1, &term, values + 2}};
    auto base = Dataset::Make()->NumElements(3)->Ids(labels)->SparseVectors(vectors)->Owner(false);
    REQUIRE(index.Build(base).empty());
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(vectors)->Owner(false);
    const std::string options = R"({"sindi":{"n_candidate":20,"window_parallelism":4}})";
    for (const bool request_api : {false, true}) {
        auto search = [&]() {
            if (request_api) {
                SearchRequest request;
                request.query_ = query;
                request.topk_ = 2;
                request.params_str_ = options;
                return index.SearchWithRequest(request);
            }
            return index.KnnSearch(query, 2, options, nullptr);
        };
        const auto before = pool->enqueues.load();
        const auto external = search();
        REQUIRE(pool->enqueues.load() > before);
        const auto after_external = pool->enqueues.load();
        std::promise<DatasetPtr> nested_result;
        auto result_future = nested_result.get_future();
        auto nested = pool->Enqueue([&]() {
            try {
                if (not safe->RequiresInlineExecution()) {
                    throw std::runtime_error("custom pool worker identity was lost");
                }
                nested_result.set_value(search());
            } catch (...) {
                nested_result.set_exception(std::current_exception());
            }
        });
        nested.get();
        auto result = result_future.get();
        REQUIRE(pool->enqueues.load() == after_external + 1);
        REQUIRE(result->GetDim() == external->GetDim());
        REQUIRE(result->GetIds()[0] == 2);
        REQUIRE(result->GetIds()[1] == 1);
    }
}

TEST_CASE("SINDI deferred pool recovery also releases later query stages", "[ut][SindiSearcher]") {
    DeferredPool pool;
    std::promise<void> first_stage;
    auto first_ready = first_stage.get_future();
    auto query = std::async(std::launch::async, [&]() {
        auto first = pool.Enqueue([]() {});
        first_stage.set_value();
        first.get();
        pool.Enqueue([]() {}).get();
    });
    first_ready.get();
    pool.RecoverQueuedWork();
    const auto completed = query.wait_for(std::chrono::seconds(5));
    if (completed != std::future_status::ready) {
        pool.RecoverQueuedWork();
    }
    query.get();
    REQUIRE(completed == std::future_status::ready);
}
