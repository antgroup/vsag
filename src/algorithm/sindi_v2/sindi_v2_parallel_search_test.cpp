// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "algorithm/sindi_v2/sindi_v2.h"
#include "datacell/immutable_sindi_term_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/searcher/search_execution_context.h"
#include "impl/searcher/sindi_parallel_search.h"
#include "io/memory_block_io/memory_block_io_parameter.h"
#include "io/memory_io/memory_io_parameter.h"
#include "unittest.h"

using namespace vsag;

namespace {
struct FilterCounters {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint32_t> leases{0};
};

class PredicateFilter final : public Filter {
public:
    PredicateFilter(std::shared_ptr<FilterCounters> counters,
                    bool fail = false,
                    std::function<bool(int64_t)> predicate = {})
        : counters_(std::move(counters)), fail_(fail), predicate_(std::move(predicate)) {
    }
    bool
    CheckValid(int64_t id) const override {
        if (entered_.exchange(true)) {
            throw std::logic_error("one evaluator entered concurrently");
        }
        struct Reset {
            std::atomic<bool>& entered;
            ~Reset() {
                entered.store(false);
            }
        } reset{entered_};
        ++counters_->calls;
        if (fail_)
            throw std::runtime_error("injected filter failure");
        return predicate_ ? predicate_(id) : id % 3 != 0;
    }

private:
    std::shared_ptr<FilterCounters> counters_;
    bool fail_;
    std::function<bool(int64_t)> predicate_;
    mutable std::atomic<bool> entered_{false};
};

class IndependentFilter final : public Filter, public ConcurrentFilterProvider {
public:
    explicit IndependentFilter(bool fail = false) : fail_(fail) {
    }
    bool
    CheckValid(int64_t) const override {
        throw std::logic_error("provider must lease an evaluator");
    }
    std::unique_ptr<ConcurrentFilterLease>
    TryLease(uint32_t maximum) const override {
        struct Lease final : ConcurrentFilterLease {
            explicit Lease(std::shared_ptr<FilterCounters> counters)
                : counters(std::move(counters)) {
                ++this->counters->leases;
            }
            ~Lease() override {
                --counters->leases;
            }
            std::shared_ptr<FilterCounters> counters;
        };
        auto lease = std::make_unique<Lease>(counters);
        for (uint32_t i = 0; i < maximum; ++i) {
            lease->evaluators.push_back(
                std::make_shared<PredicateFilter>(counters, fail_, predicate));
        }
        return lease;
    }
    std::shared_ptr<FilterCounters> counters{std::make_shared<FilterCounters>()};
    std::function<bool(int64_t)> predicate;

private:
    bool fail_;
};

void
SameResults(const DatasetPtr& actual, const DatasetPtr& expected) {
    REQUIRE(actual->GetDim() == expected->GetDim());
    for (int64_t i = 0; i < actual->GetDim(); ++i) {
        REQUIRE(actual->GetIds()[i] == expected->GetIds()[i]);
        REQUIRE(actual->GetDistances()[i] == expected->GetDistances()[i]);
    }
}
}  // namespace

TEST_CASE("SINDIV2 partitions preserve scores when every local candidate fits",
          "[ut][SINDIV2][SINDIV2Parallel]") {
    const auto quantization = GENERATE(SparseValueQuantizationType::FP32,
                                       SparseValueQuantizationType::FP16,
                                       SparseValueQuantizationType::SQ8);
    const bool reorder = GENERATE(false, true);
    const uint32_t workers = GENERATE(0U, 3U);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.dim_ = 3;
    common.search_resources_ = std::make_shared<SearchExecutionContext>(workers);
    auto parameters = std::make_shared<SINDIV2Parameter>();
    parameters->term_id_limit = 8;
    parameters->window_size = 5;
    parameters->immutable = true;
    parameters->doc_prune_ratio = 0.0F;
    parameters->use_reorder = reorder;
    parameters->remap_term_ids = GENERATE(false, true);
    parameters->rerank_type = "fp16";
    parameters->use_quantization = quantization != SparseValueQuantizationType::FP32;
    parameters->sparse_value_quant_type = quantization;
    parameters->term_io_parameter = std::make_shared<MemoryIOParameter>();
    parameters->rerank_io_parameter = std::make_shared<MemoryBlockIOParameter>();
    SINDIV2 index(parameters, common);

    // Unique scores, overlapping terms, and a partial final window provide
    // an exact reference without local candidate truncation.
    constexpr uint64_t count = 37;
    std::array<uint32_t, 3> terms{1, 2, 3};
    std::vector<std::array<float, 3>> weights(count);
    std::vector<SparseVector> vectors(count);
    std::vector<int64_t> labels(count);
    for (uint64_t i = 0; i < count; ++i) {
        weights[i] = {1.0F + static_cast<float>(i), 0.2F, 0.1F};
        vectors[i] = SparseVector{3, terms.data(), weights[i].data()};
        labels[i] = static_cast<int64_t>(100 + i);
    }
    auto base = Dataset::Make();
    base->NumElements(count)->Ids(labels.data())->SparseVectors(vectors.data())->Owner(false);
    REQUIRE(index.Build(base).empty());
    std::array<uint32_t, 2> query_terms{1, 3};
    std::array<float, 2> query_weights{1.0F, 0.5F};
    SparseVector sparse{2, query_terms.data(), query_weights.data()};
    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
    const auto request = [](uint32_t p, uint64_t budget) {
        return fmt::format(R"({{"sindi_v2":{{"window_parallelism":{},"n_candidate":37,
            "query_prune_ratio":0,"term_prune_ratio":0,"term_retain_threshold":0,
            "filter_callback_limit":{},"profile_search":true,"diagnose_search":true}}}})",
                           p,
                           budget);
    };
    // Each local quota exceeds its window's document count, so this fixture
    // has no local truncation. Exact comparisons here do not assert that pruned
    // or quota-limited approximate searches must reproduce serial IDs.
    auto serial_filter = std::make_shared<PredicateFilter>(std::make_shared<FilterCounters>());
    const auto serial = index.KnnSearch(query, 5, request(1, 0), serial_filter);
    for (const auto p : {2U, 4U}) {
        auto filter = std::make_shared<IndependentFilter>();
        const auto parallel = index.KnnSearch(query, 5, request(p, 0), filter);
        SameResults(parallel, serial);
        REQUIRE(parallel->GetStatistics({"sindi_v2_strategy"})[0] == "\"partition_search\"");
        REQUIRE(parallel->GetStatistics({"sindi_v2_execution_complete"})[0] == "true");
        const auto plan_stats = parallel->GetStatistics({"sindi_v2_partitions",
                                                         "sindi_v2_candidate_capacity",
                                                         "sindi_v2_n_candidate",
                                                         "sindi_v2_routed_windows",
                                                         "sindi_v2_windows_consumed"});
        REQUIRE(std::stoull(plan_stats[0]) == 8);
        REQUIRE(std::stoull(plan_stats[1]) == 74);
        REQUIRE(std::stoull(plan_stats[2]) == 37);
        REQUIRE(plan_stats[3] == plan_stats[4]);
        REQUIRE(filter->counters->leases.load() == 0);
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
    }
    SECTION("basic execution counters remain available without timed profiling") {
        auto options = JsonType::Parse(request(4, 0));
        options["sindi_v2"]["profile_search"].SetBool(false);
        options["sindi_v2"]["diagnose_search"].SetBool(false);
        const auto result = index.KnnSearch(query, 5, options.Dump(), nullptr);
        SameResults(result, index.KnnSearch(query, 5, request(1, 0), nullptr));
        const auto stats = result->GetStatistics(
            {"sindi_v2_profile_enabled", "sindi_v2_windows_computed", "sindi_v2_windows_consumed"});
        REQUIRE(stats[0] == "false");
        REQUIRE(stats[1] == "8");
        REQUIRE(stats[2] == "8");
        REQUIRE(common.search_resources_->busy == 0);
    }
    SECTION("ordinary filters cannot silently stand in for independent evaluators") {
        REQUIRE_THROWS(
            index.KnnSearch(query,
                            5,
                            request(4, 0),
                            std::make_shared<PredicateFilter>(std::make_shared<FilterCounters>())));
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
    }
    SECTION("one callback budget spans all partitions") {
        auto filter = std::make_shared<IndependentFilter>();
        const auto result = index.KnnSearch(query, 5, request(4, 3), filter);
        REQUIRE(filter->counters->calls.load() <= 3);
        REQUIRE(result->GetStatistics({"sindi_v2_execution_complete"})[0] == "false");
        REQUIRE(filter->counters->leases.load() == 0);
    }
    SECTION("signed residuals cannot contaminate the next window") {
        if (quantization == SparseValueQuantizationType::FP32) {
            for (uint64_t i = 0; i < count; ++i) {
                weights[i] = {i < 5 ? -100.0F : i < 10 ? 100.0F : 1.0F, 0.0F, 0.0F};
            }
            SINDIV2 signed_index(parameters, common);
            REQUIRE(signed_index.Build(base).empty());
            const auto result = signed_index.KnnSearch(query, 5, request(4, 0), nullptr);
            REQUIRE(result->GetDim() == 5);
            for (int64_t i = 0; i < result->GetDim(); ++i) {
                REQUIRE(result->GetIds()[i] >= 105);
                REQUIRE(result->GetIds()[i] < 110);
                // Floating point accumulation may differ by a few ULPs; residual
                // contamination would instead change the winning IDs or score materially.
                REQUIRE(std::abs(result->GetDistances()[i] + 99.0F) < 0.0001F);
            }
        }
    }
    SECTION("deadline and filter failure drain resources and leave the index usable") {
        REQUIRE_THROWS(index.KnnSearch(
            query, 5, R"({"sindi_v2":{"window_parallelism":4,"timeout_ms":0}})", nullptr));
        REQUIRE_THROWS(
            index.KnnSearch(query, 5, request(4, 0), std::make_shared<IndependentFilter>(true)));
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
        SameResults(index.KnnSearch(query, 5, request(4, 0), nullptr),
                    index.KnnSearch(query, 5, request(1, 0), nullptr));
    }
    SECTION("SearchWithRequest shares candidate and threshold semantics") {
        SearchRequest search;
        search.query_ = query;
        search.topk_ = 5;
        search.params_str_ = request(4, 0);
        search.threshold_ = 0.0F;
        auto json = JsonType::Parse(request(1, 0));
        json["threshold"].SetFloat(0.0F);
        SameResults(index.SearchWithRequest(search),
                    index.KnnSearch(query, 5, json.Dump(), nullptr));
    }
    SECTION("missing resources lease a serial evaluator from a provider") {
        IndexCommonParam serial_common = common;
        serial_common.search_resources_.reset();
        SINDIV2 serial_index(parameters, serial_common);
        REQUIRE(serial_index.Build(base).empty());
        auto provider = std::make_shared<IndependentFilter>();
        const auto actual = serial_index.KnnSearch(query, 5, request(4, 0), provider);
        SameResults(actual,
                    serial_index.KnnSearch(
                        query,
                        5,
                        request(1, 0),
                        std::make_shared<PredicateFilter>(std::make_shared<FilterCounters>())));
        REQUIRE(provider->counters->leases.load() == 0);
        REQUIRE(actual->GetStatistics({"sindi_v2_fallback"})[0] == "\"no_resources\"");
    }
    SECTION("one window retains the parallel deadline with just the caller slot") {
        auto one_window = Dataset::Make()
                              ->NumElements(4)
                              ->Ids(labels.data())
                              ->SparseVectors(vectors.data())
                              ->Owner(false);
        SINDIV2 small_index(parameters, common);
        REQUIRE(small_index.Build(one_window).empty());
        const auto actual = small_index.KnnSearch(query, 5, request(4, 0), nullptr);
        SameResults(actual, small_index.KnnSearch(query, 5, request(1, 0), nullptr));
        REQUIRE(actual->GetStatistics({"sindi_v2_strategy"})[0] == "\"partition_search\"");
        REQUIRE(actual->GetStatistics({"sindi_v2_granted_slots"})[0] == "1");
        REQUIRE_THROWS(small_index.KnnSearch(
            query, 5, R"({"sindi_v2":{"window_parallelism":4,"timeout_ms":0}})", nullptr));
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
    }
    SECTION("range remains serial when parallelism is requested") {
        const auto actual = index.RangeSearch(query, 2.0F, request(4, 0), nullptr, 10);
        SameResults(actual, index.RangeSearch(query, 2.0F, request(1, 0), nullptr, 10));
        REQUIRE(actual->GetStatistics({"sindi_v2_fallback"})[0] == "\"unsupported_mode\"");
    }
    SECTION("metadata routing keeps the same window order and boundary pruning") {
        std::vector<std::string> hosts(count);
        std::vector<int64_t> timestamps(count);
        for (uint64_t i = 0; i < count; ++i) {
            hosts[i] = i % 2 == 0 ? "host-a" : "host-b";
            timestamps[i] = (10 + i % 3) * SINDI_SECONDS_PER_DAY + i;
        }
        base->StringMetadata(SINDI_HOST_METADATA_NAME, hosts.data())
            ->Int64Metadata(SINDI_PUBLISH_TIME_METADATA_NAME, timestamps.data());
        SINDIV2 metadata_index(parameters, common);
        REQUIRE(metadata_index.Build(base).empty());
        std::string host = "host-a";
        int64_t begin = 10 * SINDI_SECONDS_PER_DAY;
        int64_t end = 11 * SINDI_SECONDS_PER_DAY + 100;
        query->StringMetadata(SINDI_HOST_METADATA_NAME, &host)
            ->Int64Metadata(SINDI_PUBLISH_TIME_BEGIN_METADATA_NAME, &begin)
            ->Int64Metadata(SINDI_PUBLISH_TIME_END_METADATA_NAME, &end);
        auto serial_filter = std::make_shared<PredicateFilter>(std::make_shared<FilterCounters>());
        auto parallel_filter = std::make_shared<IndependentFilter>();
        const auto expected = metadata_index.KnnSearch(query, 5, request(1, 0), serial_filter);
        const auto actual = metadata_index.KnnSearch(query, 5, request(4, 0), parallel_filter);
        SameResults(actual, expected);
    }
    SECTION("concurrent callers share admission without sharing mutable window state") {
        const auto expected = index.KnnSearch(query, 5, request(1, 0), nullptr);
        std::vector<std::future<DatasetPtr>> futures;
        for (uint32_t i = 0; i < 4; ++i) {
            futures.emplace_back(std::async(std::launch::async, [&] {
                return index.KnnSearch(query, 5, request(4, 0), nullptr);
            }));
        }
        for (auto& future : futures) {
            SameResults(future.get(), expected);
        }
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
    }
    SECTION("queries already on the shared pool compute inline without nested joins") {
        if (common.search_resources_->pool) {
            const auto expected = index.KnnSearch(query, 5, request(1, 0), nullptr);
            auto nested = common.search_resources_->pool->GeneralEnqueue(
                [&] { return index.KnnSearch(query, 5, request(4, 0), nullptr); });
            const auto actual = nested.get();
            SameResults(actual, expected);
            REQUIRE(actual->GetStatistics({"sindi_v2_granted_slots"})[0] == "1");
        }
    }
}

TEST_CASE("SINDIV2 partition options validate ratio and width",
          "[ut][SINDIV2][SINDIV2Parameter][SINDIV2Parallel]") {
    SINDIV2SearchParameter parameter;
    parameter.FromJson(JsonType::Parse(R"({"sindi_v2":{"window_parallelism":4,
        "profile_search":true,"diagnose_search":true}})"));
    REQUIRE(parameter.window_parallelism == 4);
    REQUIRE(parameter.profile_search);
    REQUIRE(parameter.diagnose_search);
    SINDIV2SearchParameter roundtrip;
    roundtrip.FromJson(parameter.ToJson());
    REQUIRE(roundtrip.window_parallelism == 4);
    REQUIRE(roundtrip.profile_search);
    REQUIRE(roundtrip.parallel_n_candidate_ratio == 2.0);
    parameter.FromJson(JsonType::Parse(R"({"sindi_v2":{
        "parallel_n_candidate_ratio":4}})"));
    REQUIRE(parameter.parallel_n_candidate_ratio == 4.0);
    for (const char* json : {R"({"sindi_v2":{"window_parallelism":0}})",
                             R"({"sindi_v2":{"window_parallelism":17}})",
                             R"({"sindi_v2":{"window_parallelism":2.5}})",
                             R"({"sindi_v2":{"parallel_n_candidate_ratio":0.5}})",
                             R"({"sindi_v2":{"profile_search":"true"}})"}) {
        REQUIRE_THROWS(parameter.FromJson(JsonType::Parse(json)));
    }
}

TEST_CASE("SINDI partition planners bound quotas independently of granted workers",
          "[ut][SINDI][SINDIV2][SINDIV2Parallel]") {
    const auto normal = SindiParallelPlan::Make(658, 300, 2, 100);
    REQUIRE(normal.quotas.size() == 6);
    REQUIRE(std::accumulate(normal.quotas.begin(), normal.quotas.end(), uint64_t{0}) == 600);
    for (auto quota : normal.quotas) REQUIRE(quota == 100);
    const auto threshold = SindiParallelPlan::Make(658, 300, 2, 300);
    REQUIRE(threshold.quotas.size() == 2);
    REQUIRE(SindiParallelPlan::Make(658, 3000, 2, 100).quotas.size() == 16);
    REQUIRE(SindiParallelPlan::Make(3, 300, 2, 100).quotas.size() == 3);
}

TEST_CASE("SINDIV2 pruned partitions keep known high scores across repeated queue handoffs",
          "[ut][SINDIV2][SINDIV2Parallel][SINDIV2PrunedPartitions]") {
    const auto quantization = GENERATE(SparseValueQuantizationType::FP32,
                                       SparseValueQuantizationType::FP16,
                                       SparseValueQuantizationType::SQ8);
    const bool reorder = GENERATE(false, true);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.dim_ = 4;
    common.search_resources_ = std::make_shared<SearchExecutionContext>(3);
    auto parameters = std::make_shared<SINDIV2Parameter>();
    parameters->term_id_limit = 8;
    parameters->window_size = 4;
    parameters->doc_prune_ratio = GENERATE(0.0F, 0.3F);
    parameters->immutable = true;
    parameters->use_reorder = reorder;
    parameters->remap_term_ids = GENERATE(false, true);
    parameters->rerank_type = "fp32";
    parameters->sparse_value_quant_type = quantization;
    parameters->use_quantization = quantization != SparseValueQuantizationType::FP32;
    parameters->term_io_parameter = std::make_shared<MemoryIOParameter>();
    parameters->rerank_io_parameter = std::make_shared<MemoryBlockIOParameter>();
    constexpr uint32_t count = 320;
    std::array<uint32_t, 4> terms{1, 2, 3, 4};
    std::vector<std::array<float, 4>> weights(count);
    std::vector<SparseVector> vectors(count);
    std::vector<int64_t> labels(count);
    for (uint32_t i = 0; i < count; ++i) {
        const float value = static_cast<float>(i + 1);
        weights[i] = {value, value, value, value};
        vectors[i] = SparseVector{4, terms.data(), weights[i].data()};
        labels[i] = 1000 + i;
    }
    auto base = Dataset::Make()
                    ->NumElements(count)
                    ->Ids(labels.data())
                    ->SparseVectors(vectors.data())
                    ->Owner(false);
    std::array<float, 4> query_weights{4, 2, 1, 0.25F};
    SparseVector sparse{4, terms.data(), query_weights.data()};
    auto query = Dataset::Make()->NumElements(1)->SparseVectors(&sparse)->Owner(false);
    SINDIV2 index(parameters, common);
    REQUIRE(index.Build(base).empty());
    const auto request = [](uint32_t width, uint64_t budget = 0) {
        return fmt::format(R"({{"sindi_v2":{{"window_parallelism":{},
            "n_candidate":12,"query_prune_ratio":0.3,
            "term_prune_ratio":0.25,"term_retain_threshold":240,
            "filter_callback_limit":{},"profile_search":true,"diagnose_search":true}}}})",
                           width,
                           budget);
    };
    // Optional build pruning drops term 4, leaving the two dominant terms intact.
    // Query pruning retains weights 4 and 2. Each term retains three of four
    // postings. The strong filter accepts only local position 2 in each window.
    // Both query terms touch that same ID, exercising duplicate posting visits.
    const std::array<int64_t, 4> expected{1318, 1314, 1310, 1306};
    for (const uint32_t width : {2U, 4U}) {
        for (uint32_t repeat = 0; repeat < 3; ++repeat) {
            auto filter = std::make_shared<IndependentFilter>();
            filter->predicate = [](int64_t id) { return id % 4 == 2; };
            const auto result = index.KnnSearch(query, 4, request(width), filter);
            REQUIRE(result->GetDim() == 4);
            std::set<int64_t> unique;
            for (uint32_t i = 0; i < expected.size(); ++i) {
                REQUIRE(result->GetIds()[i] == expected[i]);
                unique.insert(result->GetIds()[i]);
                const float value = static_cast<float>(expected[i] - 999);
                // SQ8 coarse accumulation uses byte codes, without dequantizing.
                // Rerank uses the original values; the code-space score is exact.
                const float coarse_value =
                    quantization == SparseValueQuantizationType::SQ8
                        ? static_cast<float>(static_cast<uint8_t>((value - 1.0F) / 319.0F * 255.0F))
                        : value;
                const float expected_distance =
                    1.0F - (reorder ? 7.25F * value : 6.0F * coarse_value);
                REQUIRE(result->GetDistances()[i] == expected_distance);
            }
            REQUIRE(unique.size() == 4);
            const auto stats = result->GetStatistics({"sindi_v2_partitions",
                                                      "sindi_v2_windows_computed",
                                                      "sindi_v2_windows_consumed",
                                                      "sindi_v2_scanned_postings",
                                                      "sindi_v2_granted_slots",
                                                      "sindi_v2_execution_complete",
                                                      "sindi_v2_candidates"});
            const uint32_t groups = 6;
            REQUIRE(std::stoull(stats[0]) == groups);
            REQUIRE(std::stoull(stats[1]) == 80);
            REQUIRE(stats[1] == stats[2]);
            REQUIRE(std::stoull(stats[3]) == 480);
            REQUIRE(stats[5] == "true");
            REQUIRE(std::stoull(stats[6]) == 12);
            if (common.search_resources_->slots > 1 and common.search_resources_->limit > 0) {
                REQUIRE(std::stoull(stats[4]) > 1);
            }
            for (uint32_t group = 0; group < groups; ++group) {
                const auto prefix = "sindi_v2_partition_" + std::to_string(group);
                const auto local = result->GetStatistics({prefix + "_windows",
                                                          prefix + "_candidates",
                                                          prefix + "_quota",
                                                          prefix + "_work_us"});
                REQUIRE(std::stoull(local[0]) > 4);
                REQUIRE(local[1] == local[2]);
                REQUIRE(std::stod(local[3]) > 0);
            }
            REQUIRE(filter->counters->leases.load() == 0);
            REQUIRE(common.search_resources_->busy == 0);
            REQUIRE(common.search_resources_->callers == 0);
        }
    }
    SECTION("pruned candidate consumption shares one callback budget") {
        auto filter = std::make_shared<IndependentFilter>();
        filter->predicate = [](int64_t id) { return id % 4 == 2; };
        const auto result = index.KnnSearch(query, 4, request(4, 3), filter);
        REQUIRE(filter->counters->calls.load() <= 3);
        REQUIRE(result->GetStatistics({"sindi_v2_stop_reason"})[0] == "\"callback_limit\"");
        REQUIRE(result->GetStatistics({"sindi_v2_execution_complete"})[0] == "false");
        REQUIRE(filter->counters->leases.load() == 0);
    }
    SECTION("a partial metadata window disables posting pruning") {
        std::vector<std::string> hosts(count, "other");
        // Use a low posting with a positive SQ8 code. The global minimum
        // encodes as zero and is not a candidate even on the serial full scan.
        hosts[4] = "target";
        base->StringMetadata(SINDI_HOST_METADATA_NAME, hosts.data());
        SINDIV2 metadata_index(parameters, common);
        REQUIRE(metadata_index.Build(base).empty());
        std::string host = "target";
        query->StringMetadata(SINDI_HOST_METADATA_NAME, &host);
        auto filter = std::make_shared<IndependentFilter>();
        filter->predicate = [](int64_t id) { return id == 1004; };
        const auto result = metadata_index.KnnSearch(query, 4, request(4), filter);
        REQUIRE(result->GetDim() == 1);
        REQUIRE(result->GetIds()[0] == 1004);
        REQUIRE(result->GetStatistics({"sindi_v2_scanned_postings"})[0] == "8");
        REQUIRE(filter->counters->leases.load() == 0);
    }
}

TEST_CASE("SINDIV2 final failure classification outranks callback stop after drain",
          "[ut][SINDI][SINDIV2][SINDIV2Parallel]") {
    const bool original_error = GENERATE(false, true);
    const bool terms = GENERATE(false, true);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto quantization = std::make_shared<QuantizationParams>();
    MutableSindiTermDataCell staging(
        4, 4, allocator.get(), SparseValueQuantizationType::FP32, quantization);
    uint32_t term = 1;
    float weight = 1;
    SparseVector query{1, &term, &weight};
    staging.InsertVector(query, 0);
    SindiSearchTermDataCellPtr cell = std::make_shared<ImmutableSindiTermDataCell>(
        4, 4, false, SparseValueQuantizationType::FP32, quantization, allocator.get());
    std::static_pointer_cast<ImmutableSindiTermDataCell>(cell)->AppendWindow(staging.GetWindow(0));
    SINDIV2SearchParameter search;
    search.query_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(query, search, allocator.get());
    const bool collect_diagnostics = GENERATE(false, true);
    SindiParallelStatistics diagnostics;
    diagnostics.profile_search = collect_diagnostics;
    SindiParallelState state;
    if (collect_diagnostics)
        state.diagnostics = &diagnostics;
    state.resources = std::make_shared<SearchExecutionContext>(0);
    state.requested = 2;
    state.callback_limit = 1;
    state.clear_window_scratch = true;
    state.map_filter = [](const FilterPtr& evaluator) { return evaluator; };
    auto filter = std::make_shared<IndependentFilter>();
    filter->predicate = [&](int64_t) {
        // A zero-worker resource runs this on the caller only: no clock-field race.
        // Model the final admitted callback crossing a deadline after a budget stop.
        state.RequestStop(SindiParallelState::Stop::CALLBACK_LIMIT);
        state.deadline = SindiParallelState::Clock::time_point::min();
        if (original_error)
            throw std::runtime_error("original evaluator failure");
        return true;
    };
    state.user_filter = filter;
    const std::vector<uint32_t> windows{0};
    InnerSearchParam param;
    param.ef = param.topk = 1;
    SearchStatistics statistics;
    SindiPartitionSearchRequest request{
        windows, cell, computer, param, {}, allocator.get(), 4, 1, terms, &statistics};
    MaxHeap heap(allocator.get());
    bool threw = false;
    try {
        SearchSindiPartitions(request, state, heap);
    } catch (const std::exception& error) {
        threw = true;
        const std::string message = error.what();
        REQUIRE(message.find(original_error ? "original evaluator failure" : "deadline exceeded") !=
                std::string::npos);
    }
    REQUIRE(threw);
    REQUIRE(filter->counters->calls.load() == 1);
    REQUIRE(state.stop.load() == (original_error ? SindiParallelState::Stop::ERROR
                                                 : SindiParallelState::Stop::DEADLINE));
    REQUIRE(statistics.is_timeout.load() == not original_error);
    SindiV2SearchProfile profile;
    profile.CapturePartitions(state);
    REQUIRE(std::string(profile.stop) == (original_error ? "error" : "deadline"));
    if (collect_diagnostics) {
        REQUIRE(diagnostics.slots.size() == 1);
        REQUIRE(diagnostics.partitions.size() == 1);
        REQUIRE(diagnostics.slots[0].computed == 1);
        REQUIRE(diagnostics.slots[0].attempted == 1);
        REQUIRE(diagnostics.slots[0].consumed == 0);
    } else {
        REQUIRE(state.diagnostics == nullptr);
        REQUIRE(diagnostics.slots.empty());
        REQUIRE(diagnostics.partitions.empty());
    }
    REQUIRE(state.resources->busy == 0);
    REQUIRE(state.resources->callers == 0);
    state.lease.reset();
    REQUIRE(filter->counters->leases.load() == 0);
}
