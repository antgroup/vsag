// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hgraph_rabitq_searcher.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/hgraph_rabitq_fused_datacell.h"
#include "datacell/rabitq_split_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/filter/white_list_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/reorder/flatten_reorder.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("HGraph RaBitQ route honors one-bit search switch", "[ut][HGraphRaBitQSearcher][route]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 64;
    constexpr uint32_t cluster_count = 16;
    auto vectors = fixtures::generate_vectors(count, dim, false, 73);

    auto param_json = JsonType::Parse(R"({
        "codes_type": "rabitq_split",
        "io_params": {"type": "memory_io"},
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 2,
            "use_fht": true
        }
    })");
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    REQUIRE(split != nullptr);
    split->TrainFusedCodec(vectors.data(), count, cluster_count);
    auto computer = split->FactoryFusedComputer(vectors.data());
    REQUIRE(computer != nullptr);

    struct EncodedNode {
        std::vector<uint8_t> filter;
        std::vector<uint8_t> supplement;
        uint32_t cluster_id{0};
        float full_distance{0.0F};
    };
    std::vector<EncodedNode> encoded;
    encoded.reserve(count);
    for (InnerIdType id = 0; id < count; ++id) {
        EncodedNode node;
        node.filter.resize(split->OneBitCodeSize());
        node.supplement.resize(split->SupplementCodeSize());
        REQUIRE(split->EncodeFused(vectors.data() + static_cast<uint64_t>(id) * dim,
                                   node.filter.data(),
                                   node.supplement.data(),
                                   &node.cluster_id));
        REQUIRE(split->ComputeFusedFull(computer,
                                        node.cluster_id,
                                        node.filter.data(),
                                        node.supplement.data(),
                                        &node.full_distance,
                                        nullptr));
        encoded.push_back(std::move(node));
    }
    const auto [best, worst] =
        std::minmax_element(encoded.begin(), encoded.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.full_distance < rhs.full_distance;
        });
    REQUIRE(best != encoded.end());
    REQUIRE(worst != encoded.end());
    REQUIRE(worst->full_distance > best->full_distance + 1e-4F);

    EncodedNode coarse_best_full_worst = *worst;
    EncodedNode coarse_worst_full_best = *best;
    RaBitQFusedTraversalQuery traversal_query;
    REQUIRE(split->GetFusedTraversalQuery(computer, &traversal_query));
    const auto set_coarse_score = [&](EncodedNode& node, float score) {
        const float filter_add = score - traversal_query.cluster_g_add[node.cluster_id];
        constexpr float filter_rescale = 0.0F;
        constexpr float filter_error = 0.0F;
        auto* metadata = node.filter.data() + traversal_query.one_bit_metadata_offset;
        std::memcpy(metadata, &filter_add, sizeof(filter_add));
        std::memcpy(metadata + sizeof(float), &filter_rescale, sizeof(filter_rescale));
        std::memcpy(metadata + 2U * sizeof(float), &filter_error, sizeof(filter_error));
    };
    set_coarse_score(coarse_best_full_worst, -1000.0F);
    set_coarse_score(coarse_worst_full_best, 1000.0F);

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 4;
    graph_param->init_max_capacity_ = 2;
    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, split->OneBitCodeSize(), split->SupplementCodeSize(), common_param);
    graph->SetNodeCodes(0,
                        0,
                        coarse_best_full_worst.cluster_id,
                        coarse_best_full_worst.filter.data(),
                        coarse_best_full_worst.supplement.data());
    graph->SetNodeCodes(1,
                        1,
                        coarse_worst_full_best.cluster_id,
                        coarse_worst_full_best.filter.data(),
                        coarse_worst_full_best.supplement.data());
    Vector<InnerIdType> neighbor({1}, allocator.get());
    Vector<InnerIdType> no_neighbors(allocator.get());
    graph->InsertNeighborsById(0, neighbor);
    graph->InsertNeighborsById(1, no_neighbors);

    HGraphRaBitQSearcher searcher(common_param, nullptr);
    REQUIRE(searcher.Route(graph, graph, flatten, computer, 0, true) == 0);
    REQUIRE(searcher.Route(graph, graph, flatten, computer, 0, false) == 1);

    const float invalid_metadata = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(coarse_best_full_worst.filter.data() + traversal_query.one_bit_metadata_offset,
                &invalid_metadata,
                sizeof(invalid_metadata));
    std::memcpy(coarse_worst_full_best.filter.data() + traversal_query.one_bit_metadata_offset,
                &invalid_metadata,
                sizeof(invalid_metadata));
    graph->SetNodeCodes(0,
                        0,
                        coarse_best_full_worst.cluster_id,
                        coarse_best_full_worst.filter.data(),
                        coarse_best_full_worst.supplement.data());
    graph->SetNodeCodes(1,
                        1,
                        coarse_worst_full_best.cluster_id,
                        coarse_worst_full_best.filter.data(),
                        coarse_worst_full_best.supplement.data());
    REQUIRE(searcher.Route(graph, graph, flatten, computer, 0, true) == 1);

    const auto make_search_graph = [&](bool invalidate_filter_add) {
        auto search_graph_param = std::make_shared<GraphDataCellParameter>();
        search_graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        search_graph_param->max_degree_ = 4;
        search_graph_param->init_max_capacity_ = 5;
        auto search_graph = std::make_shared<HGraphRaBitQFusedDataCell>(
            search_graph_param, split->OneBitCodeSize(), split->SupplementCodeSize(), common_param);
        for (InnerIdType id = 0; id < 5; ++id) {
            auto filter = encoded[id].filter;
            auto* metadata = filter.data() + traversal_query.one_bit_metadata_offset;
            if (invalidate_filter_add) {
                std::memcpy(metadata, &invalid_metadata, sizeof(invalid_metadata));
            } else {
                const float overflowing_error = std::numeric_limits<float>::max();
                std::memcpy(
                    metadata + 2U * sizeof(float), &overflowing_error, sizeof(overflowing_error));
            }
            search_graph->SetNodeCodes(
                id, id, encoded[id].cluster_id, filter.data(), encoded[id].supplement.data());
        }
        Vector<InnerIdType> four_neighbors({1, 2, 3, 4}, allocator.get());
        search_graph->InsertNeighborsById(0, four_neighbors);
        for (InnerIdType id = 1; id < 5; ++id) {
            search_graph->InsertNeighborsById(id, no_neighbors);
        }
        return search_graph;
    };
    const auto run_search = [&](const HGraphRaBitQFusedDataCellPtr& search_graph,
                                uint32_t expected_full_count) {
        InnerSearchParam search_param;
        search_param.ep = 0;
        search_param.ef = 5;
        search_param.topk = 5;
        search_param.rerank_topk = 5;
        search_param.enable_rabitq_one_bit_search = true;
        search_param.enable_reorder = false;
        search_param.rabitq_fused_computer = computer;
        auto visited = std::make_shared<VisitedList>(5, allocator.get());
        SearchStatistics statistics;
        QueryContext context;
        context.stats = &statistics;
        auto result = searcher.Search(
            search_graph, flatten, visited, vectors.data(), search_param, &context, nullptr);
        REQUIRE(result != nullptr);
        REQUIRE(result->Size() == 5);
        REQUIRE(statistics.rabitq_filter_count.load() == 5);
        REQUIRE(statistics.rabitq_full_count.load() == expected_full_count);
        REQUIRE(statistics.rabitq_filter_fallback_full_count.load() == expected_full_count);
    };
    run_search(make_search_graph(true), 5);
    run_search(make_search_graph(false), 0);

    auto full_hint_graph = make_search_graph(false);
    InnerSearchParam full_hint_param;
    full_hint_param.ep = 0;
    full_hint_param.ef = 41;
    full_hint_param.topk = 5;
    full_hint_param.rerank_topk = 5;
    full_hint_param.enable_rabitq_one_bit_search = true;
    full_hint_param.enable_reorder = true;
    full_hint_param.rabitq_fused_computer = computer;
    auto full_hint_visited = std::make_shared<VisitedList>(5, allocator.get());
    RaBitQCandidateVector full_hint_candidates(allocator.get());
    SearchStatistics full_hint_search_statistics;
    QueryContext full_hint_search_context;
    full_hint_search_context.alloc = allocator.get();
    full_hint_search_context.stats = &full_hint_search_statistics;
    bool full_hint_search_finalized = false;
    auto full_hint_result = searcher.Search(full_hint_graph,
                                            flatten,
                                            full_hint_visited,
                                            vectors.data(),
                                            full_hint_param,
                                            &full_hint_search_context,
                                            &full_hint_candidates,
                                            &full_hint_search_finalized);
    REQUIRE(full_hint_result != nullptr);
    REQUIRE(full_hint_search_finalized);
    REQUIRE(full_hint_result->Size() == 5);
    REQUIRE(full_hint_search_statistics.rabitq_full_count.load() == 5);
    REQUIRE(full_hint_candidates.size() == 5);
    REQUIRE(std::all_of(
        full_hint_candidates.begin(), full_hint_candidates.end(), [](const auto& candidate) {
            return IsFiniteRaBitQValue(candidate.full_distance);
        }));
    const auto heap_values_by_id = [](const DistHeapPtr& heap) {
        std::vector<std::pair<InnerIdType, float>> values;
        values.reserve(heap->Size());
        const auto* data = heap->GetData();
        for (uint64_t i = 0; i < heap->Size(); ++i) {
            values.emplace_back(data[i].second, data[i].first);
        }
        std::sort(values.begin(), values.end());
        return values;
    };
    const auto full_hint_values = heap_values_by_id(full_hint_result);

    auto deferred_param = full_hint_param;
    deferred_param.ef = 5;
    auto deferred_visited = std::make_shared<VisitedList>(5, allocator.get());
    RaBitQCandidateVector deferred_candidates(allocator.get());
    SearchStatistics deferred_search_statistics;
    QueryContext deferred_search_context;
    deferred_search_context.alloc = allocator.get();
    deferred_search_context.stats = &deferred_search_statistics;
    bool deferred_search_finalized = false;
    auto deferred_result = searcher.Search(full_hint_graph,
                                           flatten,
                                           deferred_visited,
                                           vectors.data(),
                                           deferred_param,
                                           &deferred_search_context,
                                           &deferred_candidates,
                                           &deferred_search_finalized);
    REQUIRE(deferred_result != nullptr);
    REQUIRE(deferred_search_finalized);
    REQUIRE(deferred_result->Size() == 5);
    REQUIRE(deferred_search_statistics.rabitq_full_count.load() == 5);
    SearchStatistics deferred_reorder_statistics;
    QueryContext deferred_reorder_context;
    deferred_reorder_context.alloc = allocator.get();
    deferred_reorder_context.stats = &deferred_reorder_statistics;
    FlattenReorder deferred_reorder(flatten, allocator.get(), full_hint_graph);
    auto deferred_reordered = deferred_reorder.ReorderFused(deferred_result,
                                                            vectors.data(),
                                                            5,
                                                            deferred_reorder_context,
                                                            nullptr,
                                                            &deferred_candidates);
    REQUIRE(deferred_reordered != nullptr);
    REQUIRE(heap_values_by_id(deferred_reordered) == full_hint_values);
    REQUIRE(deferred_reorder_statistics.reorder_distance_count.load() == 0);
    REQUIRE(deferred_reorder_statistics.rabitq_full_count.load() == 0);
    SECTION("deferred finalize preserves reasoning reorder events") {
        ReasoningContext reasoning(allocator.get());
        Vector<int64_t> labels(allocator.get());
        UnorderedMap<int64_t, InnerIdType> label_to_inner_id(allocator.get());
        for (InnerIdType id = 0; id < 5; ++id) {
            labels.push_back(id);
            label_to_inner_id[id] = id;
        }
        reasoning.InitializeExpectedTargets(labels, label_to_inner_id);

        auto reasoning_param = deferred_param;
        reasoning_param.topk = 2;
        reasoning_param.rerank_topk = 2;
        auto reasoning_visited = std::make_shared<VisitedList>(5, allocator.get());
        RaBitQCandidateVector reasoning_candidates(allocator.get());
        QueryContext reasoning_context;
        reasoning_context.alloc = allocator.get();
        reasoning_context.reasoning_ctx = &reasoning;
        bool reasoning_finalized = false;
        auto reasoning_result = searcher.Search(full_hint_graph,
                                                flatten,
                                                reasoning_visited,
                                                vectors.data(),
                                                reasoning_param,
                                                &reasoning_context,
                                                &reasoning_candidates,
                                                &reasoning_finalized);
        REQUIRE(reasoning_result != nullptr);
        REQUIRE(reasoning_finalized);
        REQUIRE(reasoning.reorder_changes_.size() == 5);

        std::vector<std::pair<float, InnerIdType>> exact_topk;
        std::vector<InnerIdType> evicted_ids;
        for (const auto& record : reasoning.reorder_changes_) {
            REQUIRE(reasoning.expected_traces_.at(record.id).true_distance == record.dist_after);
            if (exact_topk.size() == 2 and record.dist_after > exact_topk.back().first) {
                continue;
            }
            if (exact_topk.size() == 2) {
                evicted_ids.push_back(exact_topk.back().second);
            }
            const auto position = std::lower_bound(
                exact_topk.begin(),
                exact_topk.end(),
                std::pair<float, InnerIdType>{record.dist_after, 0},
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            exact_topk.insert(position, {record.dist_after, record.id});
            if (exact_topk.size() > 2) {
                exact_topk.pop_back();
            }
        }
        for (InnerIdType id = 0; id < 5; ++id) {
            const bool expected_evicted =
                std::find(evicted_ids.begin(), evicted_ids.end(), id) != evicted_ids.end();
            REQUIRE(reasoning.expected_traces_.at(id).reorder_evicted == expected_evicted);
        }
    }
    SECTION("deferred search leaves precise reorder unfinalized without base candidates") {
        auto precise_gate_visited = std::make_shared<VisitedList>(5, allocator.get());
        bool precise_gate_finalized = true;
        auto precise_gate_result = searcher.Search(full_hint_graph,
                                                   flatten,
                                                   precise_gate_visited,
                                                   vectors.data(),
                                                   deferred_param,
                                                   nullptr,
                                                   nullptr,
                                                   &precise_gate_finalized);
        REQUIRE(precise_gate_result != nullptr);
        REQUIRE_FALSE(precise_gate_finalized);
        REQUIRE(heap_values_by_id(precise_gate_result) == full_hint_values);
    }

    SECTION("deferred finalize handles fully and partially rejected candidates") {
        auto reject_all = std::make_shared<WhiteListFilter>([](LabelType) { return false; });
        auto all_rejected_param = deferred_param;
        all_rejected_param.is_inner_id_allowed = reject_all;
        auto all_rejected_visited = std::make_shared<VisitedList>(5, allocator.get());
        RaBitQCandidateVector all_rejected_candidates(allocator.get());
        SearchStatistics all_rejected_statistics;
        QueryContext all_rejected_context;
        all_rejected_context.alloc = allocator.get();
        all_rejected_context.stats = &all_rejected_statistics;
        bool all_rejected_finalized = false;
        auto all_rejected_result = searcher.Search(full_hint_graph,
                                                   flatten,
                                                   all_rejected_visited,
                                                   vectors.data(),
                                                   all_rejected_param,
                                                   &all_rejected_context,
                                                   &all_rejected_candidates,
                                                   &all_rejected_finalized);
        REQUIRE(all_rejected_result != nullptr);
        REQUIRE(all_rejected_finalized);
        REQUIRE(all_rejected_result->Empty());
        REQUIRE(all_rejected_candidates.empty());
        REQUIRE(all_rejected_statistics.reorder_distance_count.load() == 0);

        auto allow_even =
            std::make_shared<WhiteListFilter>([](LabelType id) { return id % 2 == 0; });
        auto partially_rejected_param = deferred_param;
        partially_rejected_param.is_inner_id_allowed = allow_even;
        auto partially_rejected_visited = std::make_shared<VisitedList>(5, allocator.get());
        RaBitQCandidateVector partially_rejected_candidates(allocator.get());
        bool partially_rejected_finalized = false;
        auto partially_rejected_result = searcher.Search(full_hint_graph,
                                                         flatten,
                                                         partially_rejected_visited,
                                                         vectors.data(),
                                                         partially_rejected_param,
                                                         nullptr,
                                                         &partially_rejected_candidates,
                                                         &partially_rejected_finalized);
        REQUIRE(partially_rejected_result != nullptr);
        REQUIRE(partially_rejected_finalized);
        REQUIRE(partially_rejected_result->Size() == 3);
        REQUIRE(partially_rejected_candidates.size() == 3);
        const auto partially_rejected_values = heap_values_by_id(partially_rejected_result);
        REQUIRE(std::all_of(partially_rejected_values.begin(),
                            partially_rejected_values.end(),
                            [](const auto& value) { return value.first % 2 == 0; }));
        REQUIRE(std::all_of(partially_rejected_candidates.begin(),
                            partially_rejected_candidates.end(),
                            [](const auto& candidate) {
                                return candidate.id % 2 == 0 and
                                       IsFiniteRaBitQValue(candidate.full_distance);
                            }));
    }

    SECTION("generic fused fallback does not claim finalized results") {
        auto untrained_flatten = FlattenInterface::MakeInstance(param, common_param);
        auto untrained_split =
            std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(untrained_flatten);
        REQUIRE(untrained_split != nullptr);
        auto generic_fallback_visited = std::make_shared<VisitedList>(5, allocator.get());
        RaBitQCandidateVector generic_fallback_candidates(allocator.get());
        bool generic_fallback_finalized = true;
        auto generic_fallback_result = searcher.Search(full_hint_graph,
                                                       untrained_flatten,
                                                       generic_fallback_visited,
                                                       vectors.data(),
                                                       deferred_param,
                                                       nullptr,
                                                       &generic_fallback_candidates,
                                                       &generic_fallback_finalized);
        REQUIRE(generic_fallback_result != nullptr);
        REQUIRE(generic_fallback_result->Empty());
        REQUIRE(generic_fallback_candidates.empty());
        REQUIRE_FALSE(generic_fallback_finalized);
    }

    for (InnerIdType id = 0; id < 5; ++id) {
        auto filter = encoded[id].filter;
        const float overflowing_error = std::numeric_limits<float>::max();
        std::memcpy(filter.data() + traversal_query.one_bit_metadata_offset + 2U * sizeof(float),
                    &overflowing_error,
                    sizeof(overflowing_error));
        auto invalid_full_hint_supplement = encoded[id].supplement;
        std::memcpy(
            invalid_full_hint_supplement.data() + traversal_query.supplement_metadata_offset,
            &invalid_metadata,
            sizeof(invalid_metadata));
        full_hint_graph->SetNodeCodes(
            id, id, encoded[id].cluster_id, filter.data(), invalid_full_hint_supplement.data());
    }

    SearchStatistics full_hint_statistics;
    QueryContext full_hint_context;
    full_hint_context.alloc = allocator.get();
    full_hint_context.stats = &full_hint_statistics;
    FlattenReorder full_hint_reorder(flatten, allocator.get(), full_hint_graph);
    auto full_hint_reordered = full_hint_reorder.ReorderFused(
        full_hint_result, vectors.data(), 5, full_hint_context, nullptr, &full_hint_candidates);
    REQUIRE(full_hint_reordered != nullptr);
    REQUIRE(full_hint_reordered->Size() == 5);
    REQUIRE(heap_values_by_id(full_hint_reordered) == full_hint_values);
    REQUIRE(full_hint_statistics.reorder_distance_count.load() == 0);
    REQUIRE(full_hint_statistics.rabitq_full_count.load() == 0);

    graph->SetNodeCodes(
        0, 0, encoded[0].cluster_id, encoded[0].filter.data(), encoded[0].supplement.data());
    RaBitQCandidateVector candidates(allocator.get());
    candidates.push_back({0.0F, std::numeric_limits<float>::max(), static_cast<InnerIdType>(0)});
    SearchStatistics reorder_statistics;
    QueryContext reorder_context;
    reorder_context.alloc = allocator.get();
    reorder_context.stats = &reorder_statistics;
    FlattenReorder reorder(flatten, allocator.get(), graph);
    auto reordered =
        reorder.ReorderFused(nullptr, vectors.data(), 1, reorder_context, nullptr, &candidates);
    REQUIRE(reordered != nullptr);
    REQUIRE(reordered->Size() == 1);
    REQUIRE(reorder_statistics.reorder_distance_count.load() == 1);
    REQUIRE(reorder_statistics.rabitq_full_count.load() == 1);
    REQUIRE(reorder_statistics.rabitq_reorder_hint_full_count.load() == 0);
    REQUIRE(reorder_statistics.rabitq_reorder_fallback_full_count.load() == 1);

    auto invalid_supplement = encoded[0].supplement;
    std::memcpy(invalid_supplement.data() + traversal_query.supplement_metadata_offset,
                &invalid_metadata,
                sizeof(invalid_metadata));
    graph->SetNodeCodes(
        0, 0, encoded[0].cluster_id, encoded[0].filter.data(), invalid_supplement.data());

    RaBitQCandidateVector reused_candidates(allocator.get());
    reused_candidates.push_back(candidates.front());
    reused_candidates.push_back({encoded[0].full_distance,
                                 std::numeric_limits<float>::quiet_NaN(),
                                 static_cast<InnerIdType>(0),
                                 encoded[0].full_distance});
    SearchStatistics reused_statistics;
    QueryContext reused_context;
    reused_context.alloc = allocator.get();
    reused_context.stats = &reused_statistics;
    auto reused = reorder.ReorderFused(
        nullptr, vectors.data(), 1, reused_context, nullptr, &reused_candidates);
    REQUIRE(reused != nullptr);
    REQUIRE(reused->Size() == 1);
    REQUIRE(reused->Top().first == encoded[0].full_distance);
    REQUIRE(reused_statistics.reorder_distance_count.load() == 0);
    REQUIRE(reused_statistics.rabitq_full_count.load() == 0);

    graph->SetNodeCodes(
        1, 1, encoded[1].cluster_id, encoded[1].filter.data(), encoded[1].supplement.data());
    RaBitQCandidateVector mixed_candidates(reused_candidates, allocator.get());
    mixed_candidates.push_back({encoded[1].full_distance,
                                std::numeric_limits<float>::quiet_NaN(),
                                static_cast<InnerIdType>(1)});
    SearchStatistics mixed_statistics;
    QueryContext mixed_context;
    mixed_context.alloc = allocator.get();
    mixed_context.stats = &mixed_statistics;
    auto mixed =
        reorder.ReorderFused(nullptr, vectors.data(), 2, mixed_context, nullptr, &mixed_candidates);
    REQUIRE(mixed != nullptr);
    REQUIRE(mixed->Size() == 2);
    const auto mixed_values = heap_values_by_id(mixed);
    const std::vector<std::pair<InnerIdType, float>> expected_mixed_values{
        {static_cast<InnerIdType>(0), encoded[0].full_distance},
        {static_cast<InnerIdType>(1), encoded[1].full_distance}};
    REQUIRE(mixed_values == expected_mixed_values);
    REQUIRE(mixed_statistics.reorder_distance_count.load() == 1);
    REQUIRE(mixed_statistics.rabitq_full_count.load() == 1);
    REQUIRE(mixed_statistics.rabitq_reorder_hint_full_count.load() == 0);
    REQUIRE(mixed_statistics.rabitq_reorder_fallback_full_count.load() == 1);
    auto failed =
        reorder.ReorderFused(nullptr, vectors.data(), 1, reorder_context, nullptr, &candidates);
    REQUIRE(failed != nullptr);
    REQUIRE(failed->Empty());

    SECTION("deferred finalize drops a candidate whose full distance fails") {
        graph->SetNodeCodes(
            0, 0, encoded[0].cluster_id, encoded[0].filter.data(), encoded[0].supplement.data());
        auto invalid_neighbor_supplement = encoded[1].supplement;
        std::memcpy(invalid_neighbor_supplement.data() + traversal_query.supplement_metadata_offset,
                    &invalid_metadata,
                    sizeof(invalid_metadata));
        graph->SetNodeCodes(1,
                            1,
                            encoded[1].cluster_id,
                            encoded[1].filter.data(),
                            invalid_neighbor_supplement.data());

        auto failed_full_param = deferred_param;
        failed_full_param.ef = 2;
        failed_full_param.topk = 2;
        failed_full_param.rerank_topk = 2;
        auto failed_full_visited = std::make_shared<VisitedList>(2, allocator.get());
        RaBitQCandidateVector failed_full_candidates(allocator.get());
        SearchStatistics failed_full_statistics;
        QueryContext failed_full_context;
        failed_full_context.alloc = allocator.get();
        failed_full_context.stats = &failed_full_statistics;
        bool failed_full_finalized = false;
        auto failed_full_result = searcher.Search(graph,
                                                  flatten,
                                                  failed_full_visited,
                                                  vectors.data(),
                                                  failed_full_param,
                                                  &failed_full_context,
                                                  &failed_full_candidates,
                                                  &failed_full_finalized);
        REQUIRE(failed_full_result != nullptr);
        REQUIRE(failed_full_finalized);
        REQUIRE(failed_full_result->Size() == 1);
        REQUIRE(failed_full_result->Top().second == 0);
        REQUIRE(failed_full_candidates.size() == 2);
        const auto failed_candidate =
            std::find_if(failed_full_candidates.begin(),
                         failed_full_candidates.end(),
                         [](const auto& candidate) { return candidate.id == 1; });
        REQUIRE(failed_candidate != failed_full_candidates.end());
        REQUIRE(failed_candidate->full_distance == std::numeric_limits<float>::max());
        REQUIRE(failed_full_statistics.rabitq_full_count.load() == 2);
    }
}

}  // namespace vsag
