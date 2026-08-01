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
}

}  // namespace vsag
