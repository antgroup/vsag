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

#include "hgraph_fgim.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "hgraph.h"
#include "impl/query_computer_pool.h"
#include "vsag/constants.h"

namespace vsag {

std::vector<InnerIdType>
HGraphFGIM::ValidateSourcesAndGetOffsets(const Vector<const HGraph*>& source_graphs,
                                         std::size_t k) {
    CHECK_ARGUMENT(source_graphs.size() >= 2, "FGIM requires at least two source HGraphs");
    const bool valid_k = k > 0 && k <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    CHECK_ARGUMENT(valid_k, "FGIM k must be positive and representable as int64_t");

    std::vector<InnerIdType> offsets{0};
    uint64_t total = 0;
    int64_t dimension = 0;
    for (uint64_t i = 0; i < source_graphs.size(); ++i) {
        const auto* source = source_graphs[i];
        CHECK_ARGUMENT(source != nullptr, "FGIM source HGraph must not be null");
        CHECK_ARGUMENT(std::find(source_graphs.begin(), source_graphs.begin() + i, source) ==
                           source_graphs.begin() + i,
                       "FGIM source HGraphs must be distinct");
        const auto count = source->GetNumElements();
        CHECK_ARGUMENT(count > 0, "FGIM source HGraphs must be non-empty");
        CHECK_ARGUMENT(source->GetNumberRemoved() == 0, "FGIM does not support deleted nodes");
        CHECK_ARGUMENT(source->data_type_ == DataTypes::DATA_TYPE_FLOAT,
                       "FGIM requires float32 vectors");
        CHECK_ARGUMENT(source->metric_ == MetricType::METRIC_TYPE_L2SQR, "FGIM requires L2");
        const bool valid_dimension = source->dim_ > 0 && (i == 0 || source->dim_ == dimension);
        CHECK_ARGUMENT(valid_dimension, "FGIM requires matching positive dimensions");
        dimension = source->dim_;
        CHECK_ARGUMENT(!source->deduplicate_storage_, "FGIM does not support deduplicate storage");
        CHECK_ARGUMENT(!source->use_conjugate_graph_,
                       "FGIM internal search does not support conjugate graph enhancement");
        CHECK_ARGUMENT(
            source->basic_flatten_codes_ != nullptr &&
                source->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32,
            "FGIM requires fp32 base storage");
        // GetVectorByInnerId and search may select precise or raw storage instead of base.
        const bool valid_precise_storage =
            !source->has_precise_reorder() ||
            (source->high_precise_codes_ != nullptr &&
             source->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32);
        CHECK_ARGUMENT(valid_precise_storage,
                       "FGIM requires fp32 precise storage when reorder is enabled");
        const bool valid_raw_storage =
            !source->create_new_raw_vector_ ||
            (source->raw_vector_ != nullptr &&
             source->raw_vector_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32);
        CHECK_ARGUMENT(valid_raw_storage, "FGIM requires fp32 raw vector storage");
        CHECK_ARGUMENT(
            static_cast<uint64_t>(count) <= std::numeric_limits<InnerIdType>::max() - total,
            "FGIM total node count exceeds the merged internal ID capacity");
        const bool fully_built = source->bottom_graph_ != nullptr && source->label_table_ != nullptr &&
                                 source->bottom_graph_->TotalCount() == count &&
                                 source->basic_flatten_codes_->TotalCount() == count &&
                                 source->label_table_->GetTotalCount() == count;
        CHECK_ARGUMENT(fully_built, "FGIM requires dense, fully built source HGraphs");
        total += static_cast<uint64_t>(count);
        offsets.push_back(static_cast<InnerIdType>(total));
    }
    return offsets;
}

void
HGraphFGIM::AppendOriginalCandidates(const HGraph& source,
                                     InnerIdType local_u,
                                     InnerIdType source_offset,
                                     FGIMNeighborList& candidates) {
    Vector<InnerIdType> neighbors(source.allocator_);
    source.bottom_graph_->GetNeighbors(local_u, neighbors);
    for (InnerIdType local_v : neighbors) {
        const bool valid_neighbor = local_v < source.GetNumElements() && local_v != local_u;
        CHECK_ARGUMENT(valid_neighbor, "FGIM encountered an invalid original neighbor");
        // Both distance operands remain in the source-local internal ID space.
        // Validated fp32/L2 storage layers contain the same vectors, so base distances
        // are comparable to CrossQuery distances; bitwise equality is not assumed.
        const float distance = source.basic_flatten_codes_->ComputePairVectors(local_u, local_v);
        candidates.push_back({source_offset + local_v, distance});
    }
}

FGIMNeighborList
HGraphFGIM::CrossQuery(const HGraph& target,
                       const float* query,
                       int64_t l,
                       SearchStatistics* stats) {
    const bool valid_query = query != nullptr && l > 0 && target.GetNumElements() > 0;
    CHECK_ARGUMENT(valid_query,
                   "FGIM CrossQuery requires a query, positive L and a non-empty target");
    CHECK_ARGUMENT(!target.use_conjugate_graph_,
                   "FGIM internal search does not support conjugate graph enhancement");
    QueryComputerPool computers(query, stats);
    QueryContext ctx{.alloc = target.allocator_, .stats = stats};
    ctx.computer_pool = &computers;
    ctx.track_distance_evaluations = stats != nullptr;

    InnerSearchParam param;
    param.ep = target.entry_point_id_;
    param.topk = 1;
    param.ef = 1;
    // Reuse HGraph's routing and bottom search; no external labels or Dataset packing.
    ctx.distance_phase = DistanceEvaluationPhase::ROUTING;
    for (int64_t level = static_cast<int64_t>(target.route_graphs_.size()) - 1; level >= 0;
         --level) {
        const auto route = target.search_one_graph(query,
                                                   target.route_graphs_[level],
                                                   target.basic_flatten_codes_,
                                                   param,
                                                   VisitedListPtr{},
                                                   &ctx);
        if (!route->Empty()) {
            param.ep = route->Top().second;
        }
    }
    const int64_t topk = std::min(l, target.GetNumElements());
    param.ef = l;
    param.topk = l;
    param.rerank_topk = topk;
    param.consider_duplicate = target.support_duplicate_;
    param.is_inner_id_allowed = target.create_search_filter(nullptr, false);
    ctx.distance_phase = DistanceEvaluationPhase::APPROXIMATE;
    auto result = target.search_one_graph(
        query, target.bottom_graph_, target.basic_flatten_codes_, param, VisitedListPtr{}, &ctx);
    if (target.use_reorder_) {
        target.reorder(query, target.get_reorder_codes(), result, topk, nullptr, ctx);
    }
    while (result->Size() > static_cast<uint64_t>(topk)) {
        result->Pop();
    }
    FGIMNeighborList neighbors;
    neighbors.reserve(result->Size());
    while (!result->Empty()) {
        const auto record = result->Top();
        result->Pop();
        if (!std::isnan(record.first)) {
            neighbors.push_back({record.second, record.first});
        }
    }
    // Canonicalize ties without changing the candidate set chosen by HGraph.
    std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    });
    return neighbors;
}

FGIMKnnGraph
HGraphFGIM::BuildInitialKnnGraph(const Vector<const HGraph*>& source_graphs, std::size_t k) {
    const auto offsets = ValidateSourcesAndGetOffsets(source_graphs, k);
    const uint64_t other_count = source_graphs.size() - 1;
    // ceil(k / (m - 1)), without overflowing k + m - 2.
    const auto query_count = static_cast<int64_t>(k / other_count + (k % other_count != 0 ? 1 : 0));
    FGIMKnnGraph graph(offsets.back());
    std::vector<float> vector(source_graphs.front()->dim_);

    for (uint64_t i = 0; i < source_graphs.size(); ++i) {
        const auto& source = *source_graphs[i];
        for (InnerIdType local_u = 0; local_u < source.GetNumElements(); ++local_u) {
            auto& candidates = graph[offsets[i] + local_u];
            AppendOriginalCandidates(source, local_u, offsets[i], candidates);
            source.GetVectorByInnerId(local_u, vector.data());
            for (uint64_t j = 0; j < source_graphs.size(); ++j) {
                if (j != i) {
                    const auto cross = CrossQuery(*source_graphs[j], vector.data(), query_count);
                    for (const auto& neighbor : cross) {
                        // CrossQuery IDs are target-local; apply the target prefix exactly once.
                        candidates.push_back({offsets[j] + neighbor.id, neighbor.distance});
                    }
                }
            }
            for (const auto& candidate : candidates) {
                CHECK_ARGUMENT(std::isfinite(candidate.distance), "FGIM requires finite distances");
            }
            std::sort(candidates.begin(),
                      candidates.end(),
                      [](const FGIMNeighbor& a, const FGIMNeighbor& b) {
                          return a.distance < b.distance ||
                                 (a.distance == b.distance && a.id < b.id);
                      });
            if (candidates.size() > k) {
                candidates.resize(k);
            }
        }
    }
    return graph;
}

}  // namespace vsag
