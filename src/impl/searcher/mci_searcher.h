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

#include <limits>

#include "datacell/clique_datacell.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "index_common_param_fwd.h"
#include "metric_type.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(MCISearcher);

struct MCISearcherParam {
    // Preserve the upstream live/delta traversal when only the lock policy changes.
    bool preserve_live_traversal{false};
    uint64_t seed_count{500};
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};
    const Vector<InnerIdType>* seed_inner_ids{nullptr};
    // Exact distances aligned with seed_inner_ids, valid for this synchronous query only.
    const Vector<float>* seed_distances{nullptr};
    const float* precise_vectors{nullptr};
    uint64_t dim{0};
    uint64_t precise_vector_stride{0};
    MetricType metric{MetricType::METRIC_TYPE_L2SQR};
    bool* used_precise_float_csr{nullptr};
    // Optional dense validity bitmap indexed by inner id (non-zero means valid) and its size.
    const uint8_t* valid_bitmap{nullptr};
    uint64_t valid_bitmap_size{0};
    bool* used_bitmap_fast_path{nullptr};
    FilterPtr deleted_ids_filter{nullptr};
    // Non-zero iff `seed_inner_ids` enumerates every valid point of the filter, in which case it
    // equals that count. The seed phase then already scored every valid point with exact
    // distances, so the result heap is final and clique expansion cannot change it.
    uint64_t enumerated_valid_count{0};
    // Set to true for this search when the guard above skipped the clique expansion.
    bool* skipped_expansion{nullptr};
    // Expansion early stop; see HGraphSearchParameters::mci_expansion_*.
    uint64_t expansion_idle_window{32};
    bool expansion_lower_bound{false};
    // Complete, sorted unique valid inner IDs for the optional immutable no-lock filtered CSR.
    // This list is independent of the sampled seeds and remains alive for this synchronous call.
    bool filtered_cliques{false};
    const InnerIdType* valid_inner_ids{nullptr};
    uint64_t valid_inner_count{0};
    double filtered_id_prepare_ms{0.0};
    // Maintain the clique union of the current top-ef candidates, then probe the inverse CSR of
    // unscored valid points. Requires an immutable no-lock CSR and complete valid_inner_ids.
    bool dynamic_clique_scan{false};
    // 0 retains the original fixed-point scan; 1/2 use ordered, bounded passes.
    uint32_t clique_scan_passes{0};
    bool clique_scan_gallop{false};
    // Replace the outgoing clique activation of high-incidence candidates with base HGraph
    // neighbors. Zero disables this query-only option; complete clique membership stays intact.
    uint64_t hub_clique_threshold{0};
    uint32_t hub_hgraph_hops{1};
    GraphInterfacePtr hub_graph{nullptr};
};

class MCISearcher {
public:
    explicit MCISearcher(const IndexCommonParam& common_param);

    DistHeapPtr
    Search(const CliqueDataCellPtr& cliques,
           const FlattenInterfacePtr& flatten,
           const void* query,
           const InnerSearchParam& inner_search_param,
           const MCISearcherParam& mci_param,
           QueryContext* ctx) const;

private:
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
