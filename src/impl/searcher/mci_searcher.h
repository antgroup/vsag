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
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "index_common_param_fwd.h"
#include "metric_type.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(MCISearcher);

struct MCISearcherParam {
    uint64_t seed_count{32};
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};
    const Vector<InnerIdType>* seed_inner_ids{nullptr};
    const float* raw_vectors{nullptr};
    uint64_t dim{0};
    uint64_t raw_vector_stride{0};
    MetricType metric{MetricType::METRIC_TYPE_L2SQR};
    bool* used_raw_float_csr{nullptr};
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
