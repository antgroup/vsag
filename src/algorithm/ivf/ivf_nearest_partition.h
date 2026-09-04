
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

#include "algorithm/inner_index_interface.h"
#include "index_common_param.h"
#include "ivf_partition_strategy.h"
#include "ivf_partition_strategy_parameter.h"
#include "query_context.h"
#include "vsag/index.h"
namespace vsag {

class IVFNearestPartition : public IVFPartitionStrategy {
public:
    explicit IVFNearestPartition(BucketIdType bucket_count,
                                 const IndexCommonParam& common_param,
                                 IVFPartitionStrategyParametersPtr param);

    void
    Train(const DatasetPtr dataset) override;

    Vector<BucketIdType>
    ClassifyDatas(const void* datas,
                  int64_t count,
                  BucketIdType buckets_per_data,
                  QueryContext* ctx) const override;

    void
    GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override;

public:
    IVFPartitionStrategyParametersPtr ivf_partition_strategy_param_{nullptr};

    InnerIndexPtr route_index_ptr_{nullptr};

private:
    void
    factory_router_index(const IndexCommonParam& common_param);

    Vector<BucketIdType>
    classify_datas_by_scan(const void* datas,
                           int64_t count,
                           BucketIdType buckets_per_data,
                           QueryContext* ctx) const;

private:
    bool use_route_graph_{true};

    // centroids_ stores the trained centroids (normalized for cosine), and norms_ stores the
    // per-centroid constant term used by the scan routing key :
    //   - L2SQR: norms_[b] = |c_b|^2 / 2
    //   - IP/COSINE: norms_[b] = 1.0 (distance is defined as 1 - dot, and cosine centroids are
    //     normalized during training)
    Vector<float> centroids_{allocator_};
    Vector<float> norms_{allocator_};

    // Copied common param, used to lazily create the routing HGraph when a route-graph-layout
    // index is deserialized by a reader configured with use_route_graph=false.
    IndexCommonParam common_param_;
};

}  // namespace vsag
