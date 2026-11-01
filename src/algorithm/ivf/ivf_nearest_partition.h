
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

#include <optional>

#include "algorithm/inner_index_interface.h"
#include "index_common_param.h"
#include "ivf_partition_strategy.h"
#include "ivf_partition_strategy_parameter.h"
#include "query_context.h"
#include "vsag/index.h"

#ifdef VSAG_IVF_NEAREST_PARTITION_TEST_ACCESS
class IVFNearestPartitionTestAccess;
#endif

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

    Vector<BucketIdType>
    ClassifyDatasForSearch(const void* datas,
                           int64_t count,
                           const InnerSearchParam& param,
                           QueryContext* ctx) override;

    /// With ivf_train_type "balanced_kmeans", every bucket receives floor(count / k) or
    /// ceil(count / k) of the vectors; otherwise the same as ClassifyDatas.
    Vector<BucketIdType>
    ClassifyDatasForBuild(const void* datas,
                          int64_t count,
                          BucketIdType buckets_per_data,
                          bool trained_on_datas) override;

    /// With ivf_train_type "balanced_kmeans", the balanced assignment of the training
    /// vectors to the trained centroids; otherwise the same as ClassifyDatas.
    Vector<BucketIdType>
    ClassifyDatasForTrain(const void* datas, int64_t count) override;

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
#ifdef VSAG_IVF_NEAREST_PARTITION_TEST_ACCESS
    friend class ::IVFNearestPartitionTestAccess;
#endif

    void
    factory_router_index(const IndexCommonParam& common_param);

    Vector<BucketIdType>
    classify_datas_by_scan(const void* datas,
                           int64_t count,
                           BucketIdType buckets_per_data,
                           QueryContext* ctx,
                           uint64_t dots_buffer_size_bytes) const;

    /// The routing both public entry points share.
    Vector<BucketIdType>
    classify_datas_on_host(const void* datas,
                           int64_t count,
                           BucketIdType buckets_per_data,
                           QueryContext* ctx) const;

    /// Build-time assignment on the CUDA backend; nothing when the caller
    /// should use the host path instead.
    std::optional<Vector<BucketIdType>>
    classify_datas_on_device(const void* datas, int64_t count, BucketIdType buckets_per_data) const;

    /// Balanced assignment of `count` vectors to the trained centroids, scored with the
    /// routing key of the metric.
    Vector<BucketIdType>
    assign_balanced(const void* datas, int64_t count);

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

    // The balanced trainer's assignment of its training vectors under L2, kept until the
    // next ClassifyDatasForBuild, which reuses it when the Build trained on exactly the
    // vectors it places.
    Vector<BucketIdType> train_labels_{allocator_};
};

}  // namespace vsag
