
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
#include "algorithm/brute_force.h"
#include "algorithm/brute_force_parameter.h"
#include "algorithm/inner_index_interface.h"
#include "index_common_param.h"
#include "ivf_nearest_partition.h"
#include "ivf_partition_strategy.h"
#include "ivf_partition_strategy_parameter.h"
#include "vsag/index.h"
namespace vsag {

class GNOIMIPartition : public IVFPartitionStrategy {
public:
    explicit GNOIMIPartition(const IndexCommonParam& common_param,
                             const IVFPartitionStrategyParametersPtr& param);

    void
    Train(const DatasetPtr dataset) override;

    Vector<BucketIdType>
    ClassifyDatas(const void* datas,
                  int64_t count,
                  BucketIdType buckets_per_data,
                  Statistics& stats) const override;

    Vector<BucketIdType>
    ClassifyDatasForSearch(const void* datas,
                           int64_t count,
                           const InnerSearchParam& param,
                           Statistics& stats) override;

    void
    GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

public:
    IVFNearestPartitionTrainerType trainer_type_{IVFNearestPartitionTrainerType::KMeansTrainer};
    IndexCommonParam common_param_;
    std::shared_ptr<BruteForceParameter> param_ptr_{nullptr};
    BucketIdType bucket_count_s_{0};
    BucketIdType bucket_count_t_{0};
    Vector<float> data_centroids_s_;
    Vector<float> data_centroids_t_;
    // precomputed terms for S and T to speed up the distance computation
    Vector<float> norms_s_;
    Vector<float> norms_t_;
    Vector<float> precomputed_terms_st_;

private:
    Vector<BucketIdType>
    inner_classify_datas(BruteForce& route_index, const float* datas, int64_t count);

    void
    inner_joint_classify_datas(const float* data,
                               int64_t count,
                               BucketIdType buckets_per_data,
                               BucketIdType* result,
                               Statistics& stats) const;
};

}  // namespace vsag
