
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
                  Statistics& stats) const override;

    void
    GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

public:
    IVFPartitionStrategyParametersPtr ivf_partition_strategy_param_{nullptr};

    InnerIndexPtr route_index_ptr_{nullptr};

private:
    void
    factory_router_index(const IndexCommonParam& common_param);
};

}  // namespace vsag
