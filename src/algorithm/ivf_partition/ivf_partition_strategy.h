
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

#include <iostream>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "impl/searcher/basic_searcher.h"
#include "ivf_partition_strategy_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/dataset.h"
#include "vsag/expected.hpp"

namespace vsag {

class IVFPartitionStrategy;
using IVFPartitionStrategyPtr = std::shared_ptr<IVFPartitionStrategy>;

class IVFPartitionStrategy {
public:
    static void
    Clone(const IVFPartitionStrategyPtr& from, const IVFPartitionStrategyPtr& to) {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        from->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        to->Deserialize(reader);
    }

public:
    explicit IVFPartitionStrategy(const IndexCommonParam& common_param, BucketIdType bucket_count)
        : allocator_(common_param.allocator_.get()),
          thread_pool_(common_param.thread_pool_),
          bucket_count_(bucket_count),
          dim_(common_param.dim_),
          metric_type_(common_param.metric_){};

    virtual void
    Train(const DatasetPtr dataset) = 0;

    virtual Vector<BucketIdType>
    ClassifyDatas(const void* datas,
                  int64_t count,
                  BucketIdType buckets_per_data,
                  Statistics& stats) const = 0;

    virtual Vector<BucketIdType>
    ClassifyDatasForSearch(const void* datas,
                           int64_t count,
                           const InnerSearchParam& param,
                           Statistics& stats) {
        return std::move(ClassifyDatas(datas, count, param.scan_bucket_size, stats));
    }

    virtual void
    GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) = 0;

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->is_trained_);
        StreamWriter::WriteObj(writer, this->bucket_count_);
        StreamWriter::WriteObj(writer, this->dim_);
    }

    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadObj(reader, this->is_trained_);
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->dim_);
    }

    virtual void
    GetResidual(size_t n, const float* x, float* residuals, float* centroids, BucketIdType* assign);

public:
    bool is_trained_{false};

    Allocator* const allocator_{nullptr};
    SafeThreadPoolPtr thread_pool_{nullptr};

    MetricType metric_type_{MetricType::METRIC_TYPE_L2SQR};

    BucketIdType bucket_count_{0};

    int64_t dim_{-1};
};

}  // namespace vsag
