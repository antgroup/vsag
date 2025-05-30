
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

#include <cblas.h>

#include <iostream>
#include <vector>

#include "impl/basic_searcher.h"
#include "ivf_partition_strategy_parameter.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "vsag/dataset.h"

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
          bucket_count_(bucket_count),
          dim_(common_param.dim_){};

    virtual void
    Train(const DatasetPtr dataset) = 0;

    virtual Vector<BucketIdType>
    ClassifyDatas(const void* datas, int64_t count, BucketIdType buckets_per_data) = 0;

    virtual Vector<BucketIdType>
    ClassifyDatasForSearch(const void* datas, int64_t count, const InnerSearchParam& param) {
        return std::move(ClassifyDatas(datas, count, param.scan_bucket_size));
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
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->is_trained_);
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->dim_);
    }

    virtual void
    GetResidual(
        size_t n, const float* x, float* residuals, float* centroids, BucketIdType* assign) {
        // TODO: Directly implement c = a - b.
        memcpy(residuals, x, sizeof(float) * n * dim_);
        for (size_t i = 0; i < n; ++i) {
            BucketIdType bucket_id = assign[i];
            cblas_saxpy(dim_, -1.0, centroids + bucket_id * dim_, 1, residuals + i * dim_, 1);
        }
    }

public:
    bool is_trained_{false};

    Allocator* const allocator_{nullptr};

    BucketIdType bucket_count_{0};

    int64_t dim_{-1};
};

}  // namespace vsag
