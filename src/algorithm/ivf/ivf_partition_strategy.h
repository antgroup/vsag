
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

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "impl/searcher/basic_searcher.h"
#include "ivf_partition_strategy_parameter.h"
#include "simd/fp32_simd.h"
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
          metric_type_(common_param.metric_) {};

    virtual void
    Train(const DatasetPtr dataset) = 0;

    virtual Vector<BucketIdType>
    ClassifyDatas(const void* datas,
                  int64_t count,
                  BucketIdType buckets_per_data,
                  QueryContext* ctx) const = 0;

    virtual Vector<BucketIdType>
    ClassifyDatasForSearch(const void* datas,
                           int64_t count,
                           const InnerSearchParam& param,
                           QueryContext* ctx) {
        return std::move(ClassifyDatas(datas, count, param.scan_bucket_size, ctx));
    }

    /// Buckets of the vectors written by a Build into an empty index. A strategy that
    /// balances its buckets overrides this; by default every vector goes to its
    /// nearest buckets, exactly as Add does.
    virtual Vector<BucketIdType>
    ClassifyDatasForBuild(const void* datas, int64_t count, BucketIdType buckets_per_data) {
        return ClassifyDatas(datas, count, buckets_per_data, nullptr);
    }

    /// Buckets of training vectors for codes that depend on their bucket (residuals):
    /// the same number per vector, `size() / count`, naming every bucket a Build or Add
    /// can encode it against, so the codes are trained on residuals like the ones they
    /// will encode. By default that is the nearest bucket, where both put a vector.
    virtual Vector<BucketIdType>
    ClassifyDatasForTrain(const void* datas, int64_t count) {
        return ClassifyDatas(datas, count, 1, nullptr);
    }

    /// Writes the residuals of `datas` to their buckets of ClassifyDatasForTrain, with one
    /// row of `dim_` floats per different bucket of a vector, and returns the row count.
    /// `datas` may be the storage of `residuals`, sized for it, so one bucket per vector
    /// needs no further buffer and keeps the rows in order. Further rows are shuffled in,
    /// with a fixed seed, because product quantizers train on a prefix of the rows and
    /// have to see both kinds, from every part of the batch. A slot the routing left
    /// without a bucket adds no row.
    uint64_t
    ResidualsForTrain(const float* datas, int64_t count, Vector<float>& residuals) {
        if (count <= 0) {
            residuals.clear();
            return 0;
        }
        const auto buckets = this->ClassifyDatasForTrain(datas, count);
        const auto vectors = static_cast<uint64_t>(count);
        const auto buckets_per_data = buckets.size() / vectors;
        const auto dim = static_cast<uint64_t>(dim_);
        Vector<float> centroid(dim, allocator_);
        // The further rows first, while the first ones cannot have overwritten `datas`.
        Vector<float> further(allocator_);
        for (uint64_t i = 0; i < vectors; ++i) {
            const auto* own = buckets.data() + i * buckets_per_data;
            for (uint64_t j = 1; j < buckets_per_data; ++j) {
                if (own[j] >= 0 and own[j] != own[0]) {
                    this->GetCentroid(own[j], centroid);
                    further.resize(further.size() + dim);
                    FP32Sub(datas + i * dim,
                            centroid.data(),
                            further.data() + further.size() - dim,
                            dim);
                }
            }
        }
        residuals.resize(vectors * dim);
        uint64_t rows = 0;
        for (uint64_t i = 0; i < vectors; ++i) {
            const auto bucket = buckets[i * buckets_per_data];
            if (bucket >= 0) {
                this->GetCentroid(bucket, centroid);
                FP32Sub(datas + i * dim, centroid.data(), residuals.data() + rows * dim, dim);
                ++rows;
            }
        }
        residuals.resize(rows * dim);
        if (further.empty()) {
            return rows;
        }
        residuals.insert(residuals.end(), further.begin(), further.end());
        rows += further.size() / dim;
        std::mt19937 generator(47);
        for (uint64_t row = rows - 1; row > 0; --row) {
            const uint64_t other = generator() % (row + 1);
            if (other != row) {
                std::swap_ranges(residuals.data() + row * dim,
                                 residuals.data() + (row + 1) * dim,
                                 residuals.data() + other * dim);
            }
        }
        return rows;
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
    Deserialize(LvalueOrRvalue<StreamReader> reader) {
        StreamReader::ReadObj(reader, this->is_trained_);
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->dim_);
    }

    virtual void
    GetResidual(
        uint64_t n, const float* x, float* residuals, float* centroids, BucketIdType* assign);

    virtual uint64_t
    GetMemoryUsage() const {
        return 0;
    }

public:
    bool is_trained_{false};

    Allocator* const allocator_{nullptr};
    SafeThreadPoolPtr thread_pool_{nullptr};

    MetricType metric_type_{MetricType::METRIC_TYPE_L2SQR};

    BucketIdType bucket_count_{0};

    int64_t dim_{-1};
};

}  // namespace vsag
