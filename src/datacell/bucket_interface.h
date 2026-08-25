
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
#include <string>

#include "algorithm/ivf/ivf_partition_strategy.h"
#include "bucket_datacell_parameter.h"
#include "flatten_optimized_build_interface.h"
#include "index_common_param.h"
#include "quantization/computer.h"
#include "query_context.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(BucketInterface);

class BucketInterface {
public:
    BucketInterface() = default;

    static BucketInterfacePtr
    MakeInstance(const BucketDataCellParamPtr& param, const IndexCommonParam& common_param);

public:
    virtual void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id,
                   QueryContext* ctx = nullptr,
                   InnerIdType* scanned_inner_ids = nullptr,
                   InnerIdType max_scan_size = std::numeric_limits<InnerIdType>::max(),
                   InnerIdType* scanned_size = nullptr) = 0;

    virtual uint64_t
    ScanBucketWithFilterInnerProduct(
        float* result_dists,
        float* filter_inner_products,
        const ComputerInterfacePtr& computer,
        const BucketIdType& bucket_id,
        QueryContext* ctx = nullptr,
        InnerIdType* scanned_inner_ids = nullptr,
        InnerIdType max_scan_size = std::numeric_limits<InnerIdType>::max(),
        InnerIdType* scanned_size = nullptr) {
        (void)scanned_inner_ids;
        (void)max_scan_size;
        (void)scanned_size;
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "bucket filter-inner-product scan is not supported");
    }

    virtual float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) = 0;

    [[nodiscard]] virtual bool
    SupportSplitCodeStorage() const {
        return false;
    }

    virtual void
    QueryWithCandidateFilterInnerProductBySource(float* result_dists,
                                                 const float* hint_dists,
                                                 const float* filter_inner_products,
                                                 const BucketIdType* source_bucket_ids,
                                                 const InnerIdType* source_offset_ids,
                                                 const uint64_t* source_versions,
                                                 const ComputerInterfacePtr& computer,
                                                 const InnerIdType* inner_ids,
                                                 InnerIdType id_count,
                                                 QueryContext* ctx = nullptr) {
        (void)source_bucket_ids;
        (void)source_offset_ids;
        (void)source_versions;
        (void)filter_inner_products;
        this->QueryWithDistanceHintByInnerId(
            result_dists, hint_dists, computer, inner_ids, id_count, ctx);
    }

    virtual void
    QueryWithDistanceHintByInnerId(float* result_dists,
                                   const float* hint_dists,
                                   const ComputerInterfacePtr& computer,
                                   const InnerIdType* inner_ids,
                                   InnerIdType id_count,
                                   QueryContext* ctx = nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "bucket split-code reorder is not supported");
    }

    virtual float
    ComputePairVectors(BucketIdType bucket_id, InnerIdType id1, InnerIdType id2) = 0;

    virtual void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const BucketIdType* bucket_ids,
          const InnerIdType* offset_ids,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) {
        (void)ctx;
        for (InnerIdType i = 0; i < id_count; ++i) {
            result_dists[i] = QueryOneById(computer, bucket_ids[i], offset_ids[i]);
        }
    }

    virtual ComputerInterfacePtr
    FactoryComputer(const void* query) = 0;

    // Optional query-computer specialization for callers that already know the complete set of
    // buckets to visit. The default implementation preserves the generic per-query behavior.
    virtual ComputerInterfacePtr
    FactoryComputerForBuckets(const void* query,
                              const BucketIdType* bucket_ids,
                              uint64_t bucket_count) {
        (void)bucket_ids;
        (void)bucket_count;
        return this->FactoryComputer(query);
    }

    virtual void
    Train(const void* data, uint64_t count) = 0;

    // Optional lifecycle for bulk builds. Implementations must pre-size any storage that is
    // written concurrently and keep the normal InsertVector behavior when this returns false.
    virtual bool
    BeginOptimizedBuild(const FlattenOptimizedBuildContext& context, InnerIdType capacity) {
        (void)context;
        (void)capacity;
        return false;
    }

    virtual void
    FinalizeOptimizedBuild() {
    }

    virtual void
    AbortOptimizedBuild() noexcept {
    }

    [[nodiscard]] virtual bool
    IsOptimizedBuildActive() const {
        return false;
    }

    virtual InnerIdType
    InsertVector(const void* vector, BucketIdType bucket_id, InnerIdType inner_id) = 0;

    virtual void
    BatchInsertVector(const void* vectors,
                      const BucketIdType* bucket_ids,
                      const InnerIdType* inner_ids,
                      InnerIdType count,
                      InnerIdType* out_offsets) = 0;

    // Fixed-offset inserts may create holes: GetBucketSize() includes reserved offsets and
    // GetInnerIds() reports holes as InnerIdType::max(). Therefore inner_id must not be max.
    virtual void
    InsertVectorWithOffset(const void* vector,
                           BucketIdType bucket_id,
                           InnerIdType inner_id,
                           InnerIdType offset_id) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "InsertVectorWithOffset not implemented");
    }

    virtual InnerIdType*
    GetInnerIds(BucketIdType bucket_id) = 0;

    virtual void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) = 0;

    virtual void
    GetCodesById(BucketIdType bucket_id, InnerIdType offset_id, uint8_t* data) const = 0;

    [[nodiscard]] virtual std::string
    GetQuantizerName() = 0;

    [[nodiscard]] virtual MetricType
    GetMetricType() = 0;

    [[nodiscard]] virtual bool
    UseResidual() const {
        return this->use_residual_;
    }

    [[nodiscard]] virtual InnerIdType
    GetBucketSize(BucketIdType bucket_id) = 0;

    [[nodiscard]] virtual InnerIdType
    GetBucketScanCapacity(BucketIdType bucket_id) {
        return this->GetBucketSize(bucket_id);
    }

    virtual void
    ExportModel(const BucketInterfacePtr& other) const = 0;

    virtual void
    MergeOther(const BucketInterfacePtr& other, InnerIdType bias) = 0;

    virtual void
    SetStrategy(const IVFPartitionStrategyPtr& strategy) {
        strategy_ = strategy;
    }

public:
    virtual void
    Prefetch(BucketIdType bucket_id) {
        return this->Prefetch(bucket_id, 0);
    }

    [[nodiscard]] virtual BucketIdType
    GetBucketCount() {
        return this->bucket_count_;
    }

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->bucket_count_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

    virtual void
    FinalizeLoad() {
    }

    virtual void
    InitIO(const IOParamPtr& io_param) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "InitIO not implemented in BucketInterface");
    }

    virtual void
    Package(){};

    virtual void
    Unpack(){};

    [[nodiscard]] virtual uint64_t
    GetMemoryUsage() const = 0;

public:
    BucketIdType bucket_count_{0};
    uint32_t code_size_{0};
    IVFPartitionStrategyPtr strategy_{nullptr};
    bool use_residual_{false};
    DistanceEvaluationBackend backend_{DistanceEvaluationBackend::UNKNOWN};
};

}  // namespace vsag
