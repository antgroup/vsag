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
#include <mutex>
#include <shared_mutex>

#include "bucket_interface.h"
#include "flatten_datacell_parameter.h"
#include "flatten_interface.h"
#include "utils/byte_buffer.h"

namespace vsag {

class RaBitQSplitBucketDataCell : public BucketInterface {
public:
    explicit RaBitQSplitBucketDataCell(const BucketDataCellParamPtr& param,
                                       const IndexCommonParam& common_param);

    void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id,
                   QueryContext* ctx = nullptr) override;

    void
    ScanBucketWithDistanceLowerBound(float* result_dists,
                                     float* lower_bounds,
                                     float* filter_inner_products,
                                     const ComputerInterfacePtr& computer,
                                     const BucketIdType& bucket_id,
                                     QueryContext* ctx = nullptr) override;

    float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) override;

    [[nodiscard]] bool
    SupportSplitCodeStorage() const override {
        return true;
    }

    void
    QueryWithFilterInnerProductByInnerId(float* result_dists,
                                         const float* filter_inner_products,
                                         const ComputerInterfacePtr& computer,
                                         const InnerIdType* inner_ids,
                                         InnerIdType id_count,
                                         QueryContext* ctx = nullptr) override;

    void
    QueryWithDistanceHintByInnerId(float* result_dists,
                                   const float* hint_dists,
                                   const ComputerInterfacePtr& computer,
                                   const InnerIdType* inner_ids,
                                   InnerIdType id_count,
                                   QueryContext* ctx = nullptr) override;

    float
    ComputePairVectors(BucketIdType bucket_id, InnerIdType id1, InnerIdType id2) override;

    ComputerInterfacePtr
    FactoryComputer(const void* query) override;

    void
    Train(const void* data, uint64_t count) override;

    bool
    BeginOptimizedBuild(const FlattenOptimizedBuildContext& context, InnerIdType capacity) override;

    void
    FinalizeOptimizedBuild() override;

    void
    AbortOptimizedBuild() noexcept override;

    [[nodiscard]] bool
    IsOptimizedBuildActive() const override;

    InnerIdType
    InsertVector(const void* vector, BucketIdType bucket_id, InnerIdType inner_id) override;

    void
    BatchInsertVector(const void* vectors,
                      const BucketIdType* bucket_ids,
                      const InnerIdType* inner_ids,
                      InnerIdType count,
                      InnerIdType* out_offsets) override;

    void
    InsertVectorWithOffset(const void* vector,
                           BucketIdType bucket_id,
                           InnerIdType inner_id,
                           InnerIdType offset_id) override;

    InnerIdType*
    GetInnerIds(BucketIdType bucket_id) override;

    void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) override;

    void
    GetCodesById(BucketIdType bucket_id, InnerIdType offset_id, uint8_t* data) const override;

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] InnerIdType
    GetBucketSize(BucketIdType bucket_id) override;

    void
    ExportModel(const BucketInterfacePtr& other) const override;

    void
    MergeOther(const BucketInterfacePtr& other, InnerIdType bias) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    void
    Package() override;

    void
    Unpack() override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override;

private:
    class SplitBucketComputer final : public ComputerInterface {
    public:
        SplitBucketComputer(ComputerInterfacePtr inner,
                            ComputerInterfacePtr fastscan,
                            uint64_t raw_query_size,
                            uint64_t adjustment_count,
                            uint64_t bucket_computer_count,
                            uint64_t residual_transform_size,
                            Allocator* allocator)
            : inner_(std::move(inner)),
              fastscan_(std::move(fastscan)),
              raw_query_(raw_query_size, allocator),
              query_centroid_adjustments_(adjustment_count, 0.0F, allocator),
              transformed_query_(residual_transform_size, 0.0F, allocator),
              residual_query_scratch_(residual_transform_size, 0.0F, allocator),
              bucket_inner_computers_(bucket_computer_count, nullptr, allocator),
              bucket_fastscan_computers_(bucket_computer_count, nullptr, allocator) {
        }

        ComputerInterfacePtr inner_{nullptr};
        ComputerInterfacePtr fastscan_{nullptr};
        Vector<float> raw_query_;
        Vector<float> query_centroid_adjustments_;
        Vector<float> transformed_query_;
        mutable Vector<float> residual_query_scratch_;
        mutable Vector<ComputerInterfacePtr> bucket_inner_computers_;
        mutable Vector<ComputerInterfacePtr> bucket_fastscan_computers_;
        mutable std::mutex bucket_computers_mutex_;
    };

    void
    check_valid_bucket_id(BucketIdType bucket_id) const;

    void
    check_valid_offset(BucketIdType bucket_id, InnerIdType offset_id) const;

    static const ComputerInterfacePtr&
    get_inner_computer(const ComputerInterfacePtr& computer);

    static const SplitBucketComputer&
    get_bucket_computer(const ComputerInterfacePtr& computer);

    std::pair<ComputerInterfacePtr, ComputerInterfacePtr>
    get_scan_computers(const SplitBucketComputer& computer, BucketIdType bucket_id);

    [[nodiscard]] bool
    use_l2_residual_query() const;

    void
    rebuild_residual_centroid_transforms();

    void
    ensure_residual_centroid_transforms();

    const float*
    prepare_vector(const void* vector,
                   BucketIdType bucket_id,
                   Vector<float>& prepared,
                   float& residual_bias) const;

    float
    query_centroid_adjustment(const SplitBucketComputer& computer, BucketIdType bucket_id) const;

    float
    residual_adjustment(float query_centroid_adjustment,
                        BucketIdType bucket_id,
                        InnerIdType offset_id) const;

    void
    rebuild_locations();

    void
    package_fastscan(bool force = false);

    void
    set_filter_code(Vector<uint8_t>& blocks,
                    InnerIdType offset_id,
                    const uint8_t* filter_code) const;

    void
    get_filter_code(BucketIdType bucket_id, InnerIdType offset_id, uint8_t* filter_code) const;

    void
    get_filter_code(const Vector<uint8_t>& blocks,
                    InnerIdType offset_id,
                    uint8_t* filter_code) const;

    void
    collect_filter_codes(const InnerIdType* inner_ids,
                         InnerIdType id_count,
                         Vector<uint8_t>& filter_codes,
                         Vector<float>* adjustments,
                         const SplitBucketComputer* computer) const;

    [[nodiscard]] bool
    packed_filter_codes_complete(BucketIdType bucket_id) const;

    void
    serialize_packed_filter_codes(StreamWriter& writer) const;

    void
    deserialize_packed_filter_codes(StreamReader& reader);

    static uint64_t
    pack_location(BucketIdType bucket_id, InnerIdType offset_id);

    std::pair<BucketIdType, InnerIdType>
    get_location(InnerIdType inner_id) const;

private:
    FlattenInterfacePtr codes_{nullptr};
    IndexCommonParam common_param_;
    Allocator* allocator_{nullptr};
    uint64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    Vector<Vector<InnerIdType>> inner_ids_;
    Vector<Vector<float>> residual_bias_;
    Vector<float> residual_centroid_transforms_;
    uint64_t residual_transform_size_{0};
    std::mutex residual_centroid_transforms_mutex_;
    Vector<Vector<uint8_t>> fastscan_blocks_;
    uint64_t fastscan_block_size_{0};
    mutable Vector<std::shared_mutex> bucket_mutexes_;
    Vector<uint64_t> locations_;
    mutable std::mutex locations_mutex_;
    std::mutex codes_insert_mutex_;
    FlattenOptimizedBuildInterfacePtr optimized_build_codes_{nullptr};

    static constexpr uint64_t FASTSCAN_BATCH_SIZE = 32;
    static constexpr uint64_t LOCATION_SPLIT_BIT = 32;
    static constexpr uint64_t PACKED_FILTER_STORAGE_MAGIC = 0x3154425346514252ULL;
    static constexpr uint32_t PACKED_FILTER_STORAGE_VERSION = 1;
    static constexpr InnerIdType EMPTY_INNER_ID = std::numeric_limits<InnerIdType>::max();
    static constexpr uint64_t INVALID_LOCATION = std::numeric_limits<uint64_t>::max();
};

}  // namespace vsag
