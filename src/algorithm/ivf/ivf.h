
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
#include "datacell/attribute_bucket_inverted_datacell.h"
#include "datacell/bucket_datacell.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/reorder/reorder.h"
#include "impl/searcher/basic_searcher.h"
#include "index_common_param.h"
#include "ivf_bucket_searcher.h"
#include "ivf_parameter.h"
#include "ivf_partition_strategy.h"
#include "query_context.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

/**
 * @brief IVF: Inverted File index for dense vectors.
 *
 * IVF partitions the vector space into a configurable number of buckets
 * (Voronoi cells) using a partition strategy (e.g. nearest-centroid or
 * GNO-IMI).  Each vector is assigned to one or more buckets; at search
 * time only the nprobe closest buckets are scanned, giving sub-linear
 * complexity at the cost of some recall.
 *
 * Workflow:
 *   1. Train() – build the partition centroids from a sample.
 *   2. Add()   – assign incoming vectors to their bucket(s).
 *   3. Search  – scan the nprobe nearest buckets, optionally reorder
 *               with high-precision codes.
 *
 * Supports:
 *   - Attribute-based filtering (via AttributeBucketInvertedDataCell).
 *   - Merge of pre-built shards (Merge()).
 *   - Reordering with a separate quantizer for better precision.
 *
 * @since v0.14
 */
class IVF : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit IVF(const IVFParameterPtr& param, const IndexCommonParam& common_param);

    explicit IVF(const ParamPtr& param, const IndexCommonParam& common_param)
        : IVF(std::dynamic_pointer_cast<IVFParameter>(param), common_param){};

    ~IVF() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::string
    AnalyzeIndexBySearch(const vsag::SearchRequest& request) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    DatasetPtr
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override;

    float
    CalcDistanceById(const float* query,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<IVF>(this->create_param_ptr_, param);
    }

    void
    GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const override;

    void
    GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const override;

    [[nodiscard]] IndexType
    GetIndexType() const override {
        return IndexType::IVF;
    }

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_IVF;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override;

    [[nodiscard]] int64_t
    GetNumberRemoved() const override {
        return this->delete_count_;
    }

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    std::string
    GetStats() const override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    void
    Merge(const std::vector<MergeUnit>& merge_units) override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    uint32_t
    Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE) override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Train(const DatasetPtr& data) override;

    void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override;

    void
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override;

private:
    /**
     * @brief Parse the JSON search parameter string and populate an
     *        InnerSearchParam (nprobe, ef_search, etc.).
     */
    InnerSearchParam
    create_search_param(const std::string& parameters, const FilterPtr& filter) const;

    DatasetPtr
    route_buckets_only(const DatasetPtr& query,
                       const InnerSearchParam& param,
                       QueryContext& ctx) const;

    /**
     * @brief Scan the selected buckets and return a distance heap.
     *
     * @tparam mode  KNN_SEARCH or RANGE_SEARCH.
     */
    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search(const DatasetPtr& query,
           const InnerSearchParam& param,
           QueryContext& ctx,
           ReasoningContext* reasoning_ctx = nullptr) const;

    /**
     * @brief Re-score the top candidates in @p input with high-precision
     *        codes and return the final result Dataset.
     */
    DatasetPtr
    reorder(int64_t topk,
            DistHeapPtr& input,
            const float* query,
            const InnerSearchParam& param,
            QueryContext& ctx,
            ReasoningContext* reasoning_ctx = nullptr) const;

    void
    AttachReasoningReport(const DatasetPtr& dataset_results, ReasoningContext* reasoning_ctx) const;

    /// Merge a single source shard into this index.
    void
    merge_one_unit(const MergeUnit& unit);

    /// Validate that @p unit is compatible with this index's configuration.
    void
    check_merge_illegal(const MergeUnit& unit) const;

    /// Populate location_map_ after deserialization or merge.
    void
    fill_location_map();

    /**
     * @brief Decode the packed (bucket_id, local_inner_id) pair from
     *        location_map_[inner_id].
     */
    std::pair<BucketIdType, InnerIdType>
    get_location(InnerIdType inner_id) const;

    MetadataPtr
    collect_streaming_header() const override;

    void
    serialize_streaming_body(StreamWriter& writer) const override;

    void
    deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) override;

    void
    load_streaming_body(StreamReader& reader,
                        const MetadataPtr& metadata,
                        const LoadParameters& parameters) override;

    void
    read_streaming_body(StreamReader& reader, const MetadataPtr& metadata);

    /// Recalculate and cache the memory-usage counter (throttled).
    void
    cal_memory_usage();

private:
    BucketInterfacePtr bucket_{nullptr};  // bucket storage (raw or attribute-aware)
    IVFBucketSearcherPtr bucket_searcher_{nullptr};

    IVFPartitionStrategyPtr partition_strategy_{nullptr};  // centroid / partition logic
    BucketIdType buckets_per_data_;                        // buckets each vector is assigned to

    int64_t total_elements_{0};  // total inserted (incl. deleted)
    bool is_trained_{false};     // true after Train() succeeds

    FlattenInterfacePtr reorder_codes_{nullptr};  // high-precision codes for reranking
    ReorderInterfacePtr reorder_{nullptr};        // reordering engine

    std::shared_ptr<SafeThreadPool> thread_pool_{nullptr};  // for parallel bucket scans

    /**
     * Packed mapping: location_map_[inner_id] = (bucket_id << 32) | local_id.
     * LOCATION_SPLIT_BIT controls the split position.
     */
    Vector<uint64_t> location_map_;

    static const uint64_t LOCATION_SPLIT_BIT = 32;  // bit position of bucket/local split

    std::atomic<int64_t> delete_count_{0};

    // Throttle cal_memory_usage(): skip if element delta < interval
    int64_t last_cal_memory_element_{0};
    int64_t cal_memory_element_interval_{1024L};
};
}  // namespace vsag
