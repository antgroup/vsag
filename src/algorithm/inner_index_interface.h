
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

#include <atomic>
#include <shared_mutex>
#include <vector>

#include "data_type.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/extra_info_interface.h"
#include "dataset_impl.h"
#include "inner_index_parameter.h"
#include "metric_type.h"
#include "parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/function_exists_check.h"
#include "utils/pointer_define.h"
#include "vsag/dataset.h"
#include "vsag/index.h"

namespace vsag {

DEFINE_POINTER2(InnerIndex, InnerIndexInterface);
DEFINE_POINTER(LabelTable);
DEFINE_POINTER(IndexFeatureList);

class IndexCommonParam;

class Statistics {
public:
    [[nodiscard]] std::string
    Dump() const {
        JsonType j;
        j["is_timeout"].SetBool(is_timeout.load(std::memory_order_relaxed));
        j["dist_cmp"].SetInt(dist_cmp.load(std::memory_order_relaxed));
        j["hops"].SetInt(hops.load(std::memory_order_relaxed));
        return j.Dump();
    }

public:
    std::atomic<bool> is_timeout{false};
    std::atomic<uint32_t> dist_cmp{0};
    std::atomic<uint32_t> hops{0};
};

class InnerIndexInterface {
public:
    InnerIndexInterface() = default;

    explicit InnerIndexInterface(const InnerIndexParameterPtr& index_param,
                                 const IndexCommonParam& common_param);

    virtual ~InnerIndexInterface();

    constexpr static char fast_string_delimiter = '|';

    static InnerIndexPtr
    FastCreateIndex(const std::string& index_fast_str, const IndexCommonParam& common_param);

    virtual std::vector<int64_t>
    Add(const DatasetPtr& base) = 0;

    virtual std::string
    AnalyzeIndexBySearch(const SearchRequest& request) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support analyze index by search");
    }

    virtual std::vector<int64_t>
    Build(const DatasetPtr& base);

    virtual float
    CalcDistanceById(const DatasetPtr& vector, int64_t id) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support calculate distance by id");
    };

    virtual float
    CalcDistanceById(const float* query, int64_t id) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support calculate distance by id");
    }

    virtual DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const;

    virtual DatasetPtr
    CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count) const;

    virtual uint64_t
    CalSerializeSize() const;

    [[nodiscard]] virtual bool
    CheckFeature(IndexFeature feature) const;

    [[nodiscard]] virtual bool
    CheckIdExist(int64_t id) const;

    virtual InnerIndexPtr
    Clone(const IndexCommonParam& param);

    virtual Index::Checkpoint
    ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support ContinueBuild");
    }

    virtual void
    Deserialize(const BinarySet& binary_set);

    virtual void
    Deserialize(const ReaderSet& reader_set);

    virtual void
    Deserialize(std::istream& in_stream);

    virtual void
    Deserialize(StreamReader& reader) = 0;

    [[nodiscard]] virtual uint64_t
    EstimateMemory(uint64_t num_elements) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support EstimateMemory");
    }

    virtual DatasetPtr
    ExportIDs() const;

    virtual InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support ExportModel");
    }

    virtual uint32_t
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support Feedback");
    }

    [[nodiscard]] virtual InnerIndexPtr
    Fork(const IndexCommonParam& param) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Index doesn't support Fork");
    }

    virtual void
    GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetAttributeSetByInnerId");
    }

    virtual void
    GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetCodeByInnerId");
    }

    [[nodiscard]] virtual DatasetPtr
    GetDataByIds(const int64_t* ids, int64_t count) const;

    [[nodiscard]] virtual DatasetPtr
    GetDataByIdsWithFlag(const int64_t* ids, int64_t count, uint64_t selected_data_flag) const;

    virtual std::vector<IndexDetailInfo>
    GetIndexDetailInfos() const;

    virtual DetailDataPtr
    GetDetailDataByName(const std::string& name, IndexDetailInfo& info) const;

    [[nodiscard]] virtual int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetEstimateBuildMemory");
    }

    virtual void
    GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const;

    [[nodiscard]] virtual IndexType
    GetIndexType() const = 0;

    [[nodiscard]] virtual int64_t
    GetMemoryUsage() const {
        return static_cast<int64_t>(this->CalSerializeSize());
    }

    [[nodiscard]] virtual std::string
    GetMemoryUsageDetail() const {
        // TODO(deming): implement func for every types of inner index
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetMemoryUsageDetail");
    }

    virtual std::pair<int64_t, int64_t>
    GetMinAndMaxId() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetMinAndMaxId");
    }

    [[nodiscard]] virtual std::string
    GetName() const = 0;

    [[nodiscard]] virtual int64_t
    GetNumElements() const = 0;

    [[nodiscard]] virtual int64_t
    GetNumberRemoved() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetNumberRemoved");
    }

    [[nodiscard]] virtual std::string
    GetStats() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetStats");
    }

    virtual void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetVectorByInnerId");
    }

    virtual void
    GetSparseVectorByInnerId(InnerIdType inner_id, SparseVector* data) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetSparseVectorByInnerId");
    }

    virtual DatasetPtr
    GetVectorByIds(const int64_t* ids, int64_t count) const;

    virtual void
    InitFeatures() = 0;

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const = 0;

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const BitsetPtr& invalid) const;

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              Allocator* allocator) const {
        throw std::runtime_error("Index doesn't support new filter");
    };

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              Allocator* allocator,
              IteratorContext*& iter_ctx,
              bool is_last_filter) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support new filter");
    };

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const;

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    virtual void
    Merge(const std::vector<MergeUnit>& merge_units) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Index doesn't support Merge");
    }

    virtual uint32_t
    Pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support Pretrain");
    }

    [[nodiscard]] virtual DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const = 0;

    [[nodiscard]] virtual DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const BitsetPtr& invalid,
                int64_t limited_size = -1) const;

    [[nodiscard]] virtual DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                Allocator* allocator) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    [[nodiscard]] virtual DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const;

    [[nodiscard]] virtual DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const {
        FilterPtr filter = nullptr;
        return this->RangeSearch(query, radius, parameters, filter, limited_size);
    }

    virtual bool
    Remove(int64_t id) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Index doesn't support Remove");
    }

    [[nodiscard]] virtual DatasetPtr
    SearchWithRequest(const SearchRequest& request) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support SearchWithRequest");
    }

    virtual void
    Serialize(std::ostream& out_stream) const;

    virtual void
    Serialize(const WriteFuncType& write_func) const;

    virtual void
    Serialize(StreamWriter& writer) const = 0;

    [[nodiscard]] virtual BinarySet
    Serialize() const;

    virtual void
    SetIO(const std::shared_ptr<Reader> reader) {
    }

    virtual void
    SetImmutable() {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support SetImmutable");
    }

    virtual void
    Train(const DatasetPtr& base){};

    virtual void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support UpdateAttribute");
    }

    virtual void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support UpdateAttribute with origin attributes");
    }

    virtual bool
    UpdateExtraInfo(const DatasetPtr& new_base);

    virtual bool
    UpdateId(int64_t old_id, int64_t new_id) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support UpdateId");
    }

    virtual bool
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support UpdateVector");
    }

protected:
    void
    analyze_quantizer(JsonType& stats,
                      const float* data,
                      uint64_t sample_data_size,
                      int64_t topk,
                      const std::string& search_param) const;

    virtual DetailDataPtr
    get_detail_data_by_info(const IndexDetailInfo& info) const;

public:
    LabelTablePtr label_table_{nullptr};
    mutable std::shared_mutex label_lookup_mutex_{};  // lock for label_lookup_ & labels_

    LabelTablePtr tomb_label_table_{nullptr};

    Allocator* const allocator_{nullptr};
    int64_t dim_{0};

    mutable bool use_reorder_{false};

    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};

    IndexFeatureListUPtr index_feature_list_{nullptr};

    const InnerIndexParameterPtr create_param_ptr_{nullptr};
    bool immutable_{false};

protected:
    bool has_raw_vector_{false};
    bool has_attribute_{false};

    bool use_attribute_filter_{false};

    uint64_t extra_info_size_{0};
    ExtraInfoInterfacePtr extra_infos_{nullptr};

    uint64_t build_thread_count_{1};

    std::shared_ptr<SafeThreadPool> build_pool_{nullptr};

    AttrInvertedInterfacePtr attr_filter_index_{nullptr};
};

}  // namespace vsag
