
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
#include <shared_mutex>
#include <vector>

#include "dataset_impl.h"
#include "index/index_common_param.h"
#include "index_feature_list.h"
#include "label_table.h"
#include "parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/function_exists_check.h"
#include "vsag/dataset.h"
#include "vsag/index.h"

namespace vsag {

class InnerIndexInterface;
using InnerIndexPtr = std::shared_ptr<InnerIndexInterface>;

class InnerIndexInterface {
public:
    InnerIndexInterface() = default;

    explicit InnerIndexInterface(ParamPtr index_param, const IndexCommonParam& common_param);

    virtual ~InnerIndexInterface() = default;

    constexpr static char fast_string_delimiter = '|';

    static InnerIndexPtr
    FastCreateIndex(const std::string& index_fast_str, const IndexCommonParam& common_param);

    [[nodiscard]] virtual std::string
    GetName() const = 0;

    [[nodiscard]] virtual IndexType
    GetIndexType() = 0;

    virtual void
    InitFeatures() = 0;

    virtual std::vector<int64_t>
    Add(const DatasetPtr& base) = 0;

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const = 0;

    [[nodiscard]] virtual DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const = 0;

    virtual void
    Serialize(StreamWriter& writer) const = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;

    [[nodiscard]] virtual InnerIndexPtr
    Fork(const IndexCommonParam& param) = 0;

    [[nodiscard]] virtual int64_t
    GetNumElements() const = 0;

    [[nodiscard]] virtual int64_t
    GetNumberRemoved() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetNumberRemoved");
    }

    DatasetPtr
    GetVectorByIds(const int64_t* ids, int64_t count) const;

public:
    virtual void
    Train(const DatasetPtr& base){};

    virtual std::vector<int64_t>
    Build(const DatasetPtr& base);

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const BitsetPtr& invalid) const;

    [[nodiscard]] virtual DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const;

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
    KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    [[nodiscard]] virtual DatasetPtr
    SearchWithRequest(const SearchRequest& request) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support SearchWithRequest");
    }

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

    virtual Index::Checkpoint
    ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support ContinueBuild");
    }

    virtual bool
    Remove(int64_t id) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Index doesn't support Remove");
    }

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

    virtual void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support UpdateAttribute");
    }

    virtual uint32_t
    Pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support Pretrain");
    }

    virtual uint32_t
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support Feedback");
    }

    virtual float
    CalcDistanceById(const float* query, int64_t id) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support calculate distance by id");
    }

    virtual DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const;

    virtual std::pair<int64_t, int64_t>
    GetMinAndMaxId() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetMinAndMaxId");
    }

    virtual void
    GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetExtraInfoByIds");
    }

    virtual void
    Merge(const std::vector<MergeUnit>& merge_units) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Index doesn't support Merge");
    }

    virtual InnerIndexPtr
    Clone(const IndexCommonParam& param);

    virtual InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support ExportModel");
    }

    virtual void
    SetImmutable() {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support SetImmutable");
    }

    [[nodiscard]] virtual BinarySet
    Serialize() const;

    virtual void
    Serialize(std::ostream& out_stream) const;

    virtual void
    Deserialize(const BinarySet& binary_set);

    virtual void
    Deserialize(const ReaderSet& reader_set);

    virtual void
    Deserialize(std::istream& in_stream);

    virtual uint64_t
    CalSerializeSize() const;

    [[nodiscard]] virtual bool
    CheckFeature(IndexFeature feature) const {
        return this->index_feature_list_->CheckFeature(feature);
    }

    [[nodiscard]] virtual int64_t
    GetMemoryUsage() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetMemoryUsage");
    }

    [[nodiscard]] virtual std::string
    GetMemoryUsageDetail() const {
        // TODO(deming): implement func for every types of inner index
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetMemoryUsageDetail");
    }

    [[nodiscard]] virtual uint64_t
    EstimateMemory(uint64_t num_elements) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support EstimateMemory");
    }

    [[nodiscard]] virtual int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetEstimateBuildMemory");
    }

    [[nodiscard]] virtual std::string
    GetStats() const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetStats");
    }

    [[nodiscard]] virtual bool
    CheckIdExist(int64_t id) const {
        return this->label_table_->CheckLabel(id);
    }

    virtual void
    GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetCodeByInnerId");
    }

    virtual void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Index doesn't support GetVectorByInnerId");
    }

    virtual void
    SetIO(const std::shared_ptr<Reader> reader) {
    }

public:
    LabelTablePtr label_table_{nullptr};

    Allocator* allocator_{nullptr};

    IndexFeatureListPtr index_feature_list_{nullptr};

    mutable std::shared_mutex label_lookup_mutex_{};  // lock for label_lookup_ & labels_

    const ParamPtr create_param_ptr_{nullptr};

    int64_t dim_{0};
    bool immutable_{false};

    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};

    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
};

}  // namespace vsag
