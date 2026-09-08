
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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>

#include "search_metrics_internal.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

class DatasetImpl : public Dataset, public SearchMetricsDatasetAccessor {
    using var = std::variant<int64_t,
                             const float*,
                             const char*,
                             const int8_t*,
                             const uint16_t*,
                             const int64_t*,
                             const std::string*,
                             const SparseVector*,
                             const AttributeSet*,
                             const uint32_t*,
                             const MultiVector*>;

public:
    DatasetImpl() = default;

    ~DatasetImpl() override;

    DatasetImpl(const DatasetImpl&) = delete;
    DatasetImpl&
    operator=(const DatasetImpl&) = delete;

    DatasetImpl(DatasetImpl&& other) noexcept {
        this->owner_ = other.owner_;
        other.owner_ = false;
        this->data_ = std::move(other.data_);
        this->allocator_ = other.allocator_;
        this->Statistics_ = std::move(other.Statistics_);
        this->StatisticsCacheValid_ = other.StatisticsCacheValid_;
        this->SearchMetrics_ = std::move(other.SearchMetrics_);
        this->Reasoning_ = std::move(other.Reasoning_);

        other.data_.clear();
        other.allocator_ = nullptr;
        other.Statistics_ = "{}";
        other.StatisticsCacheValid_ = true;
        other.SearchMetrics_.reset();
        other.Reasoning_ = "{}";
    }

    DatasetPtr
    Owner(bool is_owner, Allocator* allocator) override {
        this->owner_ = is_owner;
        this->allocator_ = allocator;
        return shared_from_this();
    }

    DatasetPtr
    DeepCopy(Allocator* allocator) const override;

    DatasetPtr
    Append(const vsag::DatasetPtr& other) override;

public:
    DatasetPtr
    NumElements(int64_t num_elements) override {
        this->data_[NUM_ELEMENTS] = num_elements;
        return shared_from_this();
    }

    int64_t
    GetNumElements() const override {
        if (auto iter = this->data_.find(NUM_ELEMENTS); iter != this->data_.end()) {
            return std::get<int64_t>(iter->second);
        }

        return 0;
    }

    DatasetPtr
    Dim(int64_t dim) override {
        this->data_[DIM] = dim;
        return shared_from_this();
    }

    int64_t
    GetDim() const override {
        if (auto iter = this->data_.find(DIM); iter != this->data_.end()) {
            return std::get<int64_t>(iter->second);
        }

        return 0;
    }

    DatasetPtr
    Ids(const int64_t* ids) override {
        this->data_[IDS] = ids;
        return shared_from_this();
    }

    const int64_t*
    GetIds() const override {
        if (auto iter = this->data_.find(IDS); iter != this->data_.end()) {
            return std::get<const int64_t*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Distances(const float* dists) override {
        this->data_[DISTS] = dists;
        return shared_from_this();
    }

    const float*
    GetDistances() const override {
        if (auto iter = this->data_.find(DISTS); iter != this->data_.end()) {
            return std::get<const float*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Int8Vectors(const int8_t* vectors) override {
        this->data_[INT8_VECTORS] = vectors;
        return shared_from_this();
    }

    const int8_t*
    GetInt8Vectors() const override {
        if (auto iter = this->data_.find(INT8_VECTORS); iter != this->data_.end()) {
            return std::get<const int8_t*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Float32Vectors(const float* vectors) override {
        this->data_[FLOAT32_VECTORS] = vectors;
        return shared_from_this();
    }

    const float*
    GetFloat32Vectors() const override {
        if (auto iter = this->data_.find(FLOAT32_VECTORS); iter != this->data_.end()) {
            return std::get<const float*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Float16Vectors(const uint16_t* vectors) override {
        this->data_[FLOAT16_VECTORS] = vectors;
        return shared_from_this();
    }

    const uint16_t*
    GetFloat16Vectors() const override {
        if (auto iter = this->data_.find(FLOAT16_VECTORS); iter != this->data_.end()) {
            return std::get<const uint16_t*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    SparseVectors(const SparseVector* sparse_vectors) override {
        this->data_[SPARSE_VECTORS] = sparse_vectors;
        return shared_from_this();
    }

    const SparseVector*
    GetSparseVectors() const override {
        if (auto iter = this->data_.find(SPARSE_VECTORS); iter != this->data_.end()) {
            return std::get<const SparseVector*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    AttributeSets(const AttributeSet* attr_sets) override {
        this->data_[ATTRIBUTE_SETS] = attr_sets;
        return shared_from_this();
    }

    const AttributeSet*
    GetAttributeSets() const override {
        if (auto iter = this->data_.find(ATTRIBUTE_SETS); iter != this->data_.end()) {
            return std::get<const AttributeSet*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    Paths(const std::string* paths) override {
        this->data_[DATASET_PATHS] = paths;
        return shared_from_this();
    }

    DatasetPtr
    Paths(const std::string& hierarchy_name, const std::string* paths) override {
        if (hierarchy_name.empty()) {
            return Paths(paths);
        }
        this->data_[HierarchyPathsKey(hierarchy_name)] = paths;
        return shared_from_this();
    }

    const std::string*
    GetPaths() const override {
        if (auto iter = this->data_.find(DATASET_PATHS); iter != this->data_.end()) {
            return std::get<const std::string*>(iter->second);
        }
        return nullptr;
    }

    const std::string*
    GetPaths(const std::string& hierarchy_name) const override {
        if (hierarchy_name.empty()) {
            return GetPaths();
        }
        if (auto iter = this->data_.find(HierarchyPathsKey(hierarchy_name));
            iter != this->data_.end()) {
            return std::get<const std::string*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    UInt32Metadata(const std::string& name, const uint32_t* values) override {
        this->data_[UInt32MetadataKey(name)] = values;
        return shared_from_this();
    }

    const uint32_t*
    GetUInt32Metadata(const std::string& name) const override {
        if (auto iter = this->data_.find(UInt32MetadataKey(name)); iter != this->data_.end()) {
            return std::get<const uint32_t*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    StringMetadata(const std::string& name, const std::string* values) override {
        this->data_[StringMetadataKey(name)] = values;
        return shared_from_this();
    }

    const std::string*
    GetStringMetadata(const std::string& name) const override {
        if (auto iter = this->data_.find(StringMetadataKey(name)); iter != this->data_.end()) {
            return std::get<const std::string*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    ExtraInfos(const char* extra_info) override {
        this->data_[EXTRA_INFOS] = extra_info;
        return shared_from_this();
    }

    const char*
    GetExtraInfos() const override {
        if (auto iter = this->data_.find(EXTRA_INFOS); iter != this->data_.end()) {
            return std::get<const char*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    ExtraInfoSize(int64_t extra_info_size) override {
        this->data_[EXTRA_INFO_SIZE] = extra_info_size;
        return shared_from_this();
    }

    int64_t
    GetExtraInfoSize() const override {
        if (auto iter = this->data_.find(EXTRA_INFO_SIZE); iter != this->data_.end()) {
            return std::get<int64_t>(iter->second);
        }
        return 0;
    }

    DatasetPtr
    Statistics(const std::string& statistics) override {
        std::unique_lock lock(this->StatisticsMutex_);
        this->SearchMetrics_.reset();
        this->Statistics_ = statistics;
        this->StatisticsCacheValid_ = true;
        return shared_from_this();
    }

    void
    SetSearchMetricsInternal(SearchResultMetrics metrics) override {
        std::unique_lock lock(this->StatisticsMutex_);
        this->SearchMetrics_ = std::move(metrics);
        this->StatisticsCacheValid_ = false;
    }

    std::optional<SearchResultMetrics>
    GetSearchMetricsInternal() const override {
        std::shared_lock lock(this->StatisticsMutex_);
        return this->SearchMetrics_;
    }

    std::vector<std::string>
    GetStatistics(const std::vector<std::string>& stat_keys) const override;

    std::string
    GetStatistics() const override {
        {
            std::shared_lock lock(this->StatisticsMutex_);
            if (this->StatisticsCacheValid_) {
                return this->Statistics_;
            }
        }
        std::unique_lock lock(this->StatisticsMutex_);
        if (not this->StatisticsCacheValid_ && this->SearchMetrics_.has_value()) {
            this->Statistics_ = this->SearchMetrics_->Dump();
            this->StatisticsCacheValid_ = true;
        }
        return this->Statistics_;
    }

    DatasetPtr
    Reasoning(const std::string& reasoning_json) override {
        this->Reasoning_ = reasoning_json;
        return shared_from_this();
    }

    const std::string&
    GetReasoning() const override {
        return this->Reasoning_;
    }

    DatasetPtr
    VectorCounts(const uint32_t* counts) override {
        this->data_[VECTOR_COUNTS] = counts;
        return shared_from_this();
    }

    const uint32_t*
    GetVectorCounts() const override {
        if (auto iter = this->data_.find(VECTOR_COUNTS); iter != this->data_.end()) {
            return std::get<const uint32_t*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    MultiVectors(const MultiVector* multi_vectors) override {
        this->data_[MULTI_VECTORS] = multi_vectors;
        return shared_from_this();
    }

    const MultiVector*
    GetMultiVectors() const override {
        if (auto iter = this->data_.find(MULTI_VECTORS); iter != this->data_.end()) {
            return std::get<const MultiVector*>(iter->second);
        }
        return nullptr;
    }

    DatasetPtr
    MultiVectorDim(int64_t dim) override {
        this->data_[MULTI_VECTOR_DIM] = dim;
        return shared_from_this();
    }

    int64_t
    GetMultiVectorDim() const override {
        if (auto iter = this->data_.find(MULTI_VECTOR_DIM); iter != this->data_.end()) {
            return std::get<int64_t>(iter->second);
        }
        return 0;
    }

    DatasetPtr
    SourceID(const std::string* source_id) override {
        this->data_[SOURCE_ID] = source_id;
        return shared_from_this();
    }

    const std::string*
    GetSourceID() const override {
        if (auto iter = this->data_.find(SOURCE_ID); iter != this->data_.end()) {
            return std::get<const std::string*>(iter->second);
        }
        return nullptr;
    }

    static DatasetPtr
    MakeEmptyDataset();

private:
    static const std::string&
    HierarchyPathsPrefix() {
        static const std::string prefix = std::string(DATASET_PATHS) + ":";
        return prefix;
    }

    static std::string
    HierarchyPathsKey(const std::string& hierarchy_name) {
        return HierarchyPathsPrefix() + hierarchy_name;
    }

    static bool
    IsHierarchyPathsKey(const std::string& key) {
        return key.rfind(HierarchyPathsPrefix(), 0) == 0;
    }

    static std::string
    HierarchyNameFromPathsKey(const std::string& key) {
        return key.substr(HierarchyPathsPrefix().size());
    }

    static constexpr std::string_view
    UInt32MetadataPrefix() {
        return "uint32_metadata:";
    }

    static std::string
    UInt32MetadataKey(const std::string& name) {
        return std::string(UInt32MetadataPrefix()) + name;
    }

    static bool
    IsUInt32MetadataKey(const std::string& key) {
        return key.rfind(UInt32MetadataPrefix(), 0) == 0;
    }

    static std::string
    UInt32MetadataNameFromKey(const std::string& key) {
        return key.substr(UInt32MetadataPrefix().size());
    }

    static constexpr std::string_view
    StringMetadataPrefix() {
        return "string_metadata:";
    }

    static std::string
    StringMetadataKey(const std::string& name) {
        return std::string(StringMetadataPrefix()) + name;
    }

    static bool
    IsStringMetadataKey(const std::string& key) {
        return key.rfind(StringMetadataPrefix(), 0) == 0;
    }

    static std::string
    StringMetadataNameFromKey(const std::string& key) {
        return key.substr(StringMetadataPrefix().size());
    }

private:
    bool owner_{true};
    std::unordered_map<std::string, var> data_;
    Allocator* allocator_ = nullptr;

    mutable std::shared_mutex StatisticsMutex_;
    // These mutable fields are a derived serialization cache. Populating them from a typed snapshot
    // in a const getter does not change the Dataset's observable statistics value.
    mutable std::string Statistics_{"{}"};
    mutable bool StatisticsCacheValid_{true};
    std::optional<SearchResultMetrics> SearchMetrics_;
    std::string Reasoning_{"{}"};
};

};  // namespace vsag
