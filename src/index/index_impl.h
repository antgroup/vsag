
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
#include "common.h"
#include "vsag/index.h"

namespace vsag {

GENERATE_HAS_STATIC_CLASS_FUNCTION(CheckAndMappingExternalParam,
                                   ParamPtr,
                                   std::declval<const JsonType&>(),
                                   std::declval<const IndexCommonParam&>());

template <class T>
class IndexImpl : public Index {
    static_assert(std::is_base_of<InnerIndexInterface, T>::value);
    static_assert(has_static_CheckAndMappingExternalParam<T>::value);

public:
    IndexImpl(const JsonType& external_param, const IndexCommonParam& common_param)
        : Index(), common_param_(common_param) {
        auto param_ptr = T::CheckAndMappingExternalParam(external_param, common_param);
        this->inner_index_ = std::make_shared<T>(param_ptr, common_param);
        this->inner_index_->InitFeatures();
    }

    IndexImpl(InnerIndexPtr inner_index, const IndexCommonParam& common_param)
        : inner_index_(std::move(inner_index)), common_param_(common_param) {
        this->inner_index_->InitFeatures();
    }

    ~IndexImpl() override {
        this->inner_index_.reset();
    }

public:
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(
                Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "immutable index no support build"));
        }
        SAFE_CALL(return this->inner_index_->Build(base));
    }

    IndexType
    GetIndexType() override {
        return this->inner_index_->GetIndexType();
    }

    tl::expected<void, Error>
    Train(const DatasetPtr& data) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(
                Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "immutable index no support train"));
        }
        SAFE_CALL(this->inner_index_->Train(data));
    }

    tl::expected<Checkpoint, Error>
    ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "immutable index no support continue build"));
        }
        SAFE_CALL(return this->inner_index_->ContinueBuild(base, binary_set));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(
                Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "immutable index no support add"));
        }
        SAFE_CALL(return this->inner_index_->Add(base));
    }

    tl::expected<bool, Error>
    Remove(int64_t id) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(
                Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "immutable index no support remove"));
        }
        SAFE_CALL(return this->inner_index_->Remove(id));
    }

    tl::expected<bool, Error>
    UpdateId(int64_t old_id, int64_t new_id) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "immutable index no support update id"));
        }
        SAFE_CALL(return this->inner_index_->UpdateId(old_id, new_id));
    }

    tl::expected<bool, Error>
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "immutable index no support update vector"));
        }
        SAFE_CALL(return this->inner_index_->UpdateVector(id, new_base, force_update));
    }

    virtual tl::expected<void, Error>
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "immutable index no support update attribute"));
        }
        SAFE_CALL(this->inner_index_->UpdateAttribute(id, new_attrs));
    }

    [[nodiscard]] tl::expected<DatasetPtr, Error>
    SearchWithRequest(const SearchRequest& request) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->SearchWithRequest(request));
    }

    [[nodiscard]] tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->KnnSearch(query, k, parameters, invalid));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->KnnSearch(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->KnnSearch(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        if (search_param.is_iter_filter) {
            SAFE_CALL(return this->inner_index_->KnnSearch(query,
                                                           k,
                                                           search_param.parameters,
                                                           search_param.filter,
                                                           search_param.allocator,
                                                           search_param.iter_ctx,
                                                           search_param.is_last_search));
        } else {
            SAFE_CALL(return this->inner_index_->KnnSearch(
                query, k, search_param.parameters, search_param.filter, search_param.allocator));
        }
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              IteratorContext*& iter_ctx,
              bool is_last_filter) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->KnnSearch(
            query, k, parameters, filter, nullptr, iter_ctx, is_last_filter));
    }

    [[nodiscard]] tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->RangeSearch(query, radius, parameters, limited_size));
    }

    [[nodiscard]] tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->RangeSearch(
            query, radius, parameters, invalid, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->RangeSearch(
            query, radius, parameters, filter, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override {
        if (GetNumElements() == 0) {
            return DatasetImpl::MakeEmptyDataset();
        }
        SAFE_CALL(return this->inner_index_->RangeSearch(
            query, radius, parameters, filter, limited_size));
    }

    tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids,
             uint32_t k,
             const std::string& parameters) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "immutable index no support pretrain"));
        }
        SAFE_CALL(return this->inner_index_->Pretrain(base_tag_ids, k, parameters));
    }

    tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "immutable index no support feedback"));
        }
        SAFE_CALL(return this->inner_index_->Feedback(query, k, parameters, global_optimum_tag_id));
    }

    tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return this->inner_index_->CalcDistanceById(vector, id));
    }

    tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const override {
        SAFE_CALL(return this->inner_index_->CalDistanceById(query, ids, count));
    }

    virtual tl::expected<void, Error>
    GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const override {
        SAFE_CALL(this->inner_index_->GetExtraInfoByIds(ids, count, extra_infos));
    };

    tl::expected<std::pair<int64_t, int64_t>, Error>
    GetMinAndMaxId() const override {
        SAFE_CALL(return this->inner_index_->GetMinAndMaxId());
    }

    virtual tl::expected<DatasetPtr, Error>
    GetRawVectorByIds(const int64_t* ids, int64_t count) const override {
        if (not CheckFeature(IndexFeature::SUPPORT_GET_VECTOR_BY_IDS)) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "index no support to get raw vector by ids"));
        }
        SAFE_CALL(return this->inner_index_->GetVectorByIds(ids, count));
    };

    tl::expected<void, Error>
    Merge(const std::vector<MergeUnit>& merge_units) override {
        if (this->inner_index_->immutable_) {
            return tl::unexpected(
                Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "immutable index no support merge"));
        }
        SAFE_CALL(this->inner_index_->Merge(merge_units));
    }

    tl::expected<IndexPtr, Error>
    Clone() const override {
        auto clone_value = this->clone_inner_index();
        if (not clone_value.has_value()) {
            LOG_ERROR_AND_RETURNS(clone_value.error().type, clone_value.error().message);
        }
        return std::make_shared<IndexImpl<T>>(clone_value.value(), this->common_param_);
    }

    tl::expected<IndexPtr, Error>
    ExportModel() const override {
        auto model_value = this->export_model_inner();
        if (not model_value.has_value()) {
            LOG_ERROR_AND_RETURNS(model_value.error().type, model_value.error().message);
        }
        return std::make_shared<IndexImpl<T>>(model_value.value(), this->common_param_);
    }

    tl::expected<void, Error>
    SetImmutable() override {
        SAFE_CALL(this->inner_index_->SetImmutable());
    }

    [[nodiscard]] tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->inner_index_->Serialize());
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(this->inner_index_->Deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(this->inner_index_->Deserialize(reader_set));
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(this->inner_index_->Serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(this->inner_index_->Deserialize(in_stream));
    }

    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override {
        return this->inner_index_->CheckFeature(feature);
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->inner_index_->GetNumElements();
    }

    [[nodiscard]] int64_t
    GetNumberRemoved() const override {
        return this->inner_index_->GetNumberRemoved();
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const override {
        return this->inner_index_->GetMemoryUsage();
    }

    [[nodiscard]] std::string
    GetMemoryUsageDetail() const override {
        return this->inner_index_->GetMemoryUsageDetail();
    }

    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override {
        return this->inner_index_->EstimateMemory(num_elements);
    }

    [[nodiscard]] int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const override {
        return this->inner_index_->GetEstimateBuildMemory(num_elements);
    }

    [[nodiscard]] std::string
    GetStats() const override {
        return this->inner_index_->GetStats();
    }

    [[nodiscard]] bool
    CheckIdExist(int64_t id) const override {
        return this->inner_index_->CheckIdExist(id);
    }

    [[nodiscard]] inline InnerIndexPtr
    GetInnerIndex() const {
        return this->inner_index_;
    }

    [[nodiscard]] inline const IndexCommonParam&
    GetCommonParam() const {
        return this->common_param_;
    }

private:
    tl::expected<InnerIndexPtr, Error>
    clone_inner_index() const {
        SAFE_CALL(return this->inner_index_->Clone(this->common_param_));
    }

    tl::expected<InnerIndexPtr, Error>
    export_model_inner() const {
        SAFE_CALL(return this->inner_index_->ExportModel(this->common_param_));
    }

private:
    InnerIndexPtr inner_index_{nullptr};

    IndexCommonParam common_param_{};
};

}  // namespace vsag
