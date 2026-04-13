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
#include "index_common_param.h"
#include "vsag/index.h"
namespace vsag {

GENERATE_HAS_STATIC_CLASS_FUNCTION(CheckAndMappingExternalParam,
                                   ParamPtr,
                                   std::declval<const JsonType&>(),
                                   std::declval<const IndexCommonParam&>());

/**
 * @brief Template wrapper that bridges external Index API with internal index implementations.
 *
 * This class implements the Bridge pattern, providing a type-safe wrapper around
 * various inner index implementations (T) that inherit from InnerIndexInterface.
 * It handles parameter validation, error propagation, and forwards all operations
 * to the underlying index implementation.
 *
 * @tparam T Inner index type that must inherit from InnerIndexInterface and provide
 *           a static CheckAndMappingExternalParam method for parameter validation.
 */
template <class T>
class IndexImpl : public Index {
    static_assert(std::is_base_of<InnerIndexInterface, T>::value);
    static_assert(has_static_CheckAndMappingExternalParam<T>::value);

public:
    /**
     * @brief Constructs an index from external JSON parameters.
     *
     * Creates the inner index instance by validating and mapping the external
     * parameters to internal format.
     *
     * @param external_param JSON configuration from external API.
     * @param common_param Common parameters shared across index operations.
     */
    IndexImpl(const JsonType& external_param, const IndexCommonParam& common_param)
        : Index(), common_param_(common_param) {
        auto param_ptr = T::CheckAndMappingExternalParam(external_param, common_param);
        this->inner_index_ = std::make_shared<T>(param_ptr, common_param);
        this->inner_index_->InitFeatures();
    }

    /**
     * @brief Constructs an index wrapper from an existing inner index.
     *
     * @param inner_index Shared pointer to the inner index implementation.
     * @param common_param Common parameters shared across index operations.
     */
    IndexImpl(InnerIndexPtr inner_index, const IndexCommonParam& common_param)
        : inner_index_(std::move(inner_index)), common_param_(common_param) {
        this->inner_index_->InitFeatures();
    }

    ~IndexImpl() override {
        this->inner_index_.reset();
    }

/// Macro to return empty dataset if index has no elements.
#define CHECK_AND_RETURN_EMPTY_DATASET          \
    if (GetNumElements() == 0) {                \
        return DatasetImpl::MakeEmptyDataset(); \
    }

/// Macro to return empty dataset if query has no elements.
#define CHECK_QUERY_RETURN_EMPTY_DATASET(query) \
    if ((query)->GetNumElements() == 0) {       \
        return DatasetImpl::MakeEmptyDataset(); \
    }

/// Macro to reject operations on immutable indices.
#define CHECK_IMMUTABLE_INDEX(operation_str)                                       \
    if (this->inner_index_->immutable_) {                                          \
        return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,        \
                                    "immutable index no support " operation_str)); \
    }

/// Macro to return empty result if dataset has no elements.
#define CHECK_NONEMPTY_DATASET(dataset)     \
    if ((dataset)->GetNumElements() == 0) { \
        return std::vector<int64_t>();      \
    }

public:
    /**
     * @brief Adds vectors to the index.
     *
     * @param base Dataset containing vectors to add.
     * @param mode Add mode strategy (default or append).
     * @return Vector of added IDs on success, or error on failure.
     */
    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base, AddMode mode = AddMode::DEFAULT) override {
        CHECK_IMMUTABLE_INDEX("add");
        CHECK_NONEMPTY_DATASET(base);
        SAFE_CALL(return this->inner_index_->Add(base, mode));
    }

    /**
     * @brief Analyzes index characteristics using search request.
     *
     * @param request Search request for analysis.
     * @return Analysis result as JSON string.
     */
    std::string
    AnalyzeIndexBySearch(const SearchRequest& request) override {
        return this->inner_index_->AnalyzeIndexBySearch(request);
    }

    /**
     * @brief Builds the index from a dataset.
     *
     * @param base Dataset containing vectors for building.
     * @return Vector of IDs on success, or error on failure.
     */
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        CHECK_IMMUTABLE_INDEX("build");
        CHECK_NONEMPTY_DATASET(base);
        SAFE_CALL(return this->inner_index_->Build(base));
    }

    /**
     * @brief Tunes index parameters for performance optimization.
     *
     * @param parameters JSON string of tuning parameters.
     * @param disable_future_tuning Whether to prevent future tuning.
     * @return true on success, or error on failure.
     */
    tl::expected<bool, Error>
    Tune(const std::string& parameters, bool disable_future_tuning = false) override {
        CHECK_IMMUTABLE_INDEX("tune");
        SAFE_CALL(return this->inner_index_->Tune(parameters, disable_future_tuning));
    }

    /**
     * @brief Calculates distance from a dataset vector to a specific ID.
     *
     * @param vector Dataset containing the query vector.
     * @param id Target ID to calculate distance to.
     * @param calculate_precise_distance Whether to use precise distance calculation.
     * @return Distance value on success, or error on failure.
     */
    tl::expected<float, Error>
    CalcDistanceById(const DatasetPtr& vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override {
        SAFE_CALL(
            return this->inner_index_->CalcDistanceById(vector, id, calculate_precise_distance));
    }

    /**
     * @brief Calculates distance from a raw vector pointer to a specific ID.
     *
     * @param vector Raw pointer to the query vector data.
     * @param id Target ID to calculate distance to.
     * @param calculate_precise_distance Whether to use precise distance calculation.
     * @return Distance value on success, or error on failure.
     */
    tl::expected<float, Error>
    CalcDistanceById(const float* vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override {
        SAFE_CALL(
            return this->inner_index_->CalcDistanceById(vector, id, calculate_precise_distance));
    }

    /**
     * @brief Calculates distances from a raw query vector to multiple IDs.
     *
     * @param query Raw pointer to the query vector data.
     * @param ids Array of target IDs.
     * @param count Number of IDs in the array.
     * @param calculate_precise_distance Whether to use precise distance calculation.
     * @return Dataset containing distances on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override {
        SAFE_CALL(return this->inner_index_->CalDistanceById(
            query, ids, count, calculate_precise_distance));
    }

    /**
     * @brief Calculates distances from a dataset vector to multiple IDs.
     *
     * @param query Dataset containing the query vector.
     * @param ids Array of target IDs.
     * @param count Number of IDs in the array.
     * @param calculate_precise_distance Whether to use precise distance calculation.
     * @return Dataset containing distances on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    CalDistanceById(const DatasetPtr& query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override {
        SAFE_CALL(return this->inner_index_->CalDistanceById(
            query, ids, count, calculate_precise_distance));
    }

    /**
     * @brief Checks if a feature is supported by this index.
     *
     * @param feature Feature enum to check.
     * @return true if feature is supported, false otherwise.
     */
    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override {
        return this->inner_index_->CheckFeature(feature);
    }

    /**
     * @brief Checks if an ID exists in the index.
     *
     * @param id ID to check.
     * @return true if ID exists, false otherwise.
     */
    [[nodiscard]] bool
    CheckIdExist(int64_t id) const override {
        return this->inner_index_->CheckIdExist(id);
    }

    /**
     * @brief Creates a deep copy of the index.
     *
     * @param allocator Optional allocator for the cloned index.
     * @return Cloned index on success, or error on failure.
     */
    tl::expected<IndexPtr, Error>
    Clone(const std::shared_ptr<Allocator>& allocator = nullptr) const override {
        IndexCommonParam common_param = this->common_param_;
        if (allocator != nullptr) {
            common_param.allocator_ = allocator;
        }
        auto clone_value = this->clone_inner_index(common_param);
        if (not clone_value.has_value()) {
            LOG_ERROR_AND_RETURNS(clone_value.error().type, clone_value.error().message);
        }
        return std::make_shared<IndexImpl<T>>(clone_value.value(), common_param);
    }

    /**
     * @brief Continues building an index from a checkpoint.
     *
     * @param base Dataset containing additional vectors.
     * @param binary_set Checkpoint data from previous build.
     * @return Error on failure.
     */
    tl::expected<Checkpoint, Error>
    ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set) override {
        CHECK_IMMUTABLE_INDEX("continue build");
        SAFE_CALL(return this->inner_index_->ContinueBuild(base, binary_set));
    }

    /**
     * @brief Deserializes the index from a binary set.
     *
     * @param binary_set Binary data to deserialize.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(this->inner_index_->Deserialize(binary_set));
    }

    /**
     * @brief Deserializes the index from a reader set.
     *
     * @param reader_set Reader set providing binary data.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(this->inner_index_->Deserialize(reader_set));
    }

    /**
     * @brief Deserializes the index from an input stream.
     *
     * @param in_stream Input stream containing serialized data.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(this->inner_index_->Deserialize(in_stream));
    }

    /**
     * @brief Estimates memory usage for a given number of elements.
     *
     * @param num_elements Number of elements to estimate for.
     * @return Estimated memory in bytes.
     */
    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override {
        return this->inner_index_->EstimateMemory(num_elements);
    }

    /**
     * @brief Exports all IDs from the index.
     *
     * @return Dataset containing all IDs on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    ExportIDs() const override {
        SAFE_CALL(return this->inner_index_->ExportIDs());
    }

    /**
     * @brief Exports the index model for transfer learning.
     *
     * @return Model index on success, or error on failure.
     */
    tl::expected<IndexPtr, Error>
    ExportModel() const override {
        auto model_value = this->export_model_inner();
        if (not model_value.has_value()) {
            LOG_ERROR_AND_RETURNS(model_value.error().type, model_value.error().message);
        }
        return std::make_shared<IndexImpl<T>>(model_value.value(), this->common_param_);
    }

    /**
     * @brief Retrieves vectors by their IDs.
     *
     * @param ids Array of IDs to retrieve.
     * @param count Number of IDs.
     * @return Dataset containing vectors on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    GetDataByIds(const int64_t* ids, int64_t count) const override {
        SAFE_CALL(return this->inner_index_->GetDataByIds(ids, count));
    };

    /**
     * @brief Retrieves vectors with specific data flags by their IDs.
     *
     * @param ids Array of IDs to retrieve.
     * @param count Number of IDs.
     * @param selected_data_flag Flags indicating which data to retrieve.
     * @return Dataset containing vectors on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    GetDataByIdsWithFlag(const int64_t* ids,
                         int64_t count,
                         uint64_t selected_data_flag) const override {
        SAFE_CALL(return this->inner_index_->GetDataByIdsWithFlag(ids, count, selected_data_flag));
    };

    /**
     * @brief Gets detailed information about the index.
     *
     * @return Vector of IndexDetailInfo on success, or error on failure.
     */
    tl::expected<std::vector<IndexDetailInfo>, Error>
    GetIndexDetailInfos() const override {
        SAFE_CALL(return this->inner_index_->GetIndexDetailInfos());
    }

    /**
     * @brief Gets detailed data by name.
     *
     * @param name Name of the data to retrieve.
     * @param info Output parameter for index detail info.
     * @return Detail data pointer on success, or error on failure.
     */
    tl::expected<DetailDataPtr, Error>
    GetDetailDataByName(const std::string& name, IndexDetailInfo& info) const override {
        SAFE_CALL(return this->inner_index_->GetDetailDataByName(name, info));
    }

    /**
     * @brief Estimates memory required for building the index.
     *
     * @param num_elements Number of elements to estimate for.
     * @return Estimated memory in bytes.
     */
    [[nodiscard]] int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const override {
        return this->inner_index_->GetEstimateBuildMemory(num_elements);
    }

    /**
     * @brief Retrieves extra information for specified IDs.
     *
     * @param ids Array of IDs to get extra info for.
     * @param count Number of IDs.
     * @param extra_infos Output buffer for extra information.
     * @return Error on failure.
     */
    virtual tl::expected<void, Error>
    GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const override {
        SAFE_CALL(this->inner_index_->GetExtraInfoByIds(ids, count, extra_infos));
    };

    /**
     * @brief Gets the type of this index.
     *
     * @return Index type enum value.
     */
    IndexType
    GetIndexType() const override {
        return this->inner_index_->GetIndexType();
    }

    /**
     * @brief Gets current memory usage of the index.
     *
     * @return Memory usage in bytes.
     */
    [[nodiscard]] int64_t
    GetMemoryUsage() const override {
        return this->inner_index_->GetMemoryUsage();
    }

    /**
     * @brief Gets detailed memory usage breakdown.
     *
     * @return JSON string with memory usage details.
     */
    [[nodiscard]] std::string
    GetMemoryUsageDetail() const override {
        return this->inner_index_->GetMemoryUsageDetail();
    }

    /**
     * @brief Gets the minimum and maximum ID in the index.
     *
     * @return Pair of (min_id, max_id) on success, or error on failure.
     */
    tl::expected<std::pair<int64_t, int64_t>, Error>
    GetMinAndMaxId() const override {
        SAFE_CALL(return this->inner_index_->GetMinAndMaxId());
    }

    /**
     * @brief Gets the number of elements in the index.
     *
     * @return Number of elements.
     */
    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->inner_index_->GetNumElements();
    }

    /**
     * @brief Gets the number of removed elements.
     *
     * @return Number of removed elements.
     */
    [[nodiscard]] int64_t
    GetNumberRemoved() const override {
        return this->inner_index_->GetNumberRemoved();
    }

    /**
     * @brief Retrieves raw vectors by their IDs.
     *
     * @param ids Array of IDs to retrieve.
     * @param count Number of IDs.
     * @param specified_allocator Optional allocator for the result.
     * @return Dataset containing raw vectors on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    GetRawVectorByIds(const int64_t* ids,
                      int64_t count,
                      Allocator* specified_allocator) const override {
        if (not CheckFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS)) {
            return tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        "index no support to get raw vector by ids"));
        }
        SAFE_CALL(return this->inner_index_->GetVectorByIds(ids, count, specified_allocator));
    };

    /**
     * @brief Gets statistical information about the index.
     *
     * @return JSON string with statistics.
     */
    [[nodiscard]] std::string
    GetStats() const override {
        return this->inner_index_->GetStats();
    }

    /**
     * @brief Performs k-nearest neighbor search with bitset filter.
     *
     * @param query Dataset containing query vectors.
     * @param k Number of nearest neighbors to return.
     * @param parameters JSON string of search parameters.
     * @param invalid Bitset marking invalid IDs to exclude.
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->KnnSearch(query, k, parameters, invalid));
    }

    /**
     * @brief Performs k-nearest neighbor search with callback filter.
     *
     * @param query Dataset containing query vectors.
     * @param k Number of nearest neighbors to return.
     * @param parameters JSON string of search parameters.
     * @param filter Callback function returning true for valid IDs.
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->KnnSearch(query, k, parameters, filter));
    }

    /**
     * @brief Performs k-nearest neighbor search with FilterPtr.
     *
     * @param query Dataset containing query vectors.
     * @param k Number of nearest neighbors to return.
     * @param parameters JSON string of search parameters.
     * @param filter Filter object for ID filtering.
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->KnnSearch(query, k, parameters, filter));
    }

    /**
     * @brief Performs k-nearest neighbor search with advanced parameters.
     *
     * @param query Dataset containing query vectors.
     * @param k Number of nearest neighbors to return.
     * @param search_param Advanced search parameters including filter and allocator.
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
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

    /**
     * @brief Performs iterative k-nearest neighbor search.
     *
     * @param query Dataset containing query vectors.
     * @param k Number of nearest neighbors to return per iteration.
     * @param parameters JSON string of search parameters.
     * @param filter Filter object for ID filtering.
     * @param iter_ctx Iterator context for maintaining search state.
     * @param is_last_filter Whether this is the last filter call.
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              IteratorContext*& iter_ctx,
              bool is_last_filter) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->KnnSearch(
            query, k, parameters, filter, nullptr, iter_ctx, is_last_filter));
    }

    /**
     * @brief Merges multiple index segments.
     *
     * @param merge_units Vector of merge units to combine.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    Merge(const std::vector<MergeUnit>& merge_units) override {
        CHECK_IMMUTABLE_INDEX("merge");
        SAFE_CALL(this->inner_index_->Merge(merge_units));
    }

    /**
     * @brief Pretrains the index with labeled data.
     *
     * @param base_tag_ids Tag IDs for pretraining.
     * @param k Number of neighbors for pretraining.
     * @param parameters JSON string of pretraining parameters.
     * @return Number of pretraining iterations on success, or error on failure.
     */
    tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids,
             uint32_t k,
             const std::string& parameters) override {
        CHECK_IMMUTABLE_INDEX("pretrain");
        SAFE_CALL(return this->inner_index_->Pretrain(base_tag_ids, k, parameters));
    }

    /**
     * @brief Provides feedback for index optimization.
     *
     * @param query Dataset containing query vectors.
     * @param k Number of neighbors considered.
     * @param parameters JSON string of feedback parameters.
     * @param global_optimum_tag_id Known optimal tag ID for feedback.
     * @return Number of feedback iterations on success, or error on failure.
     */
    tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        CHECK_IMMUTABLE_INDEX("feedback");
        if (query->GetNumElements() == 0) {
            return 0;
        }
        SAFE_CALL(return this->inner_index_->Feedback(query, k, parameters, global_optimum_tag_id));
    }

    /**
     * @brief Performs range search within a radius.
     *
     * @param query Dataset containing query vectors.
     * @param radius Search radius threshold.
     * @param parameters JSON string of search parameters.
     * @param limited_size Maximum number of results (-1 for unlimited).
     * @return Dataset containing results on success, or error on failure.
     */
    [[nodiscard]] tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->RangeSearch(query, radius, parameters, limited_size));
    }

    /**
     * @brief Performs range search with bitset filter.
     *
     * @param query Dataset containing query vectors.
     * @param radius Search radius threshold.
     * @param parameters JSON string of search parameters.
     * @param invalid Bitset marking invalid IDs to exclude.
     * @param limited_size Maximum number of results (-1 for unlimited).
     * @return Dataset containing results on success, or error on failure.
     */
    [[nodiscard]] tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->RangeSearch(
            query, radius, parameters, invalid, limited_size));
    }

    /**
     * @brief Performs range search with callback filter.
     *
     * @param query Dataset containing query vectors.
     * @param radius Search radius threshold.
     * @param parameters JSON string of search parameters.
     * @param filter Callback function returning true for valid IDs.
     * @param limited_size Maximum number of results (-1 for unlimited).
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->RangeSearch(
            query, radius, parameters, filter, limited_size));
    }

    /**
     * @brief Performs range search with FilterPtr.
     *
     * @param query Dataset containing query vectors.
     * @param radius Search radius threshold.
     * @param parameters JSON string of search parameters.
     * @param filter Filter object for ID filtering.
     * @param limited_size Maximum number of results (-1 for unlimited).
     * @return Dataset containing results on success, or error on failure.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override {
        CHECK_QUERY_RETURN_EMPTY_DATASET(query);
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->RangeSearch(
            query, radius, parameters, filter, limited_size));
    }

    /**
     * @brief Removes vectors by their IDs.
     *
     * @param ids IDs to remove.
     * @param mode Removal mode (mark or erase).
     * @return Number of removed vectors on success, or error on failure.
     */
    tl::expected<uint32_t, Error>
    Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE) override {
        CHECK_IMMUTABLE_INDEX("remove");
        SAFE_CALL(return this->inner_index_->Remove(ids, mode));
    }

    /**
     * @brief Shrinks index and repairs deleted entries.
     *
     * @param timeout_ms Maximum time in milliseconds for the operation.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    ShrinkAndRepair(double timeout_ms = std::numeric_limits<double>::max()) override {
        CHECK_IMMUTABLE_INDEX("shrink and repair");
        SAFE_CALL(this->inner_index_->ShrinkAndRepair(timeout_ms));
    }

    /**
     * @brief Serializes the index to a binary set.
     *
     * @return Binary set on success, or error on failure.
     */
    [[nodiscard]] tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->inner_index_->Serialize());
    }

    /**
     * @brief Serializes the index using a write callback.
     *
     * @param write_func Function to write binary data.
     * @return Error on failure.
     */
    [[nodiscard]] tl::expected<void, Error>
    Serialize(WriteFuncType write_func) const override {
        SAFE_CALL(this->inner_index_->Serialize(write_func));
    }

    /**
     * @brief Serializes the index to an output stream.
     *
     * @param out_stream Output stream for serialization.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(this->inner_index_->Serialize(out_stream));
    }

    /**
     * @brief Performs search using a request object.
     *
     * @param request Search request containing query and parameters.
     * @return Dataset containing results on success, or error on failure.
     */
    [[nodiscard]] tl::expected<DatasetPtr, Error>
    SearchWithRequest(const SearchRequest& request) const override {
        CHECK_AND_RETURN_EMPTY_DATASET;
        SAFE_CALL(return this->inner_index_->SearchWithRequest(request));
    }

    /**
     * @brief Marks the index as immutable (read-only).
     *
     * @return Error on failure.
     */
    tl::expected<void, Error>
    SetImmutable() override {
        SAFE_CALL(this->inner_index_->SetImmutable());
    }

    /**
     * @brief Trains the index with sample data.
     *
     * @param data Dataset containing training vectors.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    Train(const DatasetPtr& data) override {
        CHECK_IMMUTABLE_INDEX("train");
        if (data->GetNumElements() != 0) {
            SAFE_CALL(this->inner_index_->Train(data));
        }
        return {};
    }

    /**
     * @brief Updates attributes for a specific ID.
     *
     * @param id ID to update.
     * @param new_attrs New attribute set.
     * @return Error on failure.
     */
    virtual tl::expected<void, Error>
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override {
        CHECK_IMMUTABLE_INDEX("update attribute");
        SAFE_CALL(this->inner_index_->UpdateAttribute(id, new_attrs));
    }

    /**
     * @brief Updates attributes with original value verification.
     *
     * @param id ID to update.
     * @param new_attrs New attribute set.
     * @param origin_attrs Expected original attributes for verification.
     * @return Error on failure.
     */
    tl::expected<void, Error>
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override {
        CHECK_IMMUTABLE_INDEX("update attribute with origin attributes");
        SAFE_CALL(this->inner_index_->UpdateAttribute(id, new_attrs, origin_attrs));
    }

    /**
     * @brief Updates extra information for multiple IDs.
     *
     * @param new_base Dataset containing new extra information.
     * @return true if update succeeded, false if no elements, or error on failure.
     */
    virtual tl::expected<bool, Error>
    UpdateExtraInfo(const DatasetPtr& new_base) override {
        CHECK_IMMUTABLE_INDEX("update extra info");
        if (new_base->GetNumElements() == 0) {
            return false;
        }
        SAFE_CALL(return this->inner_index_->UpdateExtraInfo(new_base));
    }

    /**
     * @brief Updates the ID of a vector.
     *
     * @param old_id Current ID.
     * @param new_id New ID to assign.
     * @return true on success, or error on failure.
     */
    tl::expected<bool, Error>
    UpdateId(int64_t old_id, int64_t new_id) override {
        CHECK_IMMUTABLE_INDEX("update id");
        SAFE_CALL(return this->inner_index_->UpdateId(old_id, new_id));
    }

    /**
     * @brief Updates a vector by its ID.
     *
     * @param id ID of the vector to update.
     * @param new_base Dataset containing the new vector.
     * @param force_update Whether to force update without checks.
     * @return true on success, or error on failure.
     */
    tl::expected<bool, Error>
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override {
        CHECK_IMMUTABLE_INDEX("update vector");
        if (new_base->GetNumElements() == 0) {
            return false;
        }
        SAFE_CALL(return this->inner_index_->UpdateVector(id, new_base, force_update));
    }

public:
    /**
     * @brief Gets the inner index implementation.
     *
     * @return Shared pointer to the inner index.
     */
    [[nodiscard]] inline InnerIndexPtr
    GetInnerIndex() const {
        return this->inner_index_;
    }

    /**
     * @brief Gets the common parameters for this index.
     *
     * @return Const reference to IndexCommonParam.
     */
    [[nodiscard]] inline const IndexCommonParam&
    GetCommonParam() const {
        return this->common_param_;
    }

private:
    /**
     * @brief Clones the inner index with new common parameters.
     *
     * @param common_param Common parameters for the cloned index.
     * @return Cloned inner index on success, or error on failure.
     */
    tl::expected<InnerIndexPtr, Error>
    clone_inner_index(const IndexCommonParam& common_param) const {
        SAFE_CALL(return this->inner_index_->Clone(common_param));
    }

    /**
     * @brief Exports the model from the inner index.
     *
     * @return Exported model on success, or error on failure.
     */
    tl::expected<InnerIndexPtr, Error>
    export_model_inner() const {
        SAFE_CALL(return this->inner_index_->ExportModel(this->common_param_));
    }

private:
    InnerIndexPtr inner_index_{nullptr};  ///< Internal index implementation
    IndexCommonParam common_param_{};     ///< Shared parameters for index operations
};

}  // namespace vsag