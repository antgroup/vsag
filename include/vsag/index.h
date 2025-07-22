
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
#include <limits>
#include <stdexcept>

#include "vsag/binaryset.h"
#include "vsag/bitset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/filter.h"
#include "vsag/index_features.h"
#include "vsag/iterator_context.h"
#include "vsag/readerset.h"
#include "vsag/search_param.h"
#include "vsag/search_request.h"

namespace vsag {

class Index;
using IndexPtr = std::shared_ptr<Index>;
using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

struct MergeUnit {
    IndexPtr index = nullptr;
    IdMapFunction id_map_func = nullptr;
};

enum class IndexType {
    HNSW,
    DISKANN,
    HGRAPH,
    IVF,
    PYRAMID,
    BRUTEFORCE,
    SPARSE,
};

class Index {
public:
    // [basic methods]

    /**
      * @brief Building index with all vectors
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return IDs that failed to insert into the index
      */
    virtual tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) = 0;

    /**
     * @brief Get Index Type
     * @return IndexType
     */
    virtual IndexType
    GetIndexType() {
        throw std::runtime_error("Index not support GetIndexType");
    }

    /**
      * @brief Training index with given vectors
      *
      * @param datas should contains dim, num_elements, ids and vectors
      * @return result indicates whether the train operation is successful.
      */
    virtual tl::expected<void, Error>
    Train(const DatasetPtr& data) {
        throw std::runtime_error("Index not support Train");
    }

    struct Checkpoint {
        BinarySet data;
        bool finish = false;
    };

    /**
      * @brief Provide dynamism for indexes that do not support insertions
      *
      * @param base should contains dim, num_elements, ids and vectors
      * @param binary_set contains intermediate data from the last checkpoint
      * @return intermediate data of the current checkpoint
      */
    virtual tl::expected<Checkpoint, Error>
    ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set) {
        throw std::runtime_error("Index not support partial build");
    }

    /**
      * @brief Adding vectors into a built index, only HNSW supported now, called on other index will cause exception
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return IDs that failed to insert into the index
      */
    virtual tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) {
        throw std::runtime_error("Index not support adding vectors");
    }

    /**
      * @brief Remove the vector corresponding to the given ID from the index
      *
      * @param id of the vector that need to be removed from the index
      * @return result indicates whether the remove operation is successful.
      */
    virtual tl::expected<bool, Error>
    Remove(int64_t id) {
        throw std::runtime_error("Index not support delete vector");
    }

    /**
     * @brief Update the id of a base point from the index
     *
     * @param old_id indicates the old id of a base point in index
     * @param new_id is the updated new id of the base point
     * @return result indicates whether the update operation is successful.
     */
    virtual tl::expected<bool, Error>
    UpdateId(int64_t old_id, int64_t new_id) {
        throw std::runtime_error("Index not support update id");
    }

    /**
     * @brief Update the vector of a base point from the index
     *
     * @param id indicates the old id of a base point in index
     * @param new_base is the updated new vector of the base point
     * @param force_update is false means that a check of the connectivity of the graph updated by this operation is performed
     * @return result indicates whether the update operation is successful.
     */
    virtual tl::expected<bool, Error>
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) {
        throw std::runtime_error("Index not support update vector");
    }

    /**
     * @brief Update the attribute of a base point from the index
     *
     * @param id indicates the id of a base point in index
     * @param origin_attrs is the origin attributes of the base point
     * @param new_attrs is the new attributes of the base point
     * @return result indicates whether the update operation is successful.
     */
    virtual tl::expected<void, Error>
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
        throw std::runtime_error("Index not support update attribute");
    }

    /**
      * @brief Performing single KNN search on index
      * 
      * @param query should contains dim, num_elements and vectors
      * @param k the result size of every query
      * @param invalid represents whether an element is filtered out by pre-filter
      * @return result contains 
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)
      */
    [[nodiscard]] virtual tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const = 0;

    /**
      * @brief Performing single KNN search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param k the result size of every query
      * @param filter represents whether an element is filtered out by pre-filter
      * @return result contains
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)
      */
    virtual tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const = 0;

    /**
      * @brief Performing single KNN search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param k the result size of every query
      * @param filter represents whether an element is filtered out by pre-filter
      * @return result contains
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)
      */
    virtual tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    /**
      * @brief Performing search with request on index
      * 
      * @param request @see SearchRequest
      * @return result contains 
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)               
      */
    virtual tl::expected<DatasetPtr, Error>
    SearchWithRequest(const SearchRequest& request) const {
        throw std::runtime_error("Index doesn't support Search With Request");
    }

    /**
      * @brief Performing single KNN search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param k the result size of every query
      * @param filter represents whether an element is filtered out by pre-filter
      * @param iter_ctx iterative filter context
      * @param is_last_search last iterator filter search flag
      * @return result contains
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)
      */
    virtual tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              IteratorContext*& iter_ctx,
              bool is_last_search) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    /**
      * @brief Performing single KNN search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param k the result size of every query
      * @param search_param search param contains filter, iter_ctx and allocator
      * @return result contains
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)
      */
    virtual tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    /**
      * @brief Performing single range search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param radius of search, determines which results will be returned
      * @param limited_size of search result size.
      *                - limited_size <= 0 : no limit
      *                - limited_size == 0 : error
      *                - limited_size >= 1 : limit result size to limited_size
      * @return result contains
      *                - num_elements: 1
      *                - dim: the size of results
      *                - ids, distances: length is dim
      */
    [[nodiscard]] virtual tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const = 0;

    /**
      * @brief Performing single range search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param radius of search, determines which results will be returned
      * @param limited_size of search result size.
      *                - limited_size <= 0 : no limit
      *                - limited_size == 0 : error
      *                - limited_size >= 1 : limit result size to limited_size
      * @param invalid represents whether an element is filtered out by pre-filter
      * @return result contains
      *                - num_elements: 1
      *                - dim: the size of results
      *                - ids, distances: length is dim
      */
    [[nodiscard]] virtual tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const = 0;

    /**
      * @brief Performing single range search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param radius of search, determines which results will be returned
      * @param limited_size of search result size.
      *                - limited_size <= 0 : no limit
      *                - limited_size == 0 : error
      *                - limited_size >= 1 : limit result size to limited_size
      * @param filter represents whether an element is filtered out by pre-filter
      * @return result contains
      *                - num_elements: 1
      *                - dim: the size of results
      *                - ids, distances: length is dim
      */
    virtual tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const = 0;

    /**
      * @brief Performing single range search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param radius of search, determines which results will be returned
      * @param limited_size of search result size.
      *                - limited_size <= 0 : no limit
      *                - limited_size == 0 : error
      *                - limited_size >= 1 : limit result size to limited_size
      * @param filter represents whether an element is filtered out by pre-filter
      * @return result contains
      *                - num_elements: 1
      *                - dim: the size of results
      *                - ids, distances: length is dim
      */
    virtual tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const {
        throw std::runtime_error("Index doesn't support new filter");
    }

    /**
     * @brief Pretraining the conjugate graph involves searching with generated queries and providing feedback.
     *
     * @param base_tag_ids is the label of choosen base vectors that need to be enhanced
     * @param k is the number of edges inserted into conjugate graph
     * @return result is the number of successful insertions into conjugate graph
     */
    virtual tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters) {
        throw std::runtime_error("Index doesn't support pretrain");
    };

    /**
     * @brief Performing feedback on conjugate graph
     *
     * @param query should contains dim, num_elements and vectors
     * @param k is the number of edges inserted into conjugate graph
     * @param global_optimum_tag_id is the label of exact nearest neighbor
     * @return result is the number of successful insertions into conjugate graph
     */
    virtual tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) {
        throw std::runtime_error("Index doesn't support feedback");
    };

    /**
     * @brief Calculate the distance between the query and the vector of the given ID.
     *
     * @param vector is the embedding of query
     * @param id is the unique identifier of the vector to be calculated in the index.
     * @return result is the distance between the query and the vector of the given ID.
     */
    virtual tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const {
        throw std::runtime_error("Index doesn't support get distance by id");
    };

    /**
     * @brief Calculate the distance between the query and the vector of the given ID for batch.
     *
     * @param query is the embedding of query
     * @param ids is the unique identifier of the vector to be calculated in the index.
     * @param count is the count of ids
     * @return result is valid distance of input ids. '-1' indicates an invalid distance.
     */
    virtual tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const {
        throw std::runtime_error("Index doesn't support get distance by id");
    };

    /**
     * @brief Calculate the maximum and minimum labels.
     *
     * @param min_id The minimum id returned
     * @param max_id The maximum id returned
     */
    virtual tl::expected<std::pair<int64_t, int64_t>, Error>
    GetMinAndMaxId() const {
        throw std::runtime_error("Index doesn't support get Min and Max id");
    }

    /**
     * @brief Retrieve additional data associated with vectors identified by given IDs.
     *
     * This method fetches non-vector metadata stored alongside the vectors in the index
     * (e.g., timestamps, labels, or application-specific fields).
     * The format and content of the extra data depend on how they were stored during index creation.
     *
     * @param ids Array of vector IDs for which extra information is requested.
     * @param count Number of IDs in the 'ids' array.
     * @param extra_infos A char* pointer to the retrieved extra data if successful
     * (format is implementation-specific). Returns an error object
     * if any retrieval failure occurs (e.g., invalid ID, out of memory).
     * @throws std::runtime_error If the index implementation does not support this operation
     * (default behavior for base class).
     */
    virtual tl::expected<void, Error>
    GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const {
        throw std::runtime_error("Index doesn't support GetExtraInfoByIds");
    };

    /**
     * @brief Retrieve raw vector data associated with given IDs.
     *
     * This method fetches the actual vector data stored in the index for specified IDs.
     * The returned dataset typically contains the original vector values in a format
     * compatible with the index implementation (e.g., float arrays or binary embeddings).
     * This is useful for operations requiring direct access to vector contents,
     * such as retraining models or data migration.
     *
     * @param ids Array of vector IDs for which raw data is requested.
     * @param count Number of IDs in the 'ids' array.
     * @return tl::expected<DatasetPtr, Error>
     *         - On success: A DatasetPtr containing the raw vector data
     *           (format depends on implementation, but typically includes vector arrays).
     *         - On failure: An error object (e.g., invalid ID, out of memory).
     * @throws std::runtime_error If the index implementation does not support this operation
     *            (default behavior for base class).
     */
    virtual tl::expected<DatasetPtr, Error>
    GetRawVectorByIds(const int64_t* ids, int64_t count) const {
        throw std::runtime_error("Index doesn't support GetRawVectorByIds");
    };

    /**
     * @brief Checks if the specified feature is supported by the index.
     *
     * This method checks whether the given `feature` is supported by the index.
     * @see IndexFeature
     *
     * @param feature The feature to check for support.
     * @return bool Returns true if the feature is supported, false otherwise.
     */
    [[nodiscard]] virtual bool
    CheckFeature(IndexFeature feature) const {
        throw std::runtime_error("Index doesn't support check feature");
    }

    /**
     * @brief Merges multiple graph indexes with ID mapping into the current index
     *
     * Processes MergeUnit entries to incorporate sub-indexes into this index. For each element:
     * - id_map_func determines which IDs from the sub-index are retained
     * - Specifies the ID remapping into the destination index space
     *
     * @param merge_units Vector containing:
     *   - index: Source sub-index to merge from
     *   - id_map_func: Filter+remap function that for each source ID (int64_t) returns:
     *     * bool: true if the ID should be included in the merge
     *     * int64_t: Target ID in destination index (only valid when bool is true)
     */
    virtual tl::expected<void, Error>
    Merge(const std::vector<MergeUnit>& merge_units) {
        throw std::runtime_error("Index doesn't support merge");
    }

    /**
     * @brief Clones the index.
     *
     * Creates a new index that is a deep copy of the current index.
     *
     * @return IndexPtr A pointer to the cloned index.
     */
    virtual tl::expected<IndexPtr, Error>
    Clone() const {
        throw std::runtime_error("Index doesn't support Clone");
    }

    /**
     * @brief Export the index's model as an empty index.
     * 
     * @return IndexPtr A pointer to the exported model index.
     * @throws std::runtime_error If the index does not support exporting the model.
     */
    virtual tl::expected<IndexPtr, Error>
    ExportModel() const {
        throw std::runtime_error("Index doesn't support ExportModel");
    }

    /**
     * @brief set the index to immutable state.
     * After setting this state, no further modifications are supported, such as no additions or deletions 
     *
     * @throws std::runtime_error If the index does not support to set immutable
     */
    virtual tl::expected<void, Error>
    SetImmutable() {
        throw std::runtime_error("Index doesn't support SetImmutable");
    }

public:
    // [serialize/deserialize with binaryset]

    /**
      * @brief Serialize index to a set of byte array
      *
      * @return binaryset contains all parts of the index
      */
    [[nodiscard]] virtual tl::expected<BinarySet, Error>
    Serialize() const = 0;

    /**
      * @brief Deserialize index from a set of byte array. Causing exception if this index is not empty
      *
      * @param binaryset contains all parts of the index
      */
    virtual tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) = 0;

    /**
      * @brief Deserialize index from a set of reader array. Causing exception if this index is not empty
      *
      * @param reader contains all parts of the index
      */
    virtual tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) = 0;

public:
    // [serialize/deserialize with file stream]

    /**
      * @brief Serialize index to a file stream
      *
      * @param out_stream is a already opened file stream for outputing the serialized index
      */
    virtual tl::expected<void, Error>
    Serialize(std::ostream& out_stream) {
        throw std::runtime_error("Index not support serialize to a file stream");
    }

    /**
      * @brief Deserialize index from a file stream
      * 
      * @param in_stream is a already opened file stream contains serialized index
      * @param length is the length of serialized index(may differ from the actual file size
      *   if there is additional content in the file)
      */
    virtual tl::expected<void, Error>
    Deserialize(std::istream& in_stream) {
        throw std::runtime_error("Index not support deserialize from a file stream");
    }

public:
    // [statstics methods]

    /**
      * @brief Return the number of elements in the index
      *
      * @return number of elements in the index.
      */
    [[nodiscard]] virtual int64_t
    GetNumElements() const = 0;

    /**
      * @brief Return the number of removed elements in the index
      *
      * @return number of removed elements in the index.
      */
    [[nodiscard]] virtual int64_t
    GetNumberRemoved() const {
        throw std::runtime_error("Index not support GetNumberRemoved");
    }

    /**
      * @brief Return the memory occupied by the index
      *
      * @return number of bytes occupied by the index.
      */
    [[nodiscard]] virtual int64_t
    GetMemoryUsage() const = 0;

    /**
  * @brief Return the memory usage of every component in the index
  *
  * @return a json object that contains the memory usage of every component in the index
  */
    // TODO(deming): implement func for every types of index
    // [[nodiscard]] virtual JsonType
    // GetMemoryUsageDetail() const = 0;
    [[nodiscard]] virtual std::string
    GetMemoryUsageDetail() const {
        throw std::runtime_error("Index not support GetMemoryUsageDetail");
    }

    /**
      * @brief estimate the memory used by the index with given element counts
      *
      * @param num_elements
      * @return number of bytes estimate used.
      */
    [[nodiscard]] virtual uint64_t
    EstimateMemory(uint64_t num_elements) const {
        throw std::runtime_error("Index not support estimate the memory by element counts");
    }

    /**
      * @brief Return the estimated memory required during building
      *
      * @param num_elements denotes the amount of data used to build the index.
      * @return estimated memory required during building.
      */
    [[nodiscard]] virtual int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const {
        throw std::runtime_error("Index not support estimate the memory while building");
    }

    /**
      * @brief Get the statstics from index
      *
      * @return a json string contains runtime statstics of the index.
      */
    [[nodiscard]] virtual std::string
    GetStats() const {
        throw std::runtime_error("Index not support range search");
    }

    /**
      * @brief Check if a specific ID exists in the index.
      *
      * @param id The ID to check for existence in the index.
      * @return True if the ID exists, otherwise false.
      * @throws std::runtime_error if the index does not support checking ID existence.
      */
    [[nodiscard]] virtual bool
    CheckIdExist(int64_t id) const {
        throw std::runtime_error("Index not support check id exist");
    }

public:
    virtual ~Index() = default;
};

/**
  * @brief check if the build parameter is valid
  *
  * @return true if the parameter is valid, otherwise error with detail message.
  */
tl::expected<bool, Error>
check_diskann_hnsw_build_parameters(const std::string& json_string);

/**
  * @brief check if the build parameter is valid
  *
  * @return true if the parameter is valid, otherwise error with detail message.
  */
tl::expected<bool, Error>
check_diskann_hnsw_search_parameters(const std::string& json_string);

/**
  * @brief estimate search time for index
  *
  * @return the estimated search time in milliseconds.
  */
tl::expected<float, Error>
estimate_search_time(const std::string& index_name,
                     int64_t data_num,
                     int64_t data_dim,
                     const std::string& parameters);

/**
  * [experimental]
  * @brief generate build index parameters from data size and dim
  *
  * @return the build parameter string
  */
tl::expected<std::string, Error>
generate_build_parameters(std::string metric_type,
                          int64_t num_elements,
                          int64_t dim,
                          bool use_conjugate_graph = false);

}  // namespace vsag
