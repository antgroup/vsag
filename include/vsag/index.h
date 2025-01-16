
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>

#include "bitset.h"
#include "features.h"
#include "vsag/binaryset.h"
#include "vsag/bitset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index_features.h"
#include "vsag/readerset.h"

namespace vsag {

class Index;
using IndexPtr = std::shared_ptr<Index>;

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
      * @brief Return the memory occupied by the index
      *
      * @return number of bytes occupied by the index.
      */
    [[nodiscard]] virtual int64_t
    GetMemoryUsage() const = 0;

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
                     const std::string& params);

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
