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

// to suppress deprecated warning below(no better way found that works with clang-tidy-15):
// - clang-diagnostic-deprecated-builtins
#if defined(__clang__) && (__clang_major__ >= 15)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#endif

#include <abstract_index.h>
#include <disk_utils.h>
#include <index.h>
#include <omp.h>
#include <pq_flash_index.h>

#if defined(__clang__) && (__clang_major__ >= 15)
#pragma clang diagnostic pop
#endif

#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <queue>
#include <shared_mutex>
#include <string>

#include "common.h"
#include "diskann_logger.h"
#include "diskann_zparameters.h"
#include "index_feature_list.h"
#include "typing.h"
#include "utils/window_result_queue.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/thread_pool.h"

namespace vsag {

/**
 * @brief Enumeration of index loading status.
 *
 * Tracks the current state of the DiskANN index during its lifecycle,
 * from initialization through building to fully operational states.
 */
enum IndexStatus { EMPTY = 0, MEMORY = 1, HYBRID = 2, BUILDING = 3 };

/**
 * @brief Enumeration of incremental build stages.
 *
 * Defines the sequential stages for incremental index construction,
 * enabling checkpoint-based resume capability during the build process.
 */
enum BuildStatus { BEGIN = 0, GRAPH = 1, EDGE_PRUNE = 2, PQ = 3, DISK_LAYOUT = 4, FINISH = 5 };

/**
 * @brief Disk-based Approximate Nearest Neighbor index implementation.
 *
 * This class implements a disk-resident vector index using the DiskANN algorithm,
 * which combines in-memory graph search with disk-based vector storage for
 * efficient similarity search on large datasets that exceed available memory.
 *
 * Key design decisions:
 * - Supports both memory-only and hybrid (memory+disk) operation modes
 * - Uses PQ (Product Quantization) for compressed vector representation
 * - Implements checkpoint-based incremental building for large datasets
 * - Thread-safe through read-write mutex protection
 */
class DiskANN : public Index {
public:
    using rs = std::pair<float, uint64_t>;

    /// Tuple representing a read request: (offset, length, destination buffer)
    using read_request = std::tuple<uint64_t, uint64_t, void*>;

    /**
     * @brief Constructs a DiskANN index with the specified parameters.
     *
     * @param diskann_params Configuration parameters for the DiskANN index.
     * @param index_common_param Common parameters shared across index implementations.
     */
    DiskANN(DiskannParameters& diskann_params, const IndexCommonParam& index_common_param);

    ~DiskANN() override = default;

    /**
     * @brief Builds the index from the provided dataset.
     *
     * @param base The dataset containing vectors to index.
     * @return A vector of IDs for the indexed vectors on success, or an Error on failure.
     */
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    /**
     * @brief Calculates distances between a query and multiple vectors by their IDs.
     *
     * @param query Pointer to the query vector data.
     * @param ids Array of vector IDs to calculate distances for.
     * @param count Number of IDs in the array.
     * @param calculate_precise_distance Whether to calculate precise distances (vs. approximate).
     * @return A dataset containing the calculated distances on success, or an Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override {
        SAFE_CALL(return this->cal_distance_by_id(query, ids, count, calculate_precise_distance));
    }

    /**
     * @brief Calculates the distance between a query vector and a single vector by ID.
     *
     * @param vector Pointer to the query vector data.
     * @param id The ID of the vector to calculate distance to.
     * @param calculate_precise_distance Whether to calculate precise distance (vs. approximate).
     * @return The calculated distance on success, or an Error on failure.
     */
    tl::expected<float, Error>
    CalcDistanceById(const float* vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override {
        SAFE_CALL(return this->calc_distance_by_id(vector, id, calculate_precise_distance));
    }

    /**
     * @brief Returns the index type identifier.
     *
     * @return Always returns IndexType::DISKANN.
     */
    IndexType
    GetIndexType() const override {
        return IndexType::DISKANN;
    }

    /**
     * @brief Continues building the index from a checkpoint.
     *
     * @param base The dataset containing vectors to index.
     * @param binary_set The checkpoint data from a previous build stage.
     * @return A Checkpoint indicating the next build stage on success, or an Error on failure.
     */
    tl::expected<Checkpoint, Error>
    ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set) override {
        SAFE_CALL(return this->continue_build(base, binary_set));
    }

    /**
     * @brief Performs k-nearest neighbor search with a bitset filter.
     *
     * @param query The query dataset containing the search vector.
     * @param k The number of nearest neighbors to return.
     * @param parameters JSON string containing search parameters.
     * @param invalid Bitset indicating which IDs should be excluded from results.
     * @return A dataset containing the k nearest neighbors on success, or an Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, invalid));
    }

    /**
     * @brief Performs k-nearest neighbor search with a callback filter.
     *
     * @param query The query dataset containing the search vector.
     * @param k The number of nearest neighbors to return.
     * @param parameters JSON string containing search parameters.
     * @param filter A callback function that returns true if an ID should be excluded.
     * @return A dataset containing the k nearest neighbors on success, or an Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter));
    }

    /**
     * @brief Performs range search within a specified radius.
     *
     * @param query The query dataset containing the search vector.
     * @param radius The maximum distance for neighbors to be included in results.
     * @param parameters JSON string containing search parameters.
     * @param limited_size Maximum number of results to return (-1 for unlimited).
     * @return A dataset containing neighbors within the radius on success, or an Error.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(
            query, radius, parameters, (BitsetPtr) nullptr, limited_size));
    }

    /**
     * @brief Performs range search with a callback filter.
     *
     * @param query The query dataset containing the search vector.
     * @param radius The maximum distance for neighbors to be included in results.
     * @param parameters JSON string containing search parameters.
     * @param filter A callback function that returns true if an ID should be excluded.
     * @param limited_size Maximum number of results to return (-1 for unlimited).
     * @return A dataset containing neighbors within the radius on success, or an Error.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, filter, limited_size));
    }

    /**
     * @brief Performs range search with a bitset filter.
     *
     * @param query The query dataset containing the search vector.
     * @param radius The maximum distance for neighbors to be included in results.
     * @param parameters JSON string containing search parameters.
     * @param invalid Bitset indicating which IDs should be excluded from results.
     * @param limited_size Maximum number of results to return (-1 for unlimited).
     * @return A dataset containing neighbors within the radius on success, or an Error.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, invalid, limited_size));
    }

public:
    /**
     * @brief Serializes the index to a BinarySet.
     *
     * @return A BinarySet containing the serialized index on success, or an Error on failure.
     */
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    /**
     * @brief Deserializes the index from a BinarySet.
     *
     * @param binary_set The BinarySet containing serialized index data.
     * @return void on success, or an Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    /**
     * @brief Deserializes the index from a ReaderSet.
     *
     * @param reader_set The ReaderSet providing access to serialized index data.
     * @return void on success, or an Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

public:
    /**
     * @brief Serializes the index to an output stream.
     *
     * @param out_stream The output stream to write the serialized index to.
     * @return void on success, or an Error on failure.
     */
    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    /**
     * @brief Deserializes the index from an input stream.
     *
     * @param in_stream The input stream to read the serialized index from.
     * @return void on success, or an Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

public:
    /**
     * @brief Returns the number of elements in the index.
     *
     * @return The number of indexed vectors, or 0 if the index is empty.
     */
    int64_t
    GetNumElements() const override {
        if (status_ == EMPTY)
            return 0;
        return index_->get_data_num();
    }

    /**
     * @brief Returns the estimated memory usage of the index.
     *
     * @return Memory usage in bytes, varies based on index status (MEMORY or HYBRID).
     */
    int64_t
    GetMemoryUsage() const override {
        if (status_ == MEMORY) {
            return index_->get_memory_usage() + disk_pq_compressed_vectors_.str().size() +
                   pq_pivots_stream_.str().size() + disk_layout_stream_.str().size() +
                   tag_stream_.str().size() + graph_stream_.str().size();
        } else if (status_ == HYBRID) {
            return index_->get_memory_usage();
        }
        return 0;
    }

    /**
     * @brief Estimates the memory required to build an index with the given number of elements.
     *
     * @param num_elements The number of elements to estimate memory for.
     * @return Estimated memory usage in bytes.
     */
    int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const override;

    /**
     * @brief Returns statistical information about the index.
     *
     * @return A JSON string containing index statistics.
     */
    std::string
    GetStats() const override;

    /**
     * @brief Checks if the index supports a specific feature.
     *
     * @param feature The feature to check for support.
     * @return true if the feature is supported, false otherwise.
     */
    bool
    CheckFeature(IndexFeature feature) const override;

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& base);

    tl::expected<Checkpoint, Error>
    continue_build(const DatasetPtr& base, const BinarySet& binary_set);

    DatasetPtr
    cal_distance_by_id(const float* query,
                       const int64_t* ids,
                       int64_t count,
                       bool calculate_precise_distance = true) const;

    float
    calc_distance_by_id(const float* vector,
                        int64_t id,
                        bool calculate_precise_distance = true) const;

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const BitsetPtr& invalid) const;

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const std::function<bool(int64_t)>& filter,
                 int64_t limited_size) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const BitsetPtr& invalid,
                 int64_t limited_size) const;

    tl::expected<BinarySet, Error>
    serialize() const;

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(const ReaderSet& reader_set);

    tl::expected<void, Error>
    serialize(std::ostream& out_stream);

    tl::expected<void, Error>
    deserialize(std::istream& in_stream);

    tl::expected<void, Error>
    build_partial_graph(const DatasetPtr& base,
                        const BinarySet& binary_set,
                        BinarySet& after_binary_set,
                        int round);

    tl::expected<void, Error>
    load_disk_index(const BinarySet& binary_set);

    void
    init_feature_list();

private:
    std::shared_ptr<LocalFileReader> reader_;  ///< File reader for disk-based data access
    std::shared_ptr<diskann::PQFlashIndex<float, int64_t>> index_;  ///< Main flash index instance
    std::shared_ptr<diskann::Index<float, int64_t, int64_t>>
        build_index_;  ///< Index used during build

    std::stringstream pq_pivots_stream_;            ///< Stream for PQ pivot data
    std::stringstream disk_pq_compressed_vectors_;  ///< Stream for compressed PQ vectors
    std::stringstream disk_layout_stream_;          ///< Stream for disk layout metadata
    std::stringstream tag_stream_;                  ///< Stream for tag (ID) mapping data
    std::stringstream graph_stream_;                ///< Stream for graph structure data

    const IndexCommonParam index_common_param_;  ///< Common parameters for index operations

    IndexFeatureListPtr feature_list_{nullptr};  ///< List of supported features

    std::function<void(const std::vector<read_request>&, bool, CallBack)>
        batch_read_;                              ///< Batch read callback
    diskann::Metric metric_;                      ///< Distance metric type
    std::shared_ptr<Reader> disk_layout_reader_;  ///< Reader for disk layout data

    int L_ = 200;                ///< Search list size (affects recall vs. speed)
    int R_ = 64;                 ///< Maximum graph degree
    float p_val_ = 0.5;          ///< PQ compression rate parameter
    uint64_t disk_pq_dims_ = 8;  ///< Number of PQ dimensions for disk storage
    uint64_t sector_len_;        ///< Sector length for disk I/O alignment

    int64_t build_batch_num_ = 10;  ///< Number of batches for incremental build

    bool support_calc_distance_by_id_ = false;  ///< Flag indicating distance-by-ID support

    int64_t dim_;                ///< Vector dimension
    bool use_reference_ = true;  ///< Whether to use reference vectors in PQ
    bool use_opq_ = false;       ///< Whether to use Optimized Product Quantization
    bool use_bsa_ = false;       ///< Whether to use BSA (Beam Search Aware) optimization
    bool preload_;               ///< Whether to preload data into memory
    IndexStatus status_;         ///< Current index loading status
    bool empty_index_ = false;   ///< Flag indicating if index contains no data

    mutable std::shared_mutex rw_mutex_;  ///< Read-write mutex for thread safety

    IndexCommonParam common_param_;     ///< Copy of common parameters
    DiskannParameters diskann_params_;  ///< DiskANN-specific parameters

private:                              // Request Statistics
    mutable std::mutex stats_mutex_;  ///< Mutex for protecting statistics data

    mutable std::map<std::string, WindowResultQueue>
        result_queues_;  ///< Result queues for latency tracking
};

}  // namespace vsag