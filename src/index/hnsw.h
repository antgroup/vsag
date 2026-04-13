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
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithm/hnswlib/hnswlib.h"
#include "common.h"
#include "data_type.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "hnsw_zparameters.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/conjugate_graph.h"
#include "impl/filter/filter_headers.h"
#include "impl/logger/logger.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "index_impl.h"
#include "typing.h"
#include "utils/util_functions.h"
#include "utils/window_result_queue.h"
#include "vsag/binaryset.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/iterator_context.h"
#include "vsag/readerset.h"

namespace vsag {

/**
 * @brief Enumeration representing the lifecycle status of an index.
 *
 * This enum is used to track whether an index is active or being destroyed,
 * enabling safe concurrent access during destruction.
 */
enum class VSAGIndexStatus : int {
    // start with -1

    DESTROYED = -1,  ///< Index is being destructed, access should be blocked
    ALIVE            ///< Index is alive and operational
};

/**
 * @brief Hierarchical Navigable Small World (HNSW) graph index implementation.
 *
 * This class implements a high-performance approximate nearest neighbor search
 * index using the HNSW algorithm. It supports various operations including:
 * - Vector insertion and deletion
 * - K-NN and range search with optional filtering
 * - Serialization and deserialization
 * - Graph merging and feedback-based optimization
 *
 * The implementation uses a multi-layer graph structure where each layer is
 * a navigable small world graph, providing logarithmic search complexity.
 * Thread safety is ensured through read-write locks for concurrent operations.
 */
class HNSW : public Index {
public:
    /**
     * @brief Constructs an HNSW index with the specified parameters.
     *
     * @param hnsw_params Configuration parameters for the HNSW algorithm.
     * @param index_common_param Common parameters shared across index operations.
     */
    HNSW(HnswParameters hnsw_params, const IndexCommonParam& index_common_param);

    /**
     * @brief Destructor that safely tears down the index.
     *
     * Sets the index status to DESTROYED before releasing resources to
     * prevent concurrent access during destruction.
     */
    virtual ~HNSW() {
        {
            std::unique_lock status_lock(index_status_mutex_);
            this->SetStatus(VSAGIndexStatus::DESTROYED);
        }

        alg_hnsw_ = nullptr;
        if (use_conjugate_graph_) {
            conjugate_graph_.reset();
        }
        allocator_.reset();
    }

public:
    /**
     * @brief Builds the index from a dataset.
     *
     * @param base The dataset containing vectors to index.
     * @return Expected vector of inserted IDs on success, or Error on failure.
     */
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    /**
     * @brief Returns the index type identifier.
     *
     * @return IndexType::HNSW.
     */
    IndexType
    GetIndexType() const override {
        return IndexType::HNSW;
    }

    /**
     * @brief Adds vectors to an existing index.
     *
     * @param base The dataset containing vectors to add.
     * @param mode The add mode (currently only KEEP_TOMBSTONE supported).
     * @return Expected vector of inserted IDs on success, or Error on failure.
     */
    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base, AddMode mode = AddMode::DEFAULT) override {
        // TODO(LHT): HNSW only support KEEP_TOMBSTONE mode
        SAFE_CALL(return this->add(base));
    }

    /**
     * @brief Removes vectors from the index by ID.
     *
     * @param ids The IDs of vectors to remove.
     * @param mode The remove mode (currently only MARK_REMOVE supported).
     * @return Expected count of removed vectors on success, or Error on failure.
     */
    tl::expected<uint32_t, Error>
    Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE) override {
        // TODO(LHT): HNSW only support MARK_DELETE mode
        SAFE_CALL(return this->remove(ids));
    }

    /**
     * @brief Updates the ID of an existing vector.
     *
     * @param old_id The current ID of the vector.
     * @param new_id The new ID to assign.
     * @return Expected true on success, or Error on failure.
     */
    tl::expected<bool, Error>
    UpdateId(int64_t old_id, int64_t new_id) override {
        SAFE_CALL(return this->update_id(old_id, new_id));
    }

    /**
     * @brief Updates the vector data for an existing ID.
     *
     * @param id The ID of the vector to update.
     * @param new_base The new vector data.
     * @param force_update Whether to force update without checking existence.
     * @return Expected true on success, or Error on failure.
     */
    tl::expected<bool, Error>
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override {
        SAFE_CALL(return this->update_vector(id, new_base, force_update));
    }

    /**
     * @brief Performs k-nearest neighbor search with callback filter.
     *
     * @param query The query vector dataset.
     * @param k The number of nearest neighbors to return.
     * @param parameters JSON string containing search parameters.
     * @param filter Callback function to filter candidates by ID.
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, filter));
    }

    /**
     * @brief Performs k-nearest neighbor search with bitset filter.
     *
     * @param query The query vector dataset.
     * @param k The number of nearest neighbors to return.
     * @param parameters JSON string containing search parameters.
     * @param invalid Bitset marking IDs that should be excluded from results.
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, invalid));
    }

    /**
     * @brief Performs k-nearest neighbor search with filter object.
     *
     * @param query The query vector dataset.
     * @param k The number of nearest neighbors to return.
     * @param parameters JSON string containing search parameters.
     * @param filter Filter object for advanced filtering.
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter));
    }

    /**
     * @brief Performs k-nearest neighbor search with iterator context.
     *
     * @param query The query vector dataset.
     * @param k The number of nearest neighbors to return.
     * @param parameters JSON string containing search parameters.
     * @param filter Filter object for advanced filtering.
     * @param filter_ctx Iterator context for incremental filtering.
     * @param is_last_search Whether this is the final search iteration.
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              vsag::IteratorContext*& filter_ctx,
              bool is_last_search) const override {
        SAFE_CALL(return this->knn_search(
            query, k, parameters, filter, nullptr, &filter_ctx, is_last_search));
    }

    /**
     * @brief Performs k-nearest neighbor search with search parameters.
     *
     * @param query The query vector dataset.
     * @param k The number of nearest neighbors to return.
     * @param search_param Search parameters including filter and allocator.
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const override {
        if (search_param.is_iter_filter) {
            SAFE_CALL(return this->knn_search(query,
                                              k,
                                              search_param.parameters,
                                              search_param.filter,
                                              search_param.allocator,
                                              &search_param.iter_ctx,
                                              search_param.is_last_search));
        } else {
            SAFE_CALL(return this->knn_search(
                query, k, search_param.parameters, search_param.filter, search_param.allocator));
        }
    }

    /**
     * @brief Performs range search within a radius.
     *
     * @param query The query vector dataset.
     * @param radius The search radius threshold.
     * @param parameters JSON string containing search parameters.
     * @param limited_size Maximum number of results to return (-1 for unlimited).
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search_internal(
            query, radius, parameters, (BitsetPtr) nullptr, limited_size));
    }

    /**
     * @brief Performs range search with callback filter.
     *
     * @param query The query vector dataset.
     * @param radius The search radius threshold.
     * @param parameters JSON string containing search parameters.
     * @param filter Callback function to filter candidates by ID.
     * @param limited_size Maximum number of results to return (-1 for unlimited).
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->range_search_internal(query, radius, parameters, filter, limited_size));
    }

    /**
     * @brief Performs range search with bitset filter.
     *
     * @param query The query vector dataset.
     * @param radius The search radius threshold.
     * @param parameters JSON string containing search parameters.
     * @param invalid Bitset marking IDs that should be excluded from results.
     * @param limited_size Maximum number of results to return (-1 for unlimited).
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->range_search_internal(query, radius, parameters, invalid, limited_size));
    }

    /**
     * @brief Provides feedback to improve search quality.
     *
     * @param query The query vector dataset.
     * @param k The number of neighbors considered for feedback.
     * @param parameters JSON string containing feedback parameters.
     * @param global_optimum_tag_id ID of the known optimal result (optional).
     * @return Expected count of graph edges updated on success, or Error on failure.
     */
    tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        SAFE_CALL(return this->feedback(query, k, parameters, global_optimum_tag_id));
    };

    /**
     * @brief Pretrains the index with known optimal neighbors.
     *
     * @param base_tag_ids IDs of vectors with known optimal neighbors.
     * @param k The number of neighbors to pretrain per vector.
     * @param parameters JSON string containing pretrain parameters.
     * @return Expected count of graph edges added on success, or Error on failure.
     */
    tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids,
             uint32_t k,
             const std::string& parameters) override {
        SAFE_CALL(return this->pretrain(base_tag_ids, k, parameters));
    };

    /**
     * @brief Calculates distance between a vector and an indexed vector by ID.
     *
     * @param vector The query vector.
     * @param id The ID of the indexed vector.
     * @param calculate_precise_distance Whether to calculate precise distance.
     * @return Expected distance value on success, or Error on failure.
     */
    virtual tl::expected<float, Error>
    CalcDistanceById(const float* vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override {
        SAFE_CALL(return this->calc_distance_by_id(vector, id));
    };

    /**
     * @brief Calculates distances between a vector and multiple indexed vectors.
     *
     * @param vector The query vector.
     * @param ids Array of IDs of indexed vectors.
     * @param count Number of IDs in the array.
     * @param calculate_precise_distance Whether to calculate precise distances.
     * @return Expected dataset containing IDs and distances on success, or Error on failure.
     */
    virtual tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* vector,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override {
        SAFE_CALL(return this->calc_distance_by_id(vector, ids, count));
    };

    /**
     * @brief Gets the minimum and maximum IDs in the index.
     *
     * @return Expected pair of (min_id, max_id) on success, or Error on failure.
     */
    virtual tl::expected<std::pair<int64_t, int64_t>, Error>
    GetMinAndMaxId() const override {
        SAFE_CALL(return this->get_min_and_max_id());
    };

    /**
     * @brief Retrieves detailed data by name for debugging/analysis.
     *
     * @param name The name of the data to retrieve.
     * @param info Reference to store the retrieved detail info.
     * @return Expected detail data pointer on success, or Error on failure.
     */
    tl::expected<DetailDataPtr, Error>
    GetDetailDataByName(const std::string& name, IndexDetailInfo& info) const override {
        SAFE_CALL(return this->get_detail_data_by_name(name, info));
    }

    /**
     * @brief Retrieves raw vectors by their IDs.
     *
     * @param ids Array of vector IDs to retrieve.
     * @param count Number of IDs in the array.
     * @param specified_allocator Optional allocator for result memory.
     * @return Expected dataset containing vectors on success, or Error on failure.
     */
    tl::expected<DatasetPtr, Error>
    GetRawVectorByIds(const int64_t* ids,
                      int64_t count,
                      Allocator* specified_allocator = nullptr) const override {
        SAFE_CALL(return this->get_vectors_by_id(ids, count, specified_allocator));
    }

public:
    /**
     * @brief Serializes the index to a BinarySet.
     *
     * @return Expected BinarySet containing serialized data on success, or Error on failure.
     */
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    /**
     * @brief Deserializes the index from a Binary Set.
     *
     * @param binary_set Binary set containing serialized index data.
     * @return Expected void on success, or Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    /**
     * @brief Deserializes the index from a Reader Set.
     *
     * @param reader_set Reader set providing access to serialized data.
     * @return Expected void on success, or Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

public:
    /**
     * @brief Serializes the index to an output stream.
     *
     * @param out_stream Output stream to write serialized data.
     * @return Expected void on success, or Error on failure.
     */
    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    /**
     * @brief Deserializes the index from an input stream.
     *
     * @param in_stream Input stream to read serialized data from.
     * @return Expected void on success, or Error on failure.
     */
    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

public:
    /**
     * @brief Merges multiple index segments into this index.
     *
     * @param merge_units Vector of merge units to combine.
     * @return Expected void on success, or Error on failure.
     */
    tl::expected<void, Error>
    Merge(const std::vector<MergeUnit>& merge_units) override {
        SAFE_CALL(return this->merge(merge_units));
    }

public:
    /**
     * @brief Checks if the index is in a valid operational state.
     *
     * @return true if the index is alive and operational, false if destroyed.
     */
    bool
    IsValidStatus() const {
        return index_status_ != VSAGIndexStatus::DESTROYED;
    }

    /**
     * @brief Sets the index status.
     *
     * @param status The new status to set.
     */
    void
    SetStatus(VSAGIndexStatus status) {
        index_status_ = status;
    }

    /**
     * @brief Returns a string representation of the current status.
     *
     * @return String representation ("Destroyed", "Alive", or empty).
     */
    std::string
    PrintStatus() const {
        switch (index_status_) {
            case VSAGIndexStatus::DESTROYED:
                return "Destroyed";
            case VSAGIndexStatus::ALIVE:
                return "Alive";
            default:
                return "";
        }
    }

    /**
     * @brief Checks if a feature is supported by this index.
     *
     * @param feature The feature to check.
     * @return true if the feature is supported, false otherwise.
     */
    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override;

    /**
     * @brief Checks if an ID exists in the index.
     *
     * @param id The ID to check.
     * @return true if the ID exists, false otherwise.
     */
    [[nodiscard]] bool
    CheckIdExist(int64_t id) const override {
        return this->check_id_exist(id);
    }

    /**
     * @brief Returns the number of removed (marked deleted) elements.
     *
     * @return Count of removed elements.
     */
    int64_t
    GetNumberRemoved() const override {
        return this->get_num_removed_elements();
    }

    /**
     * @brief Returns the total number of elements in the index.
     *
     * @return Number of elements.
     */
    int64_t
    GetNumElements() const override {
        return this->get_num_elements();
    }

    /**
     * @brief Returns the memory usage of the index in bytes.
     *
     * @return Memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override {
        return this->get_memory_usage();
    }

    /**
     * @brief Estimates memory usage for a given number of elements.
     *
     * @param num_elements Number of elements to estimate for.
     * @return Estimated memory usage in bytes.
     */
    uint64_t
    EstimateMemory(uint64_t num_elements) const override {
        return this->estimate_memory(num_elements);
    }

    /**
     * @brief Returns statistical information about the index.
     *
     * @return JSON string containing index statistics.
     */
    std::string
    GetStats() const override;

    /**
     * @brief Checks the integrity of the graph structure.
     *
     * Used only in unit tests to verify graph correctness.
     *
     * @return true if the graph is intact, false otherwise.
     */
    bool
    CheckGraphIntegrity() const;

    /**
     * @brief Initializes memory space for the index.
     *
     * @return Expected true on success, or Error on failure.
     */
    tl::expected<bool, Error>
    InitMemorySpace();

    /**
     * @brief Extracts data and graph from the index.
     *
     * @param[out] data Output flatten interface for vector data.
     * @param[out] graph Output graph interface.
     * @param[out] ids Output vector of labels.
     * @param func Function to map internal IDs to external labels.
     * @param allocator Allocator for output data structures.
     * @return true on success, false on failure.
     */
    bool
    ExtractDataAndGraph(FlattenInterfacePtr& data,
                        GraphInterfacePtr& graph,
                        Vector<LabelType>& ids,
                        const IdMapFunction& func,
                        Allocator* allocator);

    /**
     * @brief Sets data and graph into the index.
     *
     * @param data Flatten interface containing vector data.
     * @param graph Graph interface containing graph structure.
     * @param ids Vector of labels for the data.
     * @return true on success, false on failure.
     */
    bool
    SetDataAndGraph(FlattenInterfacePtr& data, GraphInterfacePtr& graph, Vector<LabelType>& ids);

    /**
     * @brief Marks the index as immutable (read-only).
     *
     * @return Expected void on success, or Error on failure.
     */
    tl::expected<void, Error>
    SetImmutable() override {
        SAFE_CALL(this->set_immutable());
    }

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const DatasetPtr& base);

    tl::expected<uint32_t, Error>
    remove(const std::vector<int64_t>& ids);

    tl::expected<bool, Error>
    update_id(int64_t old_id, int64_t new_id);

    tl::expected<bool, Error>
    update_vector(int64_t id, const DatasetPtr& new_base, bool force_update);

    template <typename FilterType>
    tl::expected<DatasetPtr, Error>
    knn_search_internal(const DatasetPtr& query,
                        int64_t k,
                        const std::string& parameters,
                        const FilterType& filter_obj) const;

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const FilterPtr& filter_ptr,
               vsag::Allocator* allocator = nullptr,
               vsag::IteratorContext** iter_ctx = nullptr,
               bool is_last_filter = false) const;

    template <typename FilterType>
    tl::expected<DatasetPtr, Error>
    range_search_internal(const DatasetPtr& query,
                          float radius,
                          const std::string& parameters,
                          const FilterType& filter_obj,
                          int64_t limited_size) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const FilterPtr& filter_ptr,
                 int64_t limited_size) const;

    tl::expected<uint32_t, Error>
    feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id);

    tl::expected<uint32_t, Error>
    feedback(const DatasetPtr& result, int64_t global_optimum_tag_id, int64_t k);

    tl::expected<DatasetPtr, Error>
    brute_force(const DatasetPtr& query, int64_t k);

    tl::expected<uint32_t, Error>
    pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters);

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
    merge(const std::vector<MergeUnit>& merge_units);

    tl::expected<float, Error>
    calc_distance_by_id(const float* vector, int64_t id) const;

    tl::expected<DatasetPtr, Error>
    calc_distance_by_id(const float* vector, const int64_t* ids, int64_t count) const;

    tl::expected<std::pair<int64_t, int64_t>, Error>
    get_min_and_max_id() const;

    tl::expected<DatasetPtr, Error>
    get_vectors_by_id(const int64_t* ids, int64_t count, Allocator* specified_allocator) const;

    bool
    check_id_exist(int64_t id) const;

    int64_t
    get_num_elements() const;

    int64_t
    get_num_removed_elements() const;

    int64_t
    get_memory_usage() const;

    void
    init_feature_list();

    uint64_t
    estimate_memory(uint64_t num_elements) const;

    void
    set_immutable();

    DetailDataPtr
    get_detail_data_by_name(const std::string& name, IndexDetailInfo& info) const;

private:
    std::shared_ptr<hnswlib::AlgorithmInterface<float>>
        alg_hnsw_;                                    ///< Core HNSW algorithm implementation
    std::shared_ptr<hnswlib::SpaceInterface> space_;  ///< Distance metric space

    bool use_conjugate_graph_;  ///< Whether conjugate graph optimization is enabled
    std::shared_ptr<ConjugateGraph>
        conjugate_graph_;  ///< Optional conjugate graph for enhanced connectivity

    int64_t dim_;                      ///< Vector dimension
    bool empty_index_ = false;         ///< Flag indicating an empty index
    bool use_reversed_edges_ = false;  ///< Whether reversed edges are stored for graph operations
    bool is_init_memory_ = false;      ///< Whether memory space has been initialized
    int64_t max_degree_{0};            ///< Maximum degree of the graph

    DataTypes type_;  ///< Data type of vectors

    std::shared_ptr<Allocator> allocator_;  ///< Memory allocator for index operations

    mutable std::mutex stats_mutex_;  ///< Mutex for statistics operations
    mutable std::map<std::string, WindowResultQueue>
        result_queues_;  ///< Result queues for windowed search

    mutable std::shared_mutex rw_mutex_;            ///< Read-write lock for concurrent access
    mutable std::shared_mutex index_status_mutex_;  ///< Mutex for index status changes

    VSAGIndexStatus index_status_{
        VSAGIndexStatus::ALIVE};                 ///< Current lifecycle status of the index
    IndexFeatureList feature_list_{};            ///< List of supported features
    const IndexCommonParam index_common_param_;  ///< Common parameters for index operations

    bool use_old_serial_format_{false};  ///< Whether to use legacy serialization format
};

}  // namespace vsag