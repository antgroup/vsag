
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
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "data_type.h"
#include "hnsw_zparameters.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/conjugate_graph.h"
#include "impl/filter/filter_headers.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "index_impl.h"
#include "logger.h"
#include "typing.h"
#include "utils/window_result_queue.h"
#include "vsag/binaryset.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/iterator_context.h"
#include "vsag/readerset.h"

namespace vsag {

enum class VSAGIndexStatus : int {
    // start with -1

    DESTROYED = -1,  // index is destructing
    ALIVE            // index is alive
};

class HNSW : public Index {
public:
    HNSW(HnswParameters hnsw_params, const IndexCommonParam& index_common_param);

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
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        SAFE_CALL(return this->add(base));
    }

    tl::expected<bool, Error>
    Remove(int64_t id) override {
        SAFE_CALL(return this->remove(id));
    }

    tl::expected<bool, Error>
    UpdateId(int64_t old_id, int64_t new_id) override {
        SAFE_CALL(return this->update_id(old_id, new_id));
    }

    tl::expected<bool, Error>
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override {
        SAFE_CALL(return this->update_vector(id, new_base, force_update));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, invalid));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter));
    }

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

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search_internal(
            query, radius, parameters, (BitsetPtr) nullptr, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->range_search_internal(query, radius, parameters, filter, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->range_search_internal(query, radius, parameters, invalid, limited_size));
    }

    tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        SAFE_CALL(return this->feedback(query, k, parameters, global_optimum_tag_id));
    };

    tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids,
             uint32_t k,
             const std::string& parameters) override {
        SAFE_CALL(return this->pretrain(base_tag_ids, k, parameters));
    };

    virtual tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return this->calc_distance_by_id(vector, id));
    };

    virtual tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* vector, const int64_t* ids, int64_t count) const override {
        SAFE_CALL(return this->calc_distance_by_id(vector, ids, count));
    };

    virtual tl::expected<std::pair<int64_t, int64_t>, Error>
    GetMinAndMaxId() const override {
        SAFE_CALL(return this->get_min_and_max_id());
    };

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

public:
    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

public:
    tl::expected<void, Error>
    Merge(const std::vector<MergeUnit>& merge_units) override {
        SAFE_CALL(return this->merge(merge_units));
    }

public:
    bool
    IsValidStatus() const {
        return index_status_ != VSAGIndexStatus::DESTROYED;
    }

    void
    SetStatus(VSAGIndexStatus status) {
        index_status_ = status;
    }

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

    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override;

    [[nodiscard]] bool
    CheckIdExist(int64_t id) const override {
        return this->check_id_exist(id);
    }

    int64_t
    GetNumberRemoved() const override {
        return this->get_num_removed_elements();
    }

    int64_t
    GetNumElements() const override {
        return this->get_num_elements();
    }

    int64_t
    GetMemoryUsage() const override {
        return this->get_memory_usage();
    }

    uint64_t
    EstimateMemory(uint64_t num_elements) const override {
        return this->estimate_memory(num_elements);
    }

    std::string
    GetStats() const override;

    // used to test the integrity of graphs, used only in UT.
    bool
    CheckGraphIntegrity() const;

    tl::expected<bool, Error>
    InitMemorySpace();

    bool
    ExtractDataAndGraph(FlattenInterfacePtr& data,
                        GraphInterfacePtr& graph,
                        Vector<LabelType>& ids,
                        const IdMapFunction& func,
                        Allocator* allocator);

    bool
    SetDataAndGraph(FlattenInterfacePtr& data, GraphInterfacePtr& graph, Vector<LabelType>& ids);

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const DatasetPtr& base);

    tl::expected<bool, Error>
    remove(int64_t id);

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

    void
    get_vectors(const DatasetPtr& base, void** vectors_ptr, size_t* data_size_ptr) const;

    void
    set_dataset(const DatasetPtr& base, const void* vectors_ptr, uint32_t num_element) const;

    tl::expected<void, Error>
    merge(const std::vector<MergeUnit>& merge_units);

    tl::expected<float, Error>
    calc_distance_by_id(const float* vector, int64_t id) const;

    tl::expected<DatasetPtr, Error>
    calc_distance_by_id(const float* vector, const int64_t* ids, int64_t count) const;

    tl::expected<std::pair<int64_t, int64_t>, Error>
    get_min_and_max_id() const;

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

private:
    std::shared_ptr<hnswlib::AlgorithmInterface<float>> alg_hnsw_;
    std::shared_ptr<hnswlib::SpaceInterface> space_;

    bool use_conjugate_graph_;
    std::shared_ptr<ConjugateGraph> conjugate_graph_;

    int64_t dim_;
    bool use_static_ = false;
    bool empty_index_ = false;
    bool use_reversed_edges_ = false;
    bool is_init_memory_ = false;
    int64_t max_degree_{0};

    DataTypes type_;

    std::shared_ptr<Allocator> allocator_;

    mutable std::mutex stats_mutex_;
    mutable std::map<std::string, WindowResultQueue> result_queues_;

    mutable std::shared_mutex rw_mutex_;
    mutable std::shared_mutex index_status_mutex_;

    VSAGIndexStatus index_status_{VSAGIndexStatus::ALIVE};
    IndexFeatureList feature_list_{};
    const IndexCommonParam index_common_param_;
};

}  // namespace vsag
