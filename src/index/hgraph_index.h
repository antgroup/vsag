
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

#include <nlohmann/json.hpp>
#include <random>
#include <shared_mutex>

#include "../utils.h"
#include "ThreadPool.h"
#include "algorithm/hnswlib/algorithm_interface.h"
#include "algorithm/hnswlib/visited_list_pool.h"
#include "common.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "index_common_param.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {
class HGraphIndex : public Index {
public:
    struct CompareByFirst {
        constexpr bool
        operator()(std::pair<float, InnerIdType> const& a,
                   std::pair<float, InnerIdType> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    using MaxHeap = std::priority_queue<std::pair<float, InnerIdType>,
                                        Vector<std::pair<float, InnerIdType>>,
                                        CompareByFirst>;

    HGraphIndex(const JsonType& index_param, const IndexCommonParam& common_param) noexcept;

    void
    Init();

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& data) override {
        SAFE_CALL(return this->build(data));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& data) override {
        SAFE_CALL(return this->add(data));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        auto func = [&invalid](int64_t id) -> bool {
            if (invalid == nullptr) {
                return false;
            }
            int64_t bit_index = id & ROW_ID_MASK;
            return invalid->Test(bit_index);
        };
        SAFE_CALL(return this->knn_search(query, k, parameters, func));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, nullptr, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        BitsetOrCallbackFilter filter(invalid);
        SAFE_CALL(return this->range_search(query, radius, parameters, &filter, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size) const override {
        BitsetOrCallbackFilter callback(filter);
        SAFE_CALL(return this->range_search(query, radius, parameters, &callback, limited_size));
    }

    tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return this->calc_distance_by_id(vector, id));
    };

    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    };

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

    int64_t
    GetNumElements() const override {
        return this->basic_flatten_codes_->TotalCount();
    }

    // TODO(LHT): implement
    int64_t
    GetMemoryUsage() const override {
        return 0;
    }

    void
    SetBuildThreadsCount(uint64_t count) {
        this->build_thread_count_ = count;
        this->build_pool_->set_pool_size(count);
    }

public:
    FlattenInterfacePtr basic_flatten_codes_{nullptr};
    FlattenInterfacePtr high_precise_codes_{nullptr};
    Vector<GraphInterfacePtr> route_graphs_;
    GraphInterfacePtr bottom_graph_{nullptr};

    bool use_reorder_{false};

    int64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};

    const JsonType index_param_{};
    const IndexCommonParam common_param_{};

private:
    class InnerSearchParam {
    public:
        int topk_{0};
        float radius_{0.0f};
        InnerIdType ep_{0};
        uint64_t ef_{10};
        BaseFilterFunctor* is_id_allowed_{nullptr};
    };

    enum InnerSearchMode { KNN_SEARCH_MODE = 1, RANGE_SEARCH_MODE = 2 };

    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& data);

    tl::expected<std::vector<int64_t>, Error>
    add(const DatasetPtr& data);

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    // TODO(LHT): implement
    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 BaseFilterFunctor* filter_ptr,
                 int64_t limited_size) const {
        return Dataset::Make();
    };

    tl::expected<void, Error>
    serialize(std::ostream& out_stream) const;

    tl::expected<BinarySet, Error>
    serialize() const;

    void
    serialize(StreamWriter& writer) const;

    tl::expected<void, Error>
    deserialize(const ReaderSet& reader_set);

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(std::istream& in_stream);

    void
    deserialize(StreamReader& reader);

    tl::expected<float, Error>
    calc_distance_by_id(const float* vector, int64_t id) const;

    inline int
    get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        return (int)r;
    }

    void
    hnsw_add(const DatasetPtr& data);

    GraphInterfacePtr
    generate_one_route_graph();

    MaxHeap
    search_one_graph(const float* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& codes,
                     InnerIdType ep,
                     uint64_t ef,
                     BaseFilterFunctor* is_id_allowed = nullptr) const;

    void
    select_edges_by_heuristic(MaxHeap& edges,
                              uint64_t max_size,
                              const FlattenInterfacePtr& flatten);

    InnerIdType
    mutually_connect_new_element(InnerIdType cur_c,
                                 MaxHeap& top_candidates,
                                 GraphInterfacePtr graph,
                                 FlattenInterfacePtr flatten,
                                 bool is_update);

    void
    serialize_basic_info(StreamWriter& writer) const;

    void
    deserialize_basic_info(StreamReader& reader);

    uint64_t
    cal_serialize_size() const;

private:
    std::default_random_engine level_generator_{2021};
    double mult_{1.0};

    Allocator* allocator_{nullptr};

    UnorderedMap<LabelType, InnerIdType> label_lookup_;
    mutable std::shared_mutex label_lookup_mutex_{};  // lock for label_lookup_

    InnerIdType entry_point_id_{std::numeric_limits<InnerIdType>::max()};
    uint64_t max_level_{0};

    uint64_t ef_construct_{400};
    std::mutex global_mutex_;

    // Locks operations with element by label value
    mutable vsag::Vector<std::mutex> label_op_mutex_;

    static const uint64_t MAX_LABEL_OPERATION_LOCKS = 65536;

    std::shared_ptr<hnswlib::VisitedListPool> pool_{nullptr};

    mutable vsag::Vector<std::shared_mutex> neighbors_mutex_;

    std::unique_ptr<progschj::ThreadPool> build_pool_{nullptr};
    uint64_t build_thread_count_{100};
};
}  // namespace vsag
