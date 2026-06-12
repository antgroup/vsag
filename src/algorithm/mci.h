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

#include <atomic>
#include <shared_mutex>
#include <string>
#include <utility>

#include "container_types.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/distance_heap.h"
#include "inner_index_interface.h"
#include "mci_parameter.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

class HGraph;

class MCI : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    MCI(const MCIParameterPtr& param, const IndexCommonParam& common_param);

    MCI(const ParamPtr& param, const IndexCommonParam& common_param)
        : MCI(std::dynamic_pointer_cast<MCIParameter>(param), common_param){};

    ~MCI() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& data, AddMode mode = AddMode::DEFAULT) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] IndexType
    GetIndexType() const override {
        return IndexType::MCI;
    }

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_MCI;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return static_cast<int64_t>(this->total_count_.load());
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Train(const DatasetPtr& data) override;

private:
    void
    resize(uint64_t new_size);

    std::vector<int64_t>
    add_dataset(const DatasetPtr& data,
                bool train_if_empty,
                Vector<std::pair<InnerIdType, int64_t>>* inserted_ids);

    void
    clear_clique_index();

    void
    build_clique_index(const float* vectors,
                       uint64_t data_count,
                       const Vector<std::pair<InnerIdType, int64_t>>& inserted_ids);

    void
    load_clique_index(const std::string& clique_path, uint64_t total);

    Vector<Vector<InnerIdType>>
    build_knn_graph(const FlattenInterfacePtr& build_codes, uint64_t total) const;

    Vector<Vector<InnerIdType>>
    enumerate_maximal_cliques(const Vector<Vector<InnerIdType>>& graph,
                              const FlattenInterfacePtr& build_codes,
                              uint64_t total) const;

    [[nodiscard]] bool
    has_clique_index(uint64_t total) const;

    [[nodiscard]] uint64_t
    total_logical_clique_count() const;

    void
    reset_delta_clique_index(uint64_t total);

    void
    ensure_delta_node_rows(uint64_t total);

    DistHeapPtr
    scan_knn_candidates(const FlattenInterfacePtr& codes,
                        const ComputerInterfacePtr& computer,
                        const FilterPtr& inner_filter,
                        int64_t candidate_limit,
                        bool use_distance_lower_bound,
                        QueryContext& ctx,
                        DistanceRecordVector* rabitq_lower_bound_candidates,
                        uint32_t& dist_cmp) const;

    DistHeapPtr
    search_clique_candidates(const ComputerInterfacePtr& computer,
                             const FilterPtr& inner_filter,
                             const Vector<InnerIdType>* seed_inner_ids,
                             const MCISearchParameters& search_params,
                             int64_t candidate_limit,
                             QueryContext& ctx,
                             DistanceRecordVector* rabitq_lower_bound_candidates,
                             uint32_t& dist_cmp,
                             uint32_t& hops) const;

    const float*
    get_float_vectors(const DatasetPtr& data) const;

    FlattenInterfacePtr
    get_search_codes() const;

    FlattenInterfacePtr
    get_reorder_codes() const;

    void
    cal_memory_usage();

    void
    load_hgraph_index(const std::string& index_path);

    [[nodiscard]] bool
    should_use_hgraph_hybrid(const SearchRequest& request, float valid_ratio) const;

    [[nodiscard]] std::string
    get_hgraph_search_params(const std::string& request_params) const;

    [[nodiscard]] DatasetPtr
    search_hgraph_hybrid(const SearchRequest& request, float valid_ratio) const;

    [[nodiscard]] DatasetPtr
    build_dataset_from_heap(DistHeapPtr& heap) const;

    void
    collect_node_clique_ids(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const;

    void
    incremental_update_clique(InnerIdType new_inner_id, const float* vector);

    [[nodiscard]] Vector<InnerIdType>
    find_knn_for_new_node(InnerIdType new_inner_id, const float* vector) const;

    bool
    try_join_existing_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids);

    void
    build_incremental_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids);

    void
    append_node_to_clique(InnerIdType node_id, InnerIdType clique_id);

    void
    append_new_clique(const Vector<InnerIdType>& members);

private:
    FlattenInterfacePtr base_codes_{nullptr};
    FlattenInterfacePtr reorder_codes_{nullptr};
    std::atomic<uint64_t> total_count_{0};
    std::atomic<uint64_t> max_capacity_{0};

    Vector<InnerIdType> p_maxc_;
    Vector<InnerIdType> maxcs_;
    Vector<InnerIdType> p_node_to_cid_;
    Vector<InnerIdType> node_to_cids_;
    uint64_t total_clique_count_{0};
    Vector<Vector<InnerIdType>> delta_cliques_;
    Vector<Vector<InnerIdType>> delta_clique_extra_;
    Vector<Vector<InnerIdType>> delta_node_to_cids_;

    uint64_t max_degree_{32};
    uint64_t mcs_{200};
    uint64_t clique_max_{50};
    float alpha_{1.2F};
    float join_ratio_threshold_{0.6F};
    uint64_t added_mct_{3};
    std::string knng_path_{};
    std::string clique_path_{};
    bool reorder_by_base_{false};

    std::shared_ptr<HGraph> hgraph_index_{nullptr};
    bool use_hgraph_hybrid_{false};
    float hgraph_valid_ratio_threshold_{1.0F};
    std::string hgraph_index_path_{};
    int64_t hgraph_ef_search_{100};

    mutable std::shared_mutex global_mutex_;
    mutable std::shared_mutex add_mutex_;
};

}  // namespace vsag
