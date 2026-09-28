
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

#include <limits>

#include "../index_search_parameter.h"
#include "../inner_index_parameter.h"
#include "data_type.h"
#include "impl/pipnn/pipnn_graph_builder.h"
#include "utils/filter_search_skip_strategy.h"
#include "utils/pointer_define.h"
#include "vsag/constants.h"

namespace vsag {
DEFINE_POINTER2(ExtraInfoDataCellParam, ExtraInfoDataCellParameter);
DEFINE_POINTER2(FlattenInterfaceParam, FlattenInterfaceParameter);
DEFINE_POINTER2(GraphInterfaceParam, GraphInterfaceParameter);
DEFINE_POINTER2(SparseGraphDatacellParam, SparseGraphDatacellParameter);
DEFINE_POINTER(ODescentParameter);

DEFINE_POINTER(HGraphParameter);

struct HGraphMCIParameters {
    bool enabled{false};
    uint64_t mcs{200};
    // Minimum size of a maximal clique the full build must keep; the clique itself is stored in
    // full, so this is a lower bound and not a cap.
    uint64_t clique_max{50};
    float alpha{1.2F};
    std::string knng_source{HGRAPH_MCI_KNNG_SOURCE_HGRAPH};
    std::string knng_path{};
    float incremental_join_ratio_threshold{0.6F};
    uint64_t incremental_added_mct{3};
    uint64_t incremental_clique_max{50};
};
class HGraphParameter : public InnerIndexParameter {
public:
    explicit HGraphParameter(const JsonType& json);

    HGraphParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    FlattenInterfaceParamPtr base_codes_param{nullptr};
    GraphInterfaceParamPtr bottom_graph_param{nullptr};
    SparseGraphDatacellParamPtr hierarchical_graph_param{nullptr};

    ODescentParameterPtr odescent_param{nullptr};
    PiPNNGraphBuilderParameter pipnn_param{};

    std::string graph_type{GRAPH_TYPE_VALUE_NSW};

    bool use_elp_optimizer{false};
    bool ignore_reorder{false};
    bool build_by_base{false};
    bool rabitq_fused_datacell{false};

    uint64_t ef_construction{400};
    uint64_t resize_increase_count_bit{DEFAULT_RESIZE_INCREASE_COUNT_BIT};
    float alpha{1.0F};

    bool support_duplicate{false};
    bool deduplicate_storage{false};
    float duplicate_distance_threshold{0.0F};
    bool support_force_remove{false};

    bool persist_source_id{false};
    bool use_conjugate_graph{false};

    HGraphMCIParameters mci_parameters{};

    DataTypes data_type{DataTypes::DATA_TYPE_FLOAT};

    std::string name;
};

class HGraphSearchParameters : public IndexSearchParameter {
public:
    static HGraphSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{30};
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};
    bool use_reorder{false};
    bool use_extra_info_filter{false};
    bool rabitq_one_bit_search{false};
    bool use_mci{true};
    // Dynamic neighbor traversal: expand the sparse Hgraph neighbors and the MCI
    // clique members inside one traversal, with the MCI part bounded by the
    // virtual-overhead boundary hybrid_vob.
    bool use_hybrid_traversal{false};
    double hybrid_vob{0.0};
    // O_filter / O_dist of the virtual-overhead model, i.e. the relative cost of one
    // predicate filtering against one distance computation. hybrid_vob is therefore
    // expressed in units of O_dist.
    double hybrid_filter_cost_ratio{1.0};
    bool use_conjugate_graph_search{true};
    float mci_seed_ratio{0.1F};
    // Seed budget is `max(ceil(sqrt(N) * mci_seed_ratio), ceil(mci_seed_coverage * valid))`
    // whenever the coverage part fits into `mci_seed_max_count` (0 = unlimited) and does not
    // exceed N; otherwise the coverage part is dropped entirely rather than truncated. Seeding
    // every valid point makes the seed phase exact, which is cheaper than the expansion tail
    // whenever the valid set is small.
    float mci_seed_coverage{1.0F};
    // Cap for the coverage term above. Enumerating the valid set only pays off while the
    // enumeration itself is cheap, so it is capped at a size that still stays in cache
    // (32768 inner ids are 128 KiB) instead of growing a wide predicate into a scan. 0
    // means unlimited.
    int64_t mci_seed_max_count{32768};
    float mci_hgraph_valid_ratio_threshold{0.05F};
    // If > 0 and the active filter's ValidRatio() <= brute_force_threshold,
    // the search bypasses the graph traversal and runs an exact scan over the
    // valid inner ids using the best available flatten codes. Default 0
    // preserves existing behaviour.
    float brute_force_threshold{0.0F};
    float rabitq_error_rate{std::numeric_limits<float>::quiet_NaN()};
    float skip_ratio{0.2F};
    FilterSearchSkipStrategyType skip_strategy_type{
        FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE};

private:
    HGraphSearchParameters() = default;
};

}  // namespace vsag
