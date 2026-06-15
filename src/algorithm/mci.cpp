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

#include "mci.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>

#include "algorithm/hgraph/hgraph.h"
#include "common.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "dataset_impl.h"
#include "impl/filter/black_list_filter.h"
#include "impl/filter/combined_filter.h"
#include "impl/filter/inner_id_wrapper_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/logger/logger.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "impl/reorder/flatten_reorder.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "io/memory_io_parameter.h"
#include "quantization/scalar_quantization/scalar_quantizer_parameter.h"
#include "storage/serialization.h"
#include "utils/util_functions.h"
#include "vsag/constants.h"

namespace vsag {
namespace {

const std::string MCI_PARAMS_TEMPLATE = R"(
    {
        "type": "mci",
        "use_reorder": false,
        "reorder_source": "precise",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50,
        "alpha": 1.2,
        "join_ratio_threshold": 0.6,
        "added_mct": 3,
        "knng_path": "",
        "clique_path": "",
        "use_hgraph_hybrid": false,
        "hgraph_valid_ratio_threshold": 1.0,
        "hgraph_index_path": "",
        "hgraph_ef_search": 100,
        "base_codes": {
            "io_params": {
                "type": "memory_io",
                "file_path": "./default_file_path"
            },
            "codes_type": "flatten",
            "quantization_params": {
                "type": "fp32",
                "sq4_uniform_trunc_rate": 0.05,
                "pca_dim": 0,
                "rabitq_version": "standard",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 1,
                "rabitq_error_rate": 1.9,
                "tq_chain": "",
                "nbits": 8,
                "pq_dim": 1,
                "hold_molds": false
            }
        },
        "precise_codes": {
            "io_params": {
                "type": "block_memory_io",
                "file_path": "./default_file_path"
            },
            "codes_type": "flatten",
            "quantization_params": {
                "type": "fp32",
                "sq4_uniform_trunc_rate": 0.05,
                "pca_dim": 0,
                "pq_dim": 1,
                "hold_molds": false
            }
        },
        "build_thread_count": 1,
        "use_attribute_filter": false,
        "attr_params": {
            "has_buckets": true
        }
    })";

Vector<Vector<uint64_t>>
run_local_ccr_mce(const Vector<InnerIdType>& local_nodes,
                  const Vector<uint8_t>& local_edges,
                  const std::vector<std::atomic<uint32_t>>& num_cliques_per_node,
                  uint64_t threshold,
                  uint64_t max_saved_cliques,
                  Allocator* allocator) {
    Vector<Vector<uint64_t>> saved_cliques(allocator);
    const auto local_count = local_nodes.size();
    if (local_count < threshold or max_saved_cliques == 0) {
        return saved_cliques;
    }
    auto has_local_edge = [&](uint64_t lhs, uint64_t rhs) {
        return local_edges[lhs * local_count + rhs] != 0;
    };

    Vector<uint64_t> degree(local_count, 0, allocator);
    for (uint64_t i = 0; i < local_count; ++i) {
        for (uint64_t j = 0; j < local_count; ++j) {
            if (has_local_edge(i, j)) {
                ++degree[i];
            }
        }
    }

    Vector<uint64_t> active_to_local(allocator);
    Vector<uint64_t> local_to_active(local_count, local_count, allocator);
    active_to_local.reserve(local_count);
    for (uint64_t local_id = 0; local_id < local_count; ++local_id) {
        if (degree[local_id] == 0) {
            continue;
        }
        local_to_active[local_id] = active_to_local.size();
        active_to_local.push_back(local_id);
    }
    if (active_to_local.size() < threshold) {
        return saved_cliques;
    }

    Vector<uint64_t> active_degree(allocator);
    active_degree.reserve(active_to_local.size());
    for (auto local_id : active_to_local) {
        active_degree.push_back(degree[local_id]);
    }

    Vector<uint64_t> order_to_local(allocator);
    Vector<uint64_t> local_to_order(local_count, local_count, allocator);
    Vector<uint8_t> removed(active_to_local.size(), 0, allocator);
    order_to_local.reserve(active_to_local.size());
    for (uint64_t order = 0; order < active_to_local.size(); ++order) {
        uint64_t best = active_to_local.size();
        uint64_t best_degree = std::numeric_limits<uint64_t>::max();
        for (uint64_t active_id = 0; active_id < active_to_local.size(); ++active_id) {
            const auto local_id = active_to_local[active_id];
            if (removed[active_id] == 0 and
                (active_degree[active_id] < best_degree or
                 (active_degree[active_id] == best_degree and
                  (best == active_to_local.size() or local_id < active_to_local[best])))) {
                best = active_id;
                best_degree = active_degree[active_id];
            }
        }
        if (best == active_to_local.size()) {
            break;
        }
        const auto best_local = active_to_local[best];
        removed[best] = 1;
        order_to_local.push_back(best_local);
        local_to_order[best_local] = order;
        for (uint64_t active_id = 0; active_id < active_to_local.size(); ++active_id) {
            const auto neighbor = active_to_local[active_id];
            if (removed[active_id] == 0 and has_local_edge(best_local, neighbor) and
                active_degree[active_id] > 0) {
                --active_degree[active_id];
            }
        }
    }
    if (order_to_local.size() < threshold) {
        return saved_cliques;
    }

    const auto core_count = order_to_local.size();
    Vector<uint8_t> core_edges(core_count * core_count, 0, allocator);
    auto has_core_edge = [&](uint64_t lhs, uint64_t rhs) {
        return core_edges[lhs * core_count + rhs] != 0;
    };
    auto set_core_edge = [&](uint64_t lhs, uint64_t rhs) {
        core_edges[lhs * core_count + rhs] = 1;
    };
    Vector<uint8_t> must_contain(core_count, 0, allocator);
    uint64_t remaining_must = 0;
    for (uint64_t lhs = 0; lhs < local_count; ++lhs) {
        const auto core_lhs = local_to_order[lhs];
        if (core_lhs >= core_count) {
            continue;
        }
        if (num_cliques_per_node[local_nodes[lhs]].load(std::memory_order_relaxed) == 0) {
            must_contain[core_lhs] = 1;
            ++remaining_must;
        }
        for (uint64_t rhs = lhs + 1; rhs < local_count; ++rhs) {
            if (not has_local_edge(lhs, rhs)) {
                continue;
            }
            const auto core_rhs = local_to_order[rhs];
            if (core_rhs < core_count) {
                set_core_edge(core_lhs, core_rhs);
                set_core_edge(core_rhs, core_lhs);  // NOLINT(readability-suspicious-call-argument)
            }
        }
    }
    if (remaining_must == 0) {
        return saved_cliques;
    }

    auto state_has_must = [&](const Vector<uint64_t>& current, const Vector<uint64_t>& candidates) {
        for (auto node : current) {
            if (must_contain[node] != 0) {
                return true;
            }
        }
        for (auto node : candidates) {
            if (must_contain[node] != 0) {
                return true;
            }
        }
        return false;
    };

    auto save_clique = [&](const Vector<uint64_t>& current) {
        if (current.size() < threshold or saved_cliques.size() >= max_saved_cliques) {
            return false;
        }
        bool has_must = false;
        for (auto node : current) {
            if (must_contain[node] != 0) {
                has_must = true;
                break;
            }
        }
        if (not has_must) {
            return false;
        }

        saved_cliques.push_back(Vector<uint64_t>(allocator));
        auto& saved = saved_cliques.back();
        saved.reserve(current.size());
        for (auto node : current) {
            saved.push_back(order_to_local[node]);
            if (must_contain[node] != 0) {
                must_contain[node] = 0;
                --remaining_must;
            }
        }
        return remaining_must == 0 or saved_cliques.size() >= max_saved_cliques;
    };

    std::function<void(Vector<uint64_t>&, Vector<uint64_t>&, Vector<uint64_t>&)> expand =
        [&](Vector<uint64_t>& current, Vector<uint64_t>& candidates, Vector<uint64_t>& excluded) {
            if (remaining_must == 0 or saved_cliques.size() >= max_saved_cliques) {
                return;
            }
            if (current.size() + candidates.size() < threshold) {
                return;
            }
            if (not state_has_must(current, candidates)) {
                return;
            }
            if (candidates.empty() and excluded.empty()) {
                save_clique(current);
                return;
            }

            uint64_t pivot = core_count;
            uint64_t pivot_degree = 0;
            auto update_pivot = [&](uint64_t node) {
                uint64_t node_degree = 0;
                for (auto candidate : candidates) {
                    if (has_core_edge(node, candidate)) {
                        ++node_degree;
                    }
                }
                if (pivot == core_count or node_degree > pivot_degree) {
                    pivot = node;
                    pivot_degree = node_degree;
                }
            };
            for (auto node : candidates) {
                update_pivot(node);
            }
            for (auto node : excluded) {
                update_pivot(node);
            }

            Vector<uint64_t> branches(allocator);
            branches.reserve(candidates.size());
            for (auto node : candidates) {
                if (pivot == core_count or not has_core_edge(pivot, node)) {
                    branches.push_back(node);
                }
            }

            for (auto node : branches) {
                auto iter = std::find(candidates.begin(), candidates.end(), node);
                if (iter == candidates.end()) {
                    continue;
                }

                current.push_back(node);
                Vector<uint64_t> next_candidates(allocator);
                Vector<uint64_t> next_excluded(allocator);
                next_candidates.reserve(candidates.size());
                next_excluded.reserve(excluded.size());
                for (auto candidate : candidates) {
                    if (has_core_edge(node, candidate)) {
                        next_candidates.push_back(candidate);
                    }
                }
                for (auto excluded_node : excluded) {
                    if (has_core_edge(node, excluded_node)) {
                        next_excluded.push_back(excluded_node);
                    }
                }
                expand(current, next_candidates, next_excluded);
                current.pop_back();

                iter = std::find(candidates.begin(), candidates.end(), node);
                if (iter != candidates.end()) {
                    candidates.erase(iter);
                    excluded.push_back(node);
                }
                if (remaining_must == 0 or saved_cliques.size() >= max_saved_cliques) {
                    return;
                }
            }
        };

    for (uint64_t root = 0; root < core_count; ++root) {
        if (remaining_must == 0 or saved_cliques.size() >= max_saved_cliques) {
            break;
        }
        Vector<uint64_t> current(allocator);
        Vector<uint64_t> candidates(allocator);
        Vector<uint64_t> excluded(allocator);
        current.push_back(root);
        for (uint64_t node = 0; node < core_count; ++node) {
            if (not has_core_edge(root, node)) {
                continue;
            }
            if (node > root) {
                candidates.push_back(node);
            } else {
                excluded.push_back(node);
            }
        }
        if (current.size() + candidates.size() < threshold) {
            continue;
        }
        if (not state_has_must(current, candidates)) {
            continue;
        }
        expand(current, candidates, excluded);
    }
    return saved_cliques;
}

FilterPtr
make_inner_id_filter(const FilterPtr& filter, const LabelTable& label_table) {
    if (filter == nullptr) {
        return nullptr;
    }
    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(std::make_shared<InnerIdWrapperFilter>(filter, label_table));
    if (combined_filter->IsEmpty()) {
        return nullptr;
    }
    return combined_filter;
}

FlattenInterfacePtr
make_temporary_sq8_flatten(MetricType metric,
                           DataTypes data_type,
                           int64_t dim,
                           int64_t extra_info_size,
                           const std::shared_ptr<SafeThreadPool>& thread_pool,
                           Allocator* allocator) {
    auto sq8_param = std::make_shared<FlattenDataCellParameter>();
    sq8_param->quantizer_parameter = std::make_shared<ScalarQuantizerParameter<8>>();
    sq8_param->io_parameter = std::make_shared<MemoryIOParameter>();

    IndexCommonParam common_param;
    common_param.metric_ = metric;
    common_param.data_type_ = data_type;
    common_param.dim_ = dim;
    common_param.extra_info_size_ = extra_info_size;
    common_param.thread_pool_ = thread_pool;
    common_param.allocator_ = std::shared_ptr<Allocator>(allocator, [](Allocator*) {});
    return FlattenInterface::MakeInstance(sq8_param, common_param);
}

bool
need_temporary_sq8_build_data(const FlattenInterfacePtr& base_codes, bool has_precise_reorder) {
    return not has_precise_reorder and
           base_codes->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_RABITQ;
}

bool
is_connected(const Vector<Vector<InnerIdType>>& graph, InnerIdType lhs, InnerIdType rhs) {
    const auto& neighbors = graph[lhs];
    return std::binary_search(neighbors.begin(), neighbors.end(), rhs);
}

Vector<InnerIdType>
collect_valid_inner_ids(const FilterPtr& filter,
                        const LabelTable& label_table,
                        uint64_t seed_count,
                        Allocator* allocator) {
    Vector<InnerIdType> inner_ids(allocator);
    if (filter == nullptr or seed_count == 0) {
        return inner_ids;
    }

    const int64_t* valid_labels = nullptr;
    int64_t valid_count = 0;
    filter->GetValidIds(&valid_labels, valid_count);
    if (valid_labels == nullptr or valid_count <= 0) {
        return inner_ids;
    }

    const auto sampled_count = std::min<uint64_t>(seed_count, static_cast<uint64_t>(valid_count));
    inner_ids.reserve(sampled_count);
    for (uint64_t i = 0; i < sampled_count; ++i) {
        const auto offset = i * static_cast<uint64_t>(valid_count) / sampled_count;
        auto [found, inner_id] = label_table.TryGetIdByLabel(valid_labels[offset]);
        if (found) {
            inner_ids.push_back(inner_id);
        }
    }
    std::sort(inner_ids.begin(), inner_ids.end());
    inner_ids.erase(std::unique(inner_ids.begin(), inner_ids.end()), inner_ids.end());
    return inner_ids;
}

struct mci_search_candidate {
    float distance{0.0F};
    InnerIdType inner_id{0};
    bool expanded{false};
};

void
write_nested_vector(StreamWriter& writer, const Vector<Vector<InnerIdType>>& rows) {
    const auto row_count = static_cast<uint64_t>(rows.size());
    StreamWriter::WriteObj(writer, row_count);
    for (const auto& row : rows) {
        StreamWriter::WriteVector(writer, row);
    }
}

void
read_nested_vector(StreamReader& reader, Vector<Vector<InnerIdType>>& rows, Allocator* allocator) {
    uint64_t row_count = 0;
    StreamReader::ReadObj(reader, row_count);
    rows.clear();
    rows.reserve(row_count);
    for (uint64_t i = 0; i < row_count; ++i) {
        rows.emplace_back(allocator);
        StreamReader::ReadVector(reader, rows.back());
    }
}

uint64_t
nested_vector_memory(const Vector<Vector<InnerIdType>>& rows) {
    uint64_t memory = rows.capacity() * sizeof(Vector<InnerIdType>);
    for (const auto& row : rows) {
        memory += row.capacity() * sizeof(InnerIdType);
    }
    return memory;
}

}  // namespace

MCI::MCI(const MCIParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      p_maxc_(common_param.allocator_.get()),
      maxcs_(common_param.allocator_.get()),
      p_node_to_cid_(common_param.allocator_.get()),
      node_to_cids_(common_param.allocator_.get()),
      delta_cliques_(common_param.allocator_.get()),
      delta_clique_extra_(common_param.allocator_.get()),
      delta_node_to_cids_(common_param.allocator_.get()),
      max_degree_(param->max_degree),
      mcs_(param->mcs),
      clique_max_(param->clique_max),
      alpha_(param->alpha),
      join_ratio_threshold_(param->join_ratio_threshold),
      added_mct_(param->added_mct),
      knng_path_(param->knng_path),
      clique_path_(param->clique_path),
      reorder_by_base_(param->reorder_source == HGRAPH_REORDER_SOURCE_BASE),
      use_hgraph_hybrid_(param->use_hgraph_hybrid),
      hgraph_valid_ratio_threshold_(param->hgraph_valid_ratio_threshold),
      hgraph_index_path_(param->hgraph_index_path),
      hgraph_ef_search_(param->hgraph_ef_search) {
    this->base_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    if (this->use_reorder_ and not this->reorder_by_base_) {
        this->reorder_codes_ =
            FlattenInterface::MakeInstance(param->precise_codes_param, common_param);
    }
    if (this->use_hgraph_hybrid_) {
        CHECK_ARGUMENT(param->hgraph_param != nullptr,
                       "mci hgraph hybrid requires hgraph_index_param");
        this->hgraph_index_ = std::make_shared<HGraph>(param->hgraph_param, common_param);
    }
    this->p_maxc_.push_back(0);
    this->p_node_to_cid_.push_back(0);
}

std::vector<int64_t>
MCI::Build(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    if (this->hgraph_index_ != nullptr) {
        if (not this->hgraph_index_path_.empty()) {
            this->load_hgraph_index(this->hgraph_index_path_);
        } else {
            auto hgraph_failed_ids = this->hgraph_index_->Build(data);
            CHECK_ARGUMENT(hgraph_failed_ids.empty(),
                           "mci hgraph hybrid sub-index failed to build all vectors");
        }
    }
    this->Train(data);
    Vector<std::pair<InnerIdType, int64_t>> inserted_ids(this->allocator_);
    auto failed_ids = this->add_dataset(data, false, &inserted_ids);
    if (not inserted_ids.empty()) {
        this->build_clique_index(
            this->get_float_vectors(data), data->GetNumElements(), inserted_ids);
    }
    if (this->hgraph_index_ != nullptr) {
        CHECK_ARGUMENT(this->hgraph_index_->GetNumElements() == this->GetNumElements(),
                       "mci hgraph hybrid sub-index size mismatch after build");
    }
    this->cal_memory_usage();
    return failed_ids;
}

void
MCI::Train(const DatasetPtr& data) {
    auto total = data->GetNumElements();
    if (total == 0) {
        return;
    }
    const auto* vectors = this->get_float_vectors(data);
    this->base_codes_->Train(vectors, total);
    if (this->reorder_codes_ != nullptr) {
        this->reorder_codes_->Train(vectors, total);
    }
}

std::vector<int64_t>
MCI::Add(const DatasetPtr& data, AddMode mode) {
    CHECK_ARGUMENT(data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   fmt::format("MCI currently supports only {} datatype", DATATYPE_FLOAT32));
    const auto* vectors = data->GetFloat32Vectors();
    CHECK_ARGUMENT(vectors != nullptr, "float vectors are nullptr");
    const auto* labels = data->GetIds();
    CHECK_ARGUMENT(labels != nullptr, "base.ids is nullptr");
    auto base_dim = data->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));

    const auto total_new = data->GetNumElements();
    bool hgraph_added_batch = false;
    auto add_hgraph_vector = [&](int64_t local_id, const float* vector) {
        if (this->hgraph_index_ == nullptr or hgraph_added_batch) {
            return;
        }
        auto hgraph_data = Dataset::Make();
        hgraph_data->NumElements(1)
            ->Dim(this->dim_)
            ->Ids(labels + local_id)
            ->Float32Vectors(vector)
            ->Owner(false);
        auto hgraph_failed_ids = this->hgraph_index_->Add(hgraph_data, mode);
        CHECK_ARGUMENT(hgraph_failed_ids.empty(),
                       "mci hgraph hybrid sub-index failed to add vector");
        CHECK_ARGUMENT(this->hgraph_index_->GetNumElements() == this->GetNumElements(),
                       "mci hgraph hybrid sub-index size mismatch after add");
    };

    std::unique_lock<std::shared_mutex> add_lock(this->add_mutex_);
    if (this->total_count_.load() == 0) {
        this->Train(data);
    }
    const auto start_total = this->total_count_.load();
    const auto had_clique = this->has_clique_index(start_total);

    if (this->hgraph_index_ != nullptr) {
        bool can_add_hgraph_batch = true;
        Vector<int64_t> batch_labels(this->allocator_);
        batch_labels.reserve(static_cast<uint64_t>(total_new));
        {
            std::lock_guard<std::shared_mutex> label_lock(this->label_lookup_mutex_);
            for (int64_t local_id = 0; local_id < total_new; ++local_id) {
                const auto label = labels[local_id];
                if (this->label_table_->CheckLabel(label)) {
                    can_add_hgraph_batch = false;
                    break;
                }
                batch_labels.push_back(label);
            }
        }
        if (can_add_hgraph_batch) {
            std::sort(batch_labels.begin(), batch_labels.end());
            can_add_hgraph_batch =
                std::adjacent_find(batch_labels.begin(), batch_labels.end()) == batch_labels.end();
        }
        if (can_add_hgraph_batch) {
            auto hgraph_failed_ids = this->hgraph_index_->Add(data, mode);
            CHECK_ARGUMENT(hgraph_failed_ids.empty(),
                           "mci hgraph hybrid sub-index failed to add all vectors");
            hgraph_added_batch = true;
        }
    }

    std::vector<int64_t> failed_ids;
    for (int64_t local_id = 0; local_id < total_new; ++local_id) {
        const auto label = labels[local_id];
        {
            std::lock_guard<std::shared_mutex> label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
        }

        const auto inner_id = static_cast<InnerIdType>(this->total_count_.load());
        this->resize(inner_id + 1);
        {
            std::lock_guard<std::shared_mutex> label_lock(this->label_lookup_mutex_);
            this->label_table_->Insert(inner_id, label);
        }
        const auto* vector = vectors + local_id * dim_;
        this->base_codes_->InsertVector(vector, inner_id);
        if (this->reorder_codes_ != nullptr) {
            this->reorder_codes_->InsertVector(vector, inner_id);
        }
        this->total_count_.fetch_add(1);

        // Extend p_node_to_cid_ for the new node (initially no clique membership).
        {
            std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
            this->p_node_to_cid_.push_back(this->p_node_to_cid_.back());
            this->delta_node_to_cids_.emplace_back(this->allocator_);
        }
        add_hgraph_vector(local_id, vector);

        // Incrementally update the clique index if one existed before Add.
        if (had_clique) {
            this->incremental_update_clique(inner_id, vector);
        }
    }
    if (hgraph_added_batch) {
        CHECK_ARGUMENT(this->hgraph_index_->GetNumElements() == this->GetNumElements(),
                       "mci hgraph hybrid sub-index size mismatch after batch add");
    }

    // If there was no clique index before, we still don't have one —
    // the caller should use Build() on the first batch.
    this->cal_memory_usage();
    return failed_ids;
}

std::vector<int64_t>
MCI::add_dataset(const DatasetPtr& data,
                 bool train_if_empty,
                 Vector<std::pair<InnerIdType, int64_t>>* inserted_ids) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    const auto* vectors = this->get_float_vectors(data);
    const auto* labels = data->GetIds();
    CHECK_ARGUMENT(labels != nullptr, "base.ids is nullptr");

    std::unique_lock<std::shared_mutex> add_lock(this->add_mutex_);
    if (train_if_empty and this->total_count_.load() == 0) {
        this->Train(data);
    }

    const auto total = data->GetNumElements();
    for (int64_t local_id = 0; local_id < total; ++local_id) {
        const auto label = labels[local_id];
        {
            std::lock_guard<std::shared_mutex> label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
        }

        const auto inner_id = static_cast<InnerIdType>(this->total_count_.load());
        this->resize(inner_id + 1);
        {
            std::lock_guard<std::shared_mutex> label_lock(this->label_lookup_mutex_);
            this->label_table_->Insert(inner_id, label);
        }
        const auto* vector = vectors + local_id * dim_;
        this->base_codes_->InsertVector(vector, inner_id);
        if (this->reorder_codes_ != nullptr) {
            this->reorder_codes_->InsertVector(vector, inner_id);
        }
        if (inserted_ids != nullptr) {
            inserted_ids->emplace_back(inner_id, local_id);
        }
        this->total_count_.fetch_add(1);
        {
            std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
            this->p_node_to_cid_.push_back(this->p_node_to_cid_.back());
        }
    }
    return failed_ids;
}

void
MCI::clear_clique_index() {
    std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
    this->p_maxc_.clear();
    this->maxcs_.clear();
    this->p_node_to_cid_.clear();
    this->node_to_cids_.clear();
    this->p_maxc_.push_back(0);
    this->p_node_to_cid_.assign(this->total_count_.load() + 1, 0);
    this->total_clique_count_ = 0;
    this->reset_delta_clique_index(this->total_count_.load());
}

void
MCI::build_clique_index(const float* vectors,
                        uint64_t data_count,
                        const Vector<std::pair<InnerIdType, int64_t>>& inserted_ids) {
    const auto total = this->total_count_.load();
    if (total == 0) {
        this->clear_clique_index();
        return;
    }
    if (not this->clique_path_.empty()) {
        this->load_clique_index(this->clique_path_, total);
        return;
    }

    auto has_precise_reorder = use_reorder_ and not reorder_by_base_ and this->reorder_codes_;
    auto build_codes = has_precise_reorder ? this->reorder_codes_ : this->base_codes_;
    FlattenInterfacePtr temporary_sq8_build_data = nullptr;
    if (need_temporary_sq8_build_data(this->base_codes_, has_precise_reorder)) {
        temporary_sq8_build_data =
            make_temporary_sq8_flatten(this->metric_,
                                       this->data_type_,
                                       this->dim_,
                                       static_cast<int64_t>(this->extra_info_size_),
                                       this->thread_pool_,
                                       this->allocator_);
        temporary_sq8_build_data->Train(vectors, data_count);
        for (const auto& [inner_id, local_idx] : inserted_ids) {
            temporary_sq8_build_data->InsertVector(vectors + dim_ * local_idx, inner_id);
        }
        build_codes = temporary_sq8_build_data;
    }

    auto graph = this->build_knn_graph(build_codes, total);
    auto cliques = this->enumerate_maximal_cliques(graph, build_codes, total);

    Vector<InnerIdType> new_p_maxc(this->allocator_);
    Vector<InnerIdType> new_maxcs(this->allocator_);
    Vector<Vector<InnerIdType>> node_to_clique(
        total, Vector<InnerIdType>(this->allocator_), this->allocator_);
    new_p_maxc.push_back(0);
    for (InnerIdType clique_id = 0; clique_id < cliques.size(); ++clique_id) {
        for (auto inner_id : cliques[clique_id]) {
            new_maxcs.push_back(inner_id);
            node_to_clique[inner_id].push_back(clique_id);
        }
        new_p_maxc.push_back(static_cast<InnerIdType>(new_maxcs.size()));
    }

    Vector<InnerIdType> new_p_node_to_cid(this->allocator_);
    Vector<InnerIdType> new_node_to_cids(this->allocator_);
    new_p_node_to_cid.push_back(0);
    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        auto& ids = node_to_clique[inner_id];
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        new_node_to_cids.insert(new_node_to_cids.end(), ids.begin(), ids.end());
        new_p_node_to_cid.push_back(static_cast<InnerIdType>(new_node_to_cids.size()));
    }

    std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
    this->p_maxc_.swap(new_p_maxc);
    this->maxcs_.swap(new_maxcs);
    this->p_node_to_cid_.swap(new_p_node_to_cid);
    this->node_to_cids_.swap(new_node_to_cids);
    this->total_clique_count_ = cliques.size();
    this->reset_delta_clique_index(total);
}

void
MCI::load_clique_index(const std::string& clique_path, uint64_t total) {
    auto make_file_path = [](const std::string& dir, const std::string& name) {
        if (dir.empty() or dir.back() == '/') {
            return dir + name;
        }
        return dir + "/" + name;
    };
    auto read_vector = [&](const std::string& name) {
        const auto file_path = make_file_path(clique_path, name);
        std::ifstream input(file_path, std::ios::binary | std::ios::ate);
        CHECK_ARGUMENT(input.good(), fmt::format("failed to open mci clique file: {}", file_path));
        const auto file_size = static_cast<uint64_t>(input.tellg());
        CHECK_ARGUMENT(file_size % sizeof(InnerIdType) == 0,
                       fmt::format("invalid mci clique file size: {}", file_path));
        Vector<InnerIdType> values(file_size / sizeof(InnerIdType), this->allocator_);
        input.seekg(0, std::ios::beg);
        if (not values.empty()) {
            input.read(reinterpret_cast<char*>(values.data()),
                       static_cast<std::streamsize>(file_size));
            CHECK_ARGUMENT(input.good(),
                           fmt::format("failed to read mci clique file: {}", file_path));
        }
        return values;
    };

    auto new_p_maxc = read_vector("pMaxC");
    auto new_maxcs = read_vector("maxCs");
    auto new_p_node_to_cid = read_vector("pNodeToCid");
    auto new_node_to_cids = read_vector("nodeToCids");

    CHECK_ARGUMENT(not new_p_maxc.empty(), "mci pMaxC must not be empty");
    CHECK_ARGUMENT(
        new_p_node_to_cid.size() == total + 1,
        fmt::format(
            "mci pNodeToCid size {} must be total + 1 ({})", new_p_node_to_cid.size(), total + 1));
    CHECK_ARGUMENT(new_p_maxc.front() == 0 and  // NOLINT
                       new_p_node_to_cid.front() == 0,
                   "mci imported CSR offsets must start from 0");
    CHECK_ARGUMENT(
        new_p_maxc.back() == new_maxcs.size(),
        fmt::format(
            "mci pMaxC tail {} must equal maxCs size {}", new_p_maxc.back(), new_maxcs.size()));
    CHECK_ARGUMENT(new_p_node_to_cid.back() == new_node_to_cids.size(),
                   fmt::format("mci pNodeToCid tail {} must equal nodeToCids size {}",
                               new_p_node_to_cid.back(),
                               new_node_to_cids.size()));

    auto offsets_non_decreasing = [](const Vector<InnerIdType>& offsets) {
        return std::is_sorted(offsets.begin(), offsets.end());
    };
    CHECK_ARGUMENT(offsets_non_decreasing(new_p_maxc), "mci pMaxC offsets must be sorted");
    CHECK_ARGUMENT(offsets_non_decreasing(new_p_node_to_cid),
                   "mci pNodeToCid offsets must be sorted");
    if (not new_maxcs.empty()) {
        const auto max_node = *std::max_element(new_maxcs.begin(), new_maxcs.end());
        CHECK_ARGUMENT(max_node < total,
                       fmt::format("mci maxCs node {} is out of range {}", max_node, total));
    }
    if (not new_node_to_cids.empty()) {
        const auto clique_count = static_cast<InnerIdType>(new_p_maxc.size() - 1);
        const auto max_cid = *std::max_element(new_node_to_cids.begin(), new_node_to_cids.end());
        CHECK_ARGUMENT(
            max_cid < clique_count,
            fmt::format("mci nodeToCids id {} is out of range {}", max_cid, clique_count));
    }

    std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
    this->p_maxc_.swap(new_p_maxc);
    this->maxcs_.swap(new_maxcs);
    this->p_node_to_cid_.swap(new_p_node_to_cid);
    this->node_to_cids_.swap(new_node_to_cids);
    this->total_clique_count_ = this->p_maxc_.size() - 1;
    this->reset_delta_clique_index(total);
}

void
MCI::validate_clique_csr(uint64_t total) const {
    if (this->p_maxc_.empty() and this->p_node_to_cid_.empty()) {
        return;
    }
    CHECK_ARGUMENT(not this->p_maxc_.empty(), "mci pMaxC must not be empty after deserialize");
    CHECK_ARGUMENT(this->p_node_to_cid_.size() == total + 1,
                   fmt::format("mci pNodeToCid size {} must be total + 1 ({})",
                               this->p_node_to_cid_.size(),
                               total + 1));
    const auto offsets_start_at_zero =
        this->p_maxc_.front() == 0 and this->p_node_to_cid_.front() == 0;
    CHECK_ARGUMENT(offsets_start_at_zero, "mci deserialized CSR offsets must start from 0");
    CHECK_ARGUMENT(this->p_maxc_.back() == this->maxcs_.size(),
                   fmt::format("mci pMaxC tail {} must equal maxCs size {}",
                               this->p_maxc_.back(),
                               this->maxcs_.size()));
    CHECK_ARGUMENT(this->p_node_to_cid_.back() == this->node_to_cids_.size(),
                   fmt::format("mci pNodeToCid tail {} must equal nodeToCids size {}",
                               this->p_node_to_cid_.back(),
                               this->node_to_cids_.size()));
    CHECK_ARGUMENT(
        this->p_maxc_.size() == this->total_clique_count_ + 1,
        fmt::format("mci pMaxC size {} inconsistent with total_clique_count {} from metadata",
                    this->p_maxc_.size(),
                    this->total_clique_count_));
    CHECK_ARGUMENT(std::is_sorted(this->p_maxc_.begin(), this->p_maxc_.end()),
                   "mci pMaxC offsets must be sorted after deserialize");
    CHECK_ARGUMENT(std::is_sorted(this->p_node_to_cid_.begin(), this->p_node_to_cid_.end()),
                   "mci pNodeToCid offsets must be sorted after deserialize");
    if (not this->maxcs_.empty()) {
        const auto max_node = *std::max_element(this->maxcs_.begin(), this->maxcs_.end());
        CHECK_ARGUMENT(max_node < total,
                       fmt::format("mci maxCs node {} is out of range {}", max_node, total));
    }
    if (not this->node_to_cids_.empty()) {
        const auto clique_count = static_cast<InnerIdType>(this->p_maxc_.size() - 1);
        const auto max_cid =
            *std::max_element(this->node_to_cids_.begin(), this->node_to_cids_.end());
        CHECK_ARGUMENT(
            max_cid < clique_count,
            fmt::format("mci nodeToCids id {} is out of range {}", max_cid, clique_count));
    }
}

Vector<Vector<InnerIdType>>
MCI::build_knn_graph(const FlattenInterfacePtr& build_codes, uint64_t total) const {
    Vector<Vector<InnerIdType>> graph(
        total, Vector<InnerIdType>(this->allocator_), this->allocator_);
    if (total <= 1) {
        return graph;
    }

    const auto candidate_limit = std::min<uint64_t>(this->mcs_, total - 1);
    if (not this->knng_path_.empty()) {
        std::ifstream input(this->knng_path_, std::ios::binary | std::ios::ate);
        CHECK_ARGUMENT(input.good(), fmt::format("failed to open knng file: {}", knng_path_));
        const auto file_size = static_cast<uint64_t>(input.tellg());
        CHECK_ARGUMENT(file_size > 0, fmt::format("knng file is empty: {}", knng_path_));
        CHECK_ARGUMENT(file_size % sizeof(InnerIdType) == 0,
                       fmt::format("invalid knng file size: {}", knng_path_));
        const auto entry_count = file_size / sizeof(InnerIdType);
        CHECK_ARGUMENT(
            entry_count % total == 0,
            fmt::format("knng entries are not divisible by total count: {}", knng_path_));
        const auto file_degree = entry_count / total;
        CHECK_ARGUMENT(file_degree > 0, fmt::format("knng degree is zero: {}", knng_path_));

        input.seekg(0, std::ios::beg);
        Vector<InnerIdType> row(file_degree, this->allocator_);
        Vector<uint8_t> seen(total, 0, this->allocator_);
        for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
            input.read(reinterpret_cast<char*>(row.data()),
                       static_cast<std::streamsize>(file_degree * sizeof(InnerIdType)));
            CHECK_ARGUMENT(input.good(), fmt::format("failed to read knng row: {}", knng_path_));
            auto& neighbors = graph[inner_id];
            for (uint64_t rank = 0; rank < file_degree and neighbors.size() < candidate_limit;
                 ++rank) {
                const auto neighbor = row[rank];
                CHECK_ARGUMENT(neighbor < total,
                               fmt::format("knng id {} is out of range {}", neighbor, total));
                if (neighbor == inner_id or seen[neighbor] != 0) {
                    continue;
                }
                seen[neighbor] = 1;
                neighbors.push_back(neighbor);
            }
            for (auto neighbor : neighbors) {
                seen[neighbor] = 0;
            }
        }
        return graph;
    }

    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = candidate_limit;
    graph_param->support_delete_ = false;
    IndexCommonParam graph_common_param;
    graph_common_param.metric_ = this->metric_;
    graph_common_param.data_type_ = this->data_type_;
    graph_common_param.dim_ = this->dim_;
    graph_common_param.extra_info_size_ = static_cast<int64_t>(this->extra_info_size_);
    graph_common_param.thread_pool_ = this->thread_pool_;
    graph_common_param.allocator_ = std::shared_ptr<Allocator>(this->allocator_, [](Allocator*) {});
    auto graph_storage = GraphInterface::MakeInstance(graph_param, graph_common_param);

    auto odescent_param = std::make_shared<ODescentParameter>();
    odescent_param->max_degree = static_cast<int64_t>(candidate_limit);
    odescent_param->alpha = this->alpha_;
    odescent_param->sample_rate = 0.2F;
    odescent_param->turn = 30;
    odescent_param->min_in_degree = 1;
    ODescent odescent_builder(
        odescent_param, build_codes, this->allocator_, this->thread_pool_.get());
    odescent_builder.Build();
    odescent_builder.SaveGraph(graph_storage);

    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        graph_storage->GetNeighbors(inner_id, graph[inner_id]);
        auto& neighbors = graph[inner_id];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    return graph;
}

Vector<Vector<InnerIdType>>
MCI::enumerate_maximal_cliques(const Vector<Vector<InnerIdType>>& graph,
                               const FlattenInterfacePtr& build_codes,
                               uint64_t total) const {
    Vector<Vector<InnerIdType>> cliques(this->allocator_);
    std::vector<std::atomic<uint32_t>> num_cliques_per_node(total);
    for (auto& count : num_cliques_per_node) {
        count.store(0, std::memory_order_relaxed);
    }

    auto get_clique_count = [&](InnerIdType inner_id) {
        return num_cliques_per_node[inner_id].load(std::memory_order_relaxed);
    };

    auto normalize_clique = [&](const Vector<InnerIdType>& clique) {
        Vector<InnerIdType> normalized(this->allocator_);
        if (clique.empty()) {
            return normalized;
        }
        normalized.assign(clique.begin(), clique.end());
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        return normalized;
    };

    auto append_selected_clique = [&](const Vector<InnerIdType>& clique) {
        if (clique.empty()) {
            return;
        }
        cliques.push_back(Vector<InnerIdType>(this->allocator_));
        cliques.back().assign(clique.begin(), clique.end());
    };

    // Best-effort coverage: the uncovered check + fetch_add is not atomic, so parallel
    // workers may both observe a node as uncovered and emit overlapping cliques. This is
    // memory-safe (per-thread buffers, atomic counter) but may inflate clique count slightly.
    auto try_select_clique = [&](const Vector<InnerIdType>& clique,
                                 Vector<Vector<InnerIdType>>& output) {
        auto normalized = normalize_clique(clique);
        if (normalized.empty()) {
            return false;
        }
        bool has_uncovered = false;
        for (auto inner_id : normalized) {
            if (get_clique_count(inner_id) == 0) {
                has_uncovered = true;
                break;
            }
        }
        if (not has_uncovered) {
            return false;
        }
        for (auto inner_id : normalized) {
            num_cliques_per_node[inner_id].fetch_add(1, std::memory_order_relaxed);
        }
        output.push_back(std::move(normalized));
        return true;
    };

    const auto candidate_limit = std::min<uint64_t>(this->mcs_, total - 1);
    const auto clique_min =
        std::max<uint64_t>(2, std::min<uint64_t>({this->clique_max_, candidate_limit + 1, total}));
    const auto node_clique_limit = std::max<uint32_t>(3, static_cast<uint32_t>(total / 100));
    const auto max_saved_per_seed =
        std::min<uint64_t>(candidate_limit, static_cast<uint64_t>(this->max_degree_ + 2));

    auto collect_candidates =
        [&](InnerIdType seed, Vector<InnerIdType>& local_nodes, Vector<float>& seed_distances) {
            local_nodes.clear();
            seed_distances.clear();
            local_nodes.push_back(seed);
            seed_distances.push_back(0.0F);
            for (auto neighbor : graph[seed]) {
                if (neighbor >= total or neighbor == seed or
                    get_clique_count(neighbor) >= node_clique_limit) {
                    continue;
                }
                local_nodes.push_back(neighbor);
                seed_distances.push_back(build_codes->ComputePairVectors(seed, neighbor));
                if (local_nodes.size() > candidate_limit) {
                    break;
                }
            }
            Vector<uint64_t> order(this->allocator_);
            order.reserve(local_nodes.size() - 1);
            for (uint64_t i = 1; i < local_nodes.size(); ++i) {
                order.push_back(i);
            }
            std::sort(order.begin(), order.end(), [&](uint64_t lhs, uint64_t rhs) {
                if (seed_distances[lhs] != seed_distances[rhs]) {
                    return seed_distances[lhs] < seed_distances[rhs];
                }
                return local_nodes[lhs] < local_nodes[rhs];
            });
            Vector<InnerIdType> sorted_nodes(this->allocator_);
            Vector<float> sorted_distances(this->allocator_);
            sorted_nodes.reserve(local_nodes.size());
            sorted_distances.reserve(seed_distances.size());
            sorted_nodes.push_back(seed);
            sorted_distances.push_back(0.0F);
            for (auto idx : order) {
                sorted_nodes.push_back(local_nodes[idx]);
                sorted_distances.push_back(seed_distances[idx]);
            }
            local_nodes.swap(sorted_nodes);
            seed_distances.swap(sorted_distances);
        };

    auto build_local_edges = [&](float now_alpha,
                                 Vector<InnerIdType>& local_nodes,
                                 Vector<float>& seed_distances,
                                 Vector<uint8_t>& local_edges) {
        const auto local_count = local_nodes.size();
        local_edges.assign(local_count * local_count, 0);
        auto set_local_edge = [&](uint64_t lhs, uint64_t rhs) {
            local_edges[lhs * local_count + rhs] = 1;
        };
        uint64_t edge_count = 0;
        if (local_nodes.size() <= 1) {
            return edge_count;
        }
        const auto distance_limit = seed_distances[1] * now_alpha;
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            if (seed_distances[i] <= distance_limit) {
                set_local_edge(0, i);
                set_local_edge(i, 0);
                ++edge_count;
            }
        }
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            for (uint64_t j = i + 1; j < local_nodes.size(); ++j) {
                if (build_codes->ComputePairVectors(local_nodes[i], local_nodes[j]) <=
                    distance_limit) {
                    set_local_edge(i, j);
                    set_local_edge(j, i);
                    ++edge_count;
                }
            }
        }
        return edge_count;
    };

    auto append_high_alpha_fallback = [&](float now_alpha,
                                          const Vector<InnerIdType>& local_nodes,
                                          Vector<Vector<InnerIdType>>& output) {
        if (now_alpha <= 100.0F or local_nodes.empty()) {
            return false;
        }
        Vector<InnerIdType> fallback(this->allocator_);
        fallback.reserve(local_nodes.size());
        fallback.push_back(local_nodes.front());
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            if (get_clique_count(local_nodes[i]) > 0) {
                fallback.push_back(local_nodes[i]);
            }
        }
        return try_select_clique(fallback, output);
    };

    auto build_ccr_saved_group = [&](const Vector<uint64_t>& local_clique,
                                     const Vector<InnerIdType>& local_nodes,
                                     const Vector<uint8_t>& local_edges) {
        Vector<InnerIdType> group(this->allocator_);
        const auto local_count = local_nodes.size();
        if (local_clique.empty() or local_count == 0) {
            return group;
        }
        auto has_edge = [&](uint64_t lhs, uint64_t rhs) {
            return local_edges[lhs * local_count + rhs] != 0;
        };
        const auto root = local_clique.front();
        Vector<uint64_t> forward_neighbors(this->allocator_);
        forward_neighbors.reserve(local_count);
        for (uint64_t local_id = 0; local_id < local_count; ++local_id) {
            if (local_id != root and has_edge(root, local_id)) {
                forward_neighbors.push_back(local_id);
            }
        }
        if (forward_neighbors.size() + 1 < clique_min) {
            return group;
        }

        Vector<uint64_t> degree(local_count, 0, this->allocator_);
        for (uint64_t i = 0; i < forward_neighbors.size(); ++i) {
            const auto lhs = forward_neighbors[i];
            for (uint64_t j = i + 1; j < forward_neighbors.size(); ++j) {
                const auto rhs = forward_neighbors[j];
                if (has_edge(lhs, rhs)) {
                    ++degree[lhs];
                    ++degree[rhs];
                }
            }
        }

        Vector<uint8_t> removed(local_count, 0, this->allocator_);
        Vector<uint64_t> queue(this->allocator_);
        for (auto node : forward_neighbors) {
            if (degree[node] + 2 < clique_min) {
                queue.push_back(node);
            }
        }
        for (uint64_t offset = 0; offset < queue.size(); ++offset) {
            const auto node = queue[offset];
            if (removed[node] != 0) {
                continue;
            }
            removed[node] = 1;
            for (auto neighbor : forward_neighbors) {
                if (removed[neighbor] != 0 or not has_edge(node, neighbor)) {
                    continue;
                }
                if (degree[neighbor] > 0) {
                    --degree[neighbor];
                }
                if (degree[neighbor] + 2 < clique_min) {
                    queue.push_back(neighbor);
                }
            }
        }

        Vector<uint64_t> active(this->allocator_);
        active.reserve(forward_neighbors.size());
        for (auto node : forward_neighbors) {
            if (removed[node] == 0) {
                active.push_back(node);
            }
        }
        if (active.size() + 1 < clique_min) {
            return group;
        }

        Vector<uint64_t> active_degree(local_count, 0, this->allocator_);
        for (uint64_t i = 0; i < active.size(); ++i) {
            const auto lhs = active[i];
            for (uint64_t j = i + 1; j < active.size(); ++j) {
                const auto rhs = active[j];
                if (has_edge(lhs, rhs)) {
                    ++active_degree[lhs];
                    ++active_degree[rhs];
                }
            }
        }

        Vector<uint8_t> peeled(local_count, 0, this->allocator_);
        Vector<uint64_t> peel_order(this->allocator_);
        Vector<uint64_t> core_nodes(this->allocator_);
        Vector<uint64_t> p_nodes(this->allocator_);
        peel_order.reserve(active.size());
        core_nodes.reserve(active.size());
        p_nodes.reserve(active.size());
        for (uint64_t step = 0; step < active.size(); ++step) {
            uint64_t best = local_count;
            uint64_t best_degree = std::numeric_limits<uint64_t>::max();
            for (auto node : active) {
                if (peeled[node] == 0 and (active_degree[node] < best_degree or
                                           (active_degree[node] == best_degree and node < best))) {
                    best = node;
                    best_degree = active_degree[node];
                }
            }
            if (best == local_count) {
                break;
            }
            if (best_degree + step + 1 == active.size()) {
                p_nodes.assign(peel_order.begin(), peel_order.end());
                core_nodes.push_back(best);
                peeled[best] = 1;
                for (auto node : active) {
                    if (peeled[node] == 0) {
                        core_nodes.push_back(node);
                    }
                }
                break;
            }
            peeled[best] = 1;
            peel_order.push_back(best);
            for (auto neighbor : active) {
                if (peeled[neighbor] == 0 and has_edge(best, neighbor) and
                    active_degree[neighbor] > 0) {
                    --active_degree[neighbor];
                }
            }
        }
        if (core_nodes.empty()) {
            core_nodes.assign(active.begin(), active.end());
        }
        if (core_nodes.size() + 1 < clique_min) {
            return group;
        }

        Vector<uint8_t> in_core(local_count, 0, this->allocator_);
        Vector<uint8_t> in_p(local_count, 0, this->allocator_);
        for (auto node : core_nodes) {
            in_core[node] = 1;
        }
        for (auto node : p_nodes) {
            in_p[node] = 1;
        }
        auto count_core_neighbors = [&](uint64_t node) {
            uint64_t count = 0;
            for (auto core_node : core_nodes) {
                if (has_edge(node, core_node)) {
                    ++count;
                }
            }
            return count;
        };

        bool should_save = true;
        for (auto node : p_nodes) {
            if (count_core_neighbors(node) == core_nodes.size()) {
                should_save = false;
                break;
            }
        }
        if (should_save) {
            for (auto node : forward_neighbors) {
                if (node >= root or in_core[node] != 0 or in_p[node] != 0) {
                    continue;
                }
                if (count_core_neighbors(node) == core_nodes.size()) {
                    should_save = false;
                    break;
                }
            }
        }
        if (not should_save) {
            return group;
        }

        group.reserve(1 + core_nodes.size() + p_nodes.size());
        group.push_back(local_nodes[root]);
        for (auto node : core_nodes) {
            group.push_back(local_nodes[node]);
        }
        for (auto node : p_nodes) {
            group.push_back(local_nodes[node]);
        }
        return group;
    };

    auto solve_seed = [&](InnerIdType seed,
                          float now_alpha,
                          Vector<Vector<InnerIdType>>& output,
                          Vector<InnerIdType>& local_nodes,
                          Vector<float>& seed_distances,
                          Vector<uint8_t>& local_edges) {
        collect_candidates(seed, local_nodes, seed_distances);
        if (local_nodes.size() < clique_min) {
            append_high_alpha_fallback(now_alpha, local_nodes, output);
            return;
        }
        const auto edge_count =
            build_local_edges(now_alpha, local_nodes, seed_distances, local_edges);
        if (edge_count < clique_min * (clique_min - 1) / 2) {
            append_high_alpha_fallback(now_alpha, local_nodes, output);
            return;
        }
        auto local_cliques = run_local_ccr_mce(local_nodes,
                                               local_edges,
                                               num_cliques_per_node,
                                               clique_min,
                                               max_saved_per_seed,
                                               this->allocator_);
        if (local_cliques.empty()) {
            append_high_alpha_fallback(now_alpha, local_nodes, output);
            return;
        }

        uint64_t selected = 0;
        for (const auto& local_clique : local_cliques) {
            auto clique = build_ccr_saved_group(local_clique, local_nodes, local_edges);
            if (clique.empty()) {
                clique.reserve(local_clique.size());
                for (auto local_id : local_clique) {
                    clique.push_back(local_nodes[local_id]);
                }
            }
            bool has_uncovered = false;
            for (auto node : clique) {
                if (get_clique_count(node) == 0) {
                    has_uncovered = true;
                    break;
                }
            }
            if (has_uncovered and try_select_clique(clique, output)) {
                if (++selected >= this->max_degree_) {
                    break;
                }
            }
        }
    };

    auto solve_serial_round = [&](float now_alpha) {
        Vector<InnerIdType> local_nodes(this->allocator_);
        Vector<float> seed_distances(this->allocator_);
        Vector<uint8_t> local_edges(this->allocator_);
        Vector<Vector<InnerIdType>> seed_cliques(this->allocator_);
        for (InnerIdType seed = 0; seed < total; ++seed) {
            if (get_clique_count(seed) != 0) {
                continue;
            }
            seed_cliques.clear();
            solve_seed(seed, now_alpha, seed_cliques, local_nodes, seed_distances, local_edges);
            for (const auto& clique : seed_cliques) {
                append_selected_clique(clique);
            }
        }
    };

    auto solve_parallel_round = [&](float now_alpha) {
        if (this->thread_pool_ == nullptr or this->build_thread_count_ <= 1) {
            solve_serial_round(now_alpha);
            return;
        }

        const auto thread_count = this->build_thread_count_;
        const auto batch_seed_limit = std::max<uint64_t>(thread_count, thread_count * 16);
        Vector<InnerIdType> batch_seeds(this->allocator_);
        batch_seeds.reserve(batch_seed_limit);

        std::vector<Vector<Vector<InnerIdType>>> thread_cliques;
        thread_cliques.reserve(thread_count);
        for (uint64_t thread_id = 0; thread_id < thread_count; ++thread_id) {
            thread_cliques.emplace_back(this->allocator_);
        }

        auto worker = [&](uint64_t thread_id,
                          uint64_t begin,
                          uint64_t end,
                          const Vector<InnerIdType>& seeds) {
            Vector<InnerIdType> local_nodes(this->allocator_);
            Vector<float> seed_distances(this->allocator_);
            Vector<uint8_t> local_edges(this->allocator_);
            auto& output = thread_cliques[thread_id];
            for (uint64_t i = begin; i < end; ++i) {
                if (get_clique_count(seeds[i]) != 0) {
                    continue;
                }
                solve_seed(seeds[i], now_alpha, output, local_nodes, seed_distances, local_edges);
            }
        };

        InnerIdType next_seed = 0;
        while (next_seed < total) {
            batch_seeds.clear();
            while (next_seed < total and batch_seeds.size() < batch_seed_limit) {
                if (get_clique_count(next_seed) == 0) {
                    batch_seeds.push_back(next_seed);
                }
                ++next_seed;
            }
            if (batch_seeds.empty()) {
                continue;
            }

            for (auto& one_thread_cliques : thread_cliques) {
                one_thread_cliques.clear();
            }
            const auto active_thread_count = std::min<uint64_t>(thread_count, batch_seeds.size());
            const auto item_per_thread =
                (batch_seeds.size() + active_thread_count - 1) / active_thread_count;
            std::vector<std::future<void>> futures;
            futures.reserve(active_thread_count);
            for (uint64_t thread_id = 0; thread_id < active_thread_count; ++thread_id) {
                const auto begin = thread_id * item_per_thread;
                const auto end = std::min<uint64_t>(begin + item_per_thread, batch_seeds.size());
                futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    worker, thread_id, begin, end, std::cref(batch_seeds)));
            }
            for (auto& future : futures) {
                future.get();
            }
            for (const auto& one_thread_cliques : thread_cliques) {
                for (const auto& clique : one_thread_cliques) {
                    append_selected_clique(clique);
                }
            }
        }
    };

    float now_alpha = std::max(1.2F, this->alpha_);
    uint64_t previous_uncovered = total;
    for (uint64_t round = 0; round < 16; ++round) {
        const auto cliques_before_round = cliques.size();
        solve_parallel_round(now_alpha);

        uint64_t uncovered = 0;
        for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
            if (get_clique_count(inner_id) == 0) {
                ++uncovered;
            }
        }
        logger::debug(
            "mci_build_round round={} alpha={} uncovered={} round_cliques={} total_cliques={}",
            round + 1,
            now_alpha,
            uncovered,
            cliques.size() - cliques_before_round,
            cliques.size());
        if (uncovered == 0) {
            break;
        }
        if (uncovered < previous_uncovered * 9 / 10) {
            now_alpha += std::max(1.2F, this->alpha_);
        } else {
            now_alpha *= 2.0F;
        }
        previous_uncovered = uncovered;
    }

    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        if (get_clique_count(inner_id) == 0) {
            Vector<InnerIdType> singleton(this->allocator_);
            singleton.push_back(inner_id);
            for (auto neighbor : graph[inner_id]) {
                if (singleton.size() >= this->max_degree_ or neighbor >= total) {
                    break;
                }
                singleton.push_back(neighbor);
            }
            Vector<Vector<InnerIdType>> fallback_cliques(this->allocator_);
            if (try_select_clique(singleton, fallback_cliques)) {
                append_selected_clique(fallback_cliques.front());
            }
        }
    }
    {
        uint64_t max_membership = 0;
        uint64_t total_memberships = 0;
        for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
            const auto membership = static_cast<uint64_t>(get_clique_count(inner_id));
            total_memberships += membership;
            max_membership = std::max(max_membership, membership);
        }
        logger::debug(
            "mci_build_summary total_cliques={} total_memberships={} avg_membership={} "
            "max_membership={}",
            cliques.size(),
            total_memberships,
            static_cast<double>(total_memberships) / static_cast<double>(total),
            max_membership);
    }
    return cliques;
}

bool
MCI::has_clique_index(uint64_t total) const {
    return this->total_logical_clique_count() > 0 and
           this->p_maxc_.size() == this->total_clique_count_ + 1 and
           this->p_node_to_cid_.size() == total + 1 and this->delta_node_to_cids_.size() == total;
}

uint64_t
MCI::total_logical_clique_count() const {
    return this->total_clique_count_ + this->delta_cliques_.size();
}

void
MCI::reset_delta_clique_index(uint64_t total) {
    this->delta_cliques_.clear();
    this->delta_clique_extra_.clear();
    this->delta_clique_extra_.resize(this->total_clique_count_,
                                     Vector<InnerIdType>(this->allocator_));
    this->delta_node_to_cids_.clear();
    this->delta_node_to_cids_.reserve(total);
    for (uint64_t i = 0; i < total; ++i) {
        this->delta_node_to_cids_.emplace_back(this->allocator_);
    }
}

void
MCI::ensure_delta_node_rows(uint64_t total) {
    while (this->delta_node_to_cids_.size() < total) {
        this->delta_node_to_cids_.emplace_back(this->allocator_);
    }
    if (this->delta_clique_extra_.size() < this->total_clique_count_) {
        this->delta_clique_extra_.resize(this->total_clique_count_,
                                         Vector<InnerIdType>(this->allocator_));
    }
}

DistHeapPtr
MCI::scan_knn_candidates(const FlattenInterfacePtr& codes,
                         const ComputerInterfacePtr& computer,
                         const FilterPtr& inner_filter,
                         int64_t candidate_limit,
                         bool use_distance_lower_bound,
                         QueryContext& ctx,
                         DistanceRecordVector* rabitq_lower_bound_candidates,
                         uint32_t& dist_cmp) const {
    auto heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, candidate_limit);
    const auto total = static_cast<InnerIdType>(this->total_count_.load());
    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        if (inner_filter != nullptr and not inner_filter->CheckValid(inner_id)) {
            continue;
        }
        float dist = 0.0F;
        float lower_bound = std::numeric_limits<float>::max();
        if (use_distance_lower_bound) {
            codes->QueryWithDistanceLowerBound(&dist, &lower_bound, computer, &inner_id, 1, &ctx);
            if (rabitq_lower_bound_candidates != nullptr) {
                rabitq_lower_bound_candidates->emplace_back(lower_bound, inner_id);
            }
        } else {
            codes->Query(&dist, computer, &inner_id, 1, &ctx);
        }
        ++dist_cmp;
        heap->Push(dist, inner_id);
    }
    return heap;
}

DistHeapPtr
MCI::search_clique_candidates(const ComputerInterfacePtr& computer,
                              const FilterPtr& inner_filter,
                              const Vector<InnerIdType>* seed_inner_ids,
                              const MCISearchParameters& search_params,
                              int64_t candidate_limit,
                              QueryContext& ctx,
                              DistanceRecordVector* rabitq_lower_bound_candidates,
                              uint32_t& dist_cmp,
                              uint32_t& hops) const {
    auto heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, candidate_limit);
    const auto total = static_cast<InnerIdType>(this->total_count_.load());
    Vector<uint8_t> visited_nodes(total, 0, this->allocator_);
    Vector<uint8_t> visited_cliques(this->total_logical_clique_count(), 0, this->allocator_);
    Vector<mci_search_candidate> candidates(this->allocator_);
    candidates.reserve(static_cast<uint64_t>(candidate_limit));

    auto is_better = [](const mci_search_candidate& lhs, const mci_search_candidate& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.inner_id < rhs.inner_id;
    };

    auto can_update = [&](float distance) {
        return static_cast<int64_t>(candidates.size()) < candidate_limit or
               distance < candidates.back().distance;
    };

    auto insert_candidate = [&](float distance, InnerIdType inner_id) {
        if (not can_update(distance)) {
            return;
        }
        mci_search_candidate candidate{distance, inner_id, false};
        auto iter = std::lower_bound(candidates.begin(), candidates.end(), candidate, is_better);
        candidates.insert(iter, candidate);
        if (static_cast<int64_t>(candidates.size()) > candidate_limit) {
            candidates.pop_back();
        }
    };

    auto get_closest_unexpanded = [&]() -> mci_search_candidate* {
        for (auto& candidate : candidates) {
            if (not candidate.expanded) {
                candidate.expanded = true;
                return &candidate;
            }
        }
        return nullptr;
    };

    auto try_visit = [&](InnerIdType inner_id) -> bool {
        if (inner_id >= total or visited_nodes[inner_id] != 0) {
            return false;
        }
        visited_nodes[inner_id] = 1;
        if (inner_filter != nullptr and not inner_filter->CheckValid(inner_id)) {
            return false;
        }
        float dist = 0.0F;
        float lower_bound = std::numeric_limits<float>::max();
        if (search_params.rabitq_one_bit_search) {
            this->base_codes_->QueryWithDistanceLowerBound(
                &dist, &lower_bound, computer, &inner_id, 1, &ctx);
            if (rabitq_lower_bound_candidates != nullptr) {
                rabitq_lower_bound_candidates->emplace_back(lower_bound, inner_id);
            }
        } else {
            this->base_codes_->Query(&dist, computer, &inner_id, 1, &ctx);
        }
        ++dist_cmp;
        insert_candidate(dist, inner_id);
        return true;
    };

    const auto seed_target = std::min<uint64_t>(search_params.seed_count, total);
    uint64_t seeds = 0;
    if (seed_inner_ids != nullptr and not seed_inner_ids->empty()) {
        const auto seed_count = seed_inner_ids->size();
        const auto sampled_seed_count = std::min<uint64_t>(seed_target, seed_count);
        for (uint64_t i = 0; i < sampled_seed_count; ++i) {
            const auto offset = i * seed_count / sampled_seed_count;
            if (try_visit((*seed_inner_ids)[offset])) {
                ++seeds;
            }
        }
    }
    for (InnerIdType seed = 0; seed < total and seeds < seed_target; ++seed) {
        if (try_visit(seed)) {
            ++seeds;
        }
    }

    const auto hop_limit = std::min<uint32_t>(
        search_params.hops_limit, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    auto visit_clique = [&](InnerIdType clique_id) {
        if (clique_id >= visited_cliques.size() or visited_cliques[clique_id] != 0) {
            return;
        }
        visited_cliques[clique_id] = 1;
        ++hops;
        if (clique_id < this->total_clique_count_) {
            const auto node_begin = this->p_maxc_[clique_id];
            const auto node_end = this->p_maxc_[clique_id + 1];
            for (auto node_offset = node_begin; node_offset < node_end; ++node_offset) {
                try_visit(this->maxcs_[node_offset]);
            }
            if (clique_id < this->delta_clique_extra_.size()) {
                for (auto inner_id : this->delta_clique_extra_[clique_id]) {
                    try_visit(inner_id);
                }
            }
        } else {
            const auto delta_id = clique_id - this->total_clique_count_;
            if (delta_id < this->delta_cliques_.size()) {
                for (auto inner_id : this->delta_cliques_[delta_id]) {
                    try_visit(inner_id);
                }
            }
        }
    };

    while (hops < hop_limit) {
        auto* current = get_closest_unexpanded();
        if (current == nullptr) {
            break;
        }
        const auto inner_id = current->inner_id;
        Vector<InnerIdType> clique_ids(this->allocator_);
        this->collect_node_clique_ids(inner_id, clique_ids);
        for (auto clique_id : clique_ids) {
            visit_clique(clique_id);
            if (hops >= hop_limit) {
                break;
            }
        }
    }

    for (const auto& candidate : candidates) {
        heap->Push(candidate.distance, candidate.inner_id);
    }
    return heap;
}

DatasetPtr
MCI::KnnSearch(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const FilterPtr& filter) const {
    SearchRequest request;
    request.query_ = query;
    request.topk_ = k;
    request.params_str_ = parameters;
    if (filter != nullptr) {
        request.enable_filter_ = true;
        request.filter_ = filter;
    }
    return this->SearchWithRequest(request);
}

DatasetPtr
MCI::RangeSearch(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const FilterPtr& filter,
                 int64_t limited_size) const {
    SearchRequest request;
    request.query_ = query;
    request.mode_ = SearchMode::RANGE_SEARCH;
    request.radius_ = radius;
    request.limited_size_ = limited_size;
    request.params_str_ = parameters;
    if (filter != nullptr) {
        request.enable_filter_ = true;
        request.filter_ = filter;
    }
    return this->SearchWithRequest(request);
}

bool
MCI::should_use_hgraph_hybrid(const SearchRequest& request, float valid_ratio) const {
    if (not this->use_hgraph_hybrid_ or this->hgraph_index_ == nullptr) {
        return false;
    }
    if (this->hgraph_index_->GetNumElements() != this->GetNumElements()) {
        logger::debug("mci hybrid routing disabled: hgraph size {} != mci size {}",
                      this->hgraph_index_->GetNumElements(),
                      this->GetNumElements());
        return false;
    }
    if (request.enable_bitset_filter_ and request.bitset_filter_ != nullptr) {
        return false;
    }
    return valid_ratio >= this->hgraph_valid_ratio_threshold_;
}

std::string
MCI::get_hgraph_search_params(const std::string& request_params) const {
    if (not request_params.empty()) {
        auto params = JsonType::Parse(request_params);
        if (params.Contains(INDEX_TYPE_HGRAPH)) {
            return request_params;
        }
    }
    return fmt::format(R"({{"hgraph":{{"ef_search":{}}}}})", this->hgraph_ef_search_);
}

DatasetPtr
MCI::search_hgraph_hybrid(const SearchRequest& request, float valid_ratio) const {
    SearchRequest hgraph_request = request;
    hgraph_request.params_str_ = this->get_hgraph_search_params(request.params_str_);
    auto result = this->hgraph_index_->SearchWithRequest(hgraph_request);

    JsonType stats;
    auto stats_str = result->GetStatistics();
    if (not stats_str.empty()) {
        stats = JsonType::Parse(stats_str);
    }
    stats["mci_hybrid_route"].SetString("hgraph");
    stats["mci_hybrid_valid_ratio"].SetFloat(valid_ratio);
    stats["mci_hybrid_threshold"].SetFloat(this->hgraph_valid_ratio_threshold_);
    result->Statistics(stats.Dump());
    return result;
}

DatasetPtr
MCI::build_dataset_from_heap(DistHeapPtr& heap) const {
    if (heap->Empty()) {
        auto [dataset_results, dists, ids] = create_fast_dataset(0, allocator_);
        return dataset_results;
    }
    auto [dataset_results, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_);
    for (auto i = static_cast<int64_t>(heap->Size() - 1); i >= 0; --i) {
        dists[i] = heap->Top().first;
        ids[i] = this->label_table_->GetLabelById(heap->Top().second);
        heap->Pop();
    }
    return dataset_results;
}

void
MCI::collect_node_clique_ids(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const {
    const auto logical_count = this->total_logical_clique_count();
    if (node_id + 1 < this->p_node_to_cid_.size()) {
        const auto begin = this->p_node_to_cid_[node_id];
        const auto end = this->p_node_to_cid_[node_id + 1];
        for (auto offset = begin; offset < end; ++offset) {
            const auto cid = this->node_to_cids_[offset];
            if (cid < logical_count) {
                clique_ids.push_back(cid);
            }
        }
    }
    if (node_id < this->delta_node_to_cids_.size()) {
        const auto& delta_ids = this->delta_node_to_cids_[node_id];
        for (auto cid : delta_ids) {
            if (cid < logical_count) {
                clique_ids.push_back(cid);
            }
        }
    }
}

Vector<InnerIdType>
MCI::find_knn_for_new_node(InnerIdType new_inner_id, const float* vector) const {
    Vector<InnerIdType> knn_ids(this->allocator_);
    const auto total = this->total_count_.load();
    // The new node has already been inserted, so exclude it from its own neighborhood.
    if (total <= 1) {
        return knn_ids;
    }
    const auto k = std::min<uint64_t>(this->mcs_, total - 1);
    if (k == 0) {
        return knn_ids;
    }

    // Prefer the embedded HGraph sub-index when available; it returns labels,
    // which we round-trip back into MCI inner ids via the shared label table.
    const auto hgraph_count =
        this->hgraph_index_ == nullptr ? 0 : this->hgraph_index_->GetNumElements();
    if (hgraph_count >= static_cast<int64_t>(total)) {
        auto query = Dataset::Make();
        query->NumElements(1)->Dim(this->dim_)->Float32Vectors(vector)->Owner(false);
        const auto hgraph_total = static_cast<uint64_t>(hgraph_count);
        auto query_k = std::min<uint64_t>(k + 1, hgraph_total);
        while (true) {
            auto result = this->hgraph_index_->KnnSearch(
                query, static_cast<int64_t>(query_k), this->get_hgraph_search_params(""), nullptr);
            if (result == nullptr) {
                break;
            }
            knn_ids.clear();
            const auto* ids = result->GetIds();
            const auto count = result->GetDim();
            knn_ids.reserve(static_cast<uint64_t>(count));
            for (int64_t i = 0; i < count; ++i) {
                auto [found, inner_id] = this->label_table_->TryGetIdByLabel(ids[i]);
                if (found and inner_id < total and inner_id != new_inner_id) {
                    knn_ids.push_back(inner_id);
                }
            }
            if (knn_ids.size() >= k or query_k == hgraph_total) {
                break;
            }
            const auto next_query_k = std::min<uint64_t>(query_k * 2, hgraph_total);
            if (next_query_k == query_k) {
                break;
            }
            query_k = next_query_k;
        }
    } else {
        // Fall back to MCI's own search over base codes; the heap holds inner ids.
        QueryContext ctx;
        ctx.alloc = this->allocator_;
        uint32_t dist_cmp = 0;
        auto computer = this->base_codes_->FactoryComputer(vector);
        const auto candidate_limit = static_cast<int64_t>(k + 1);
        DistHeapPtr heap = nullptr;
        if (this->has_clique_index(total) and static_cast<int64_t>(total) > candidate_limit) {
            uint32_t hops = 0;
            auto search_params = MCISearchParameters::FromJson("");
            std::shared_lock<std::shared_mutex> lock(this->global_mutex_);
            heap = this->search_clique_candidates(computer,
                                                  nullptr,
                                                  nullptr,
                                                  search_params,
                                                  candidate_limit,
                                                  ctx,
                                                  nullptr,
                                                  dist_cmp,
                                                  hops);
        } else {
            heap = this->scan_knn_candidates(this->base_codes_,
                                             computer,
                                             nullptr,
                                             candidate_limit,
                                             false,
                                             ctx,
                                             nullptr,
                                             dist_cmp);
        }
        knn_ids.reserve(heap->Size());
        while (not heap->Empty()) {
            const auto inner_id = heap->Top().second;
            if (inner_id != new_inner_id) {
                knn_ids.push_back(inner_id);
            }
            heap->Pop();
        }
    }

    std::sort(knn_ids.begin(), knn_ids.end());
    knn_ids.erase(std::unique(knn_ids.begin(), knn_ids.end()), knn_ids.end());
    return knn_ids;
}

void
MCI::append_node_to_clique(InnerIdType node_id, InnerIdType clique_id) {
    std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
    this->ensure_delta_node_rows(this->total_count_.load());
    if (clique_id >= this->total_logical_clique_count() or
        node_id >= this->delta_node_to_cids_.size()) {
        return;
    }

    auto& node_cliques = this->delta_node_to_cids_[node_id];
    if (std::find(node_cliques.begin(), node_cliques.end(), clique_id) != node_cliques.end()) {
        return;  // already a member of this clique
    }

    if (clique_id < this->total_clique_count_) {
        auto& members = this->delta_clique_extra_[clique_id];
        if (std::find(members.begin(), members.end(), node_id) == members.end()) {
            members.push_back(node_id);
        }
    } else {
        const auto delta_id = clique_id - this->total_clique_count_;
        auto& members = this->delta_cliques_[delta_id];
        if (std::find(members.begin(), members.end(), node_id) == members.end()) {
            members.push_back(node_id);
        }
    }
    node_cliques.push_back(clique_id);
}

void
MCI::append_new_clique(const Vector<InnerIdType>& members) {
    if (members.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
    this->ensure_delta_node_rows(this->total_count_.load());
    const auto new_clique_id = static_cast<InnerIdType>(this->total_logical_clique_count());

    Vector<InnerIdType> normalized(this->allocator_);
    normalized.assign(members.begin(), members.end());
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    this->delta_cliques_.push_back(std::move(normalized));

    for (auto node_id : this->delta_cliques_.back()) {
        if (node_id >= this->delta_node_to_cids_.size()) {
            continue;
        }
        auto& node_cliques = this->delta_node_to_cids_[node_id];
        if (std::find(node_cliques.begin(), node_cliques.end(), new_clique_id) ==
            node_cliques.end()) {
            node_cliques.push_back(new_clique_id);
        }
    }
}

bool
MCI::try_join_existing_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids) {
    if (knn_ids.empty()) {
        return false;
    }

    // Gather one vote per (KNN neighbor, clique) membership. After sorting, each duplicate
    // run is exactly |members(C) intersect KNN(A)| for that clique.
    Vector<InnerIdType> candidate_cliques(this->allocator_);
    {
        std::shared_lock<std::shared_mutex> lock(this->global_mutex_);
        for (auto neighbor : knn_ids) {
            if (neighbor >= this->total_count_.load()) {
                continue;
            }
            this->collect_node_clique_ids(neighbor, candidate_cliques);
        }
    }
    if (candidate_cliques.empty()) {
        return false;
    }
    std::sort(candidate_cliques.begin(), candidate_cliques.end());

    // Evaluate overlap = votes(C) / |members(C)| for each candidate clique.
    // Keep only the top added_mct_ cliques by intersection size.
    Vector<std::pair<uint64_t, InnerIdType>> targets(this->allocator_);
    {
        std::shared_lock<std::shared_mutex> lock(this->global_mutex_);
        const auto logical_clique_count = this->total_logical_clique_count();
        auto iter = candidate_cliques.begin();
        while (iter != candidate_cliques.end()) {
            const auto clique_id = *iter;
            auto next = iter;
            while (next != candidate_cliques.end() and *next == clique_id) {
                ++next;
            }
            const auto inter = static_cast<uint64_t>(std::distance(iter, next));
            iter = next;
            if (clique_id >= logical_clique_count) {
                continue;
            }
            uint64_t member_count = 0;
            if (clique_id < this->total_clique_count_) {
                const auto begin = this->p_maxc_[clique_id];
                const auto end = this->p_maxc_[clique_id + 1];
                member_count = end - begin;
                if (clique_id < this->delta_clique_extra_.size()) {
                    member_count += this->delta_clique_extra_[clique_id].size();
                }
            } else {
                const auto delta_id = clique_id - this->total_clique_count_;
                if (delta_id >= this->delta_cliques_.size()) {
                    continue;
                }
                member_count = this->delta_cliques_[delta_id].size();
            }
            if (member_count == 0) {
                continue;
            }
            const auto ratio = static_cast<float>(inter) / static_cast<float>(member_count);
            if (ratio >= this->join_ratio_threshold_) {
                targets.emplace_back(inter, clique_id);
            }
        }
    }
    if (targets.empty()) {
        return false;
    }

    std::sort(targets.begin(),
              targets.end(),
              [](const std::pair<uint64_t, InnerIdType>& lhs,
                 const std::pair<uint64_t, InnerIdType>& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first > rhs.first;
                  }
                  return lhs.second < rhs.second;
              });
    const auto target_count = std::min<uint64_t>(this->added_mct_, targets.size());
    for (uint64_t i = 0; i < target_count; ++i) {
        this->append_node_to_clique(new_inner_id, targets[i].second);
    }
    return true;
}

void
MCI::build_incremental_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids) {
    Vector<InnerIdType> members(this->allocator_);
    if (knn_ids.empty()) {
        members.push_back(new_inner_id);
        this->append_new_clique(members);
        return;
    }

    // Order neighbors by ascending distance to the new node.
    Vector<std::pair<float, InnerIdType>> sorted_neighbors(this->allocator_);
    sorted_neighbors.reserve(knn_ids.size());
    for (auto neighbor : knn_ids) {
        sorted_neighbors.emplace_back(this->base_codes_->ComputePairVectors(new_inner_id, neighbor),
                                      neighbor);
    }
    std::sort(
        sorted_neighbors.begin(),
        sorted_neighbors.end(),
        [](const std::pair<float, InnerIdType>& lhs, const std::pair<float, InnerIdType>& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

    const auto total = this->total_count_.load();
    const auto candidate_limit = std::min<uint64_t>(this->mcs_, total > 0 ? total - 1 : 0);
    const auto clique_min =
        std::max<uint64_t>(2, std::min<uint64_t>({this->clique_max_, candidate_limit + 1, total}));
    const auto nearest_distance = sorted_neighbors.front().first;

    // Greedily grow a clique around the new node, escalating alpha (mirroring the
    // build-time enumerator) until it reaches clique_min, capped at 100.
    Vector<InnerIdType> best(this->allocator_);
    float now_alpha = std::max(1.2F, this->alpha_);
    while (true) {
        const auto distance_limit = nearest_distance * now_alpha;
        Vector<InnerIdType> clique(this->allocator_);
        clique.push_back(new_inner_id);
        for (const auto& [distance, neighbor] : sorted_neighbors) {
            if (distance > distance_limit) {
                break;
            }
            bool connected = true;
            for (auto member : clique) {
                if (member == new_inner_id) {
                    continue;
                }
                if (this->base_codes_->ComputePairVectors(member, neighbor) > distance_limit) {
                    connected = false;
                    break;
                }
            }
            if (connected) {
                clique.push_back(neighbor);
            }
        }
        if (clique.size() > best.size()) {
            best.swap(clique);
        }
        if (best.size() >= clique_min or now_alpha > 100.0F) {
            break;
        }
        now_alpha *= 2.0F;
    }

    // High-alpha fallback: guarantee the new node is covered by pairing it with
    // its single nearest neighbor.
    if (best.size() < 2) {
        best.clear();
        best.push_back(new_inner_id);
        best.push_back(sorted_neighbors.front().second);
    }
    this->append_new_clique(best);
}

void
MCI::incremental_update_clique(InnerIdType new_inner_id, const float* vector) {
    auto knn_ids = this->find_knn_for_new_node(new_inner_id, vector);
    if (knn_ids.empty()) {
        Vector<InnerIdType> singleton(this->allocator_);
        singleton.push_back(new_inner_id);
        this->append_new_clique(singleton);
        return;
    }
    if (not this->try_join_existing_clique(new_inner_id, knn_ids)) {
        this->build_incremental_clique(new_inner_id, knn_ids);
    }
}

DatasetPtr
MCI::SearchWithRequest(const SearchRequest& request) const {
    std::shared_lock<std::shared_mutex> read_lock(this->global_mutex_);
    CHECK_ARGUMENT(request.query_ != nullptr, "query dataset is nullptr");
    CHECK_ARGUMENT(
        request.query_->GetDim() == dim_,
        fmt::format(
            "query.dim({}) must be equal to index.dim({})", request.query_->GetDim(), dim_));
    CHECK_ARGUMENT(request.query_->GetFloat32Vectors() != nullptr, "query.float_vector is nullptr");
    CHECK_ARGUMENT(request.topk_ > 0, "mci topk must be positive");

    auto search_params = MCISearchParameters::FromJson(request.params_str_);
    const auto* query_data = request.query_->GetFloat32Vectors();
    auto final_codes = this->get_reorder_codes();
    auto final_computer = final_codes->FactoryComputer(query_data);
    FilterPtr filter = nullptr;
    if (request.enable_filter_ and request.filter_ != nullptr) {
        filter = request.filter_;
    }
    const auto hybrid_valid_ratio = filter != nullptr ? filter->ValidRatio() : 1.0F;
    if (this->should_use_hgraph_hybrid(request, hybrid_valid_ratio)) {
        return this->search_hgraph_hybrid(request, hybrid_valid_ratio);
    }
    if (request.enable_bitset_filter_ and request.bitset_filter_ != nullptr) {
        auto combined_filter = std::make_shared<CombinedFilter>();
        if (filter != nullptr) {
            combined_filter->AppendFilter(filter);
        }
        combined_filter->AppendFilter(std::make_shared<BlackListFilter>(request.bitset_filter_));
        filter = combined_filter;
    }
    auto seed_inner_ids = collect_valid_inner_ids(
        filter, *this->label_table_, search_params.seed_count, this->allocator_);
    auto* seed_inner_ids_ptr = seed_inner_ids.empty() ? nullptr : &seed_inner_ids;
    auto inner_filter = make_inner_id_filter(filter, *this->label_table_);

    auto total = static_cast<InnerIdType>(this->total_count_.load());
    if (total == 0) {
        return DatasetImpl::MakeEmptyDataset();
    }

    uint32_t dist_cmp = 0;
    if (request.mode_ == SearchMode::RANGE_SEARCH) {
        int64_t limited_size =
            request.limited_size_ < 0 ? std::numeric_limits<int64_t>::max() : request.limited_size_;
        auto heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, limited_size);
        for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
            if (inner_filter != nullptr and not inner_filter->CheckValid(inner_id)) {
                continue;
            }
            float dist = 0.0F;
            final_codes->Query(&dist, final_computer, &inner_id, 1);
            ++dist_cmp;
            if (dist <= request.radius_) {
                heap->Push(dist, inner_id);
            }
        }
        auto dataset_results = this->build_dataset_from_heap(heap);
        JsonType stats;
        stats["dist_cmp"].SetInt(static_cast<int64_t>(dist_cmp));
        stats["ef_search"].SetInt(search_params.ef_search);
        dataset_results->Statistics(stats.Dump());
        return dataset_results;
    }

    auto topk = std::max<int64_t>(request.topk_, 1);
    auto candidate_limit = std::max<int64_t>(topk, search_params.ef_search);
    QueryContext ctx;
    ctx.alloc = this->allocator_;
    uint32_t hops = 0;
    DistanceRecordVector rabitq_lower_bound_candidates(this->allocator_);
    auto* rabitq_lower_bound_candidates_ptr =
        search_params.rabitq_one_bit_search and use_reorder_ and reorder_by_base_
            ? &rabitq_lower_bound_candidates
            : nullptr;

    auto base_computer = this->base_codes_->FactoryComputer(query_data);
    DistHeapPtr heap = nullptr;
    if (this->has_clique_index(total) and static_cast<int64_t>(total) > candidate_limit) {
        heap = this->search_clique_candidates(base_computer,
                                              inner_filter,
                                              seed_inner_ids_ptr,
                                              search_params,
                                              candidate_limit,
                                              ctx,
                                              rabitq_lower_bound_candidates_ptr,
                                              dist_cmp,
                                              hops);
    } else {
        heap = this->scan_knn_candidates(this->base_codes_,
                                         base_computer,
                                         inner_filter,
                                         candidate_limit,
                                         search_params.rabitq_one_bit_search,
                                         ctx,
                                         rabitq_lower_bound_candidates_ptr,
                                         dist_cmp);
    }

    if (heap->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }

    if (use_reorder_ or search_params.rabitq_one_bit_search) {
        auto reorder_codes = use_reorder_ ? this->get_reorder_codes() : this->base_codes_;
        FlattenReorder reorder(reorder_codes, this->allocator_);
        const auto* lower_bound_candidates_for_reorder =
            use_reorder_ and reorder_by_base_ and search_params.rabitq_one_bit_search
                ? rabitq_lower_bound_candidates_ptr
                : nullptr;
        heap = reorder.Reorder(
            heap, query_data, topk, ctx, nullptr, lower_bound_candidates_for_reorder);
    } else if (static_cast<int64_t>(heap->Size()) > topk) {
        auto trimmed_heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, topk);
        const auto* candidates = heap->GetData();
        for (uint64_t i = 0; i < heap->Size(); ++i) {
            trimmed_heap->Push(candidates[i]);
        }
        heap = trimmed_heap;
    }

    auto dataset_results = this->build_dataset_from_heap(heap);
    JsonType stats;
    stats["dist_cmp"].SetInt(static_cast<int64_t>(dist_cmp));
    stats["ef_search"].SetInt(search_params.ef_search);
    stats["seed_count"].SetInt(static_cast<int64_t>(search_params.seed_count));
    stats["rabitq_one_bit_search"].SetBool(search_params.rabitq_one_bit_search);
    stats["hops"].SetInt(static_cast<int64_t>(hops));
    stats["total_clique_count"].SetInt(static_cast<int64_t>(this->total_logical_clique_count()));
    stats["base_clique_count"].SetInt(static_cast<int64_t>(this->total_clique_count_));
    stats["delta_clique_count"].SetInt(static_cast<int64_t>(this->delta_cliques_.size()));
    stats["mci_hybrid_route"].SetString("mci");
    stats["mci_hybrid_valid_ratio"].SetFloat(hybrid_valid_ratio);
    stats["mci_hybrid_threshold"].SetFloat(this->hgraph_valid_ratio_threshold_);
    dataset_results->Statistics(stats.Dump());
    return dataset_results;
}

void
MCI::Serialize(StreamWriter& writer) const {
    this->base_codes_->Serialize(writer);
    if (this->reorder_codes_ != nullptr) {
        this->reorder_codes_->Serialize(writer);
    }
    this->label_table_->Serialize(writer);
    StreamWriter::WriteVector(writer, this->p_maxc_);
    StreamWriter::WriteVector(writer, this->maxcs_);
    StreamWriter::WriteVector(writer, this->p_node_to_cid_);
    StreamWriter::WriteVector(writer, this->node_to_cids_);
    write_nested_vector(writer, this->delta_cliques_);
    write_nested_vector(writer, this->delta_clique_extra_);
    write_nested_vector(writer, this->delta_node_to_cids_);
    uint64_t hgraph_serialized_size = 0;
    if (this->hgraph_index_ != nullptr) {
        const auto hgraph_begin = writer.GetCursor();
        this->hgraph_index_->Serialize(writer);
        hgraph_serialized_size = writer.GetCursor() - hgraph_begin;
    }

    auto metadata = std::make_shared<Metadata>();
    JsonType basic_info;
    basic_info["dim"].SetInt(dim_);
    basic_info["total_count"].SetInt(static_cast<int64_t>(this->total_count_.load()));
    basic_info["max_capacity"].SetInt(static_cast<int64_t>(this->max_capacity_.load()));
    basic_info["total_clique_count"].SetInt(static_cast<int64_t>(this->total_clique_count_));
    basic_info["lazy_delta_version"].SetInt(1);
    basic_info["delta_clique_count"].SetInt(static_cast<int64_t>(this->delta_cliques_.size()));
    if (hgraph_serialized_size > 0) {
        basic_info["hgraph_serialized_size"].SetInt(static_cast<int64_t>(hgraph_serialized_size));
    }
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    metadata->Set(BASIC_INFO, basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
MCI::Deserialize(StreamReader& reader) {
    auto footer = Footer::Parse(reader);
    CHECK_ARGUMENT(footer != nullptr, "mci deserialize requires new serialization footer");
    auto metadata = footer->GetMetadata();
    if (metadata->EmptyIndex()) {
        return;
    }

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);
    auto basic_info = metadata->Get(BASIC_INFO);
    if (basic_info.Contains(INDEX_PARAM)) {
        auto index_param = std::make_shared<MCIParameter>();
        index_param->FromString(basic_info[INDEX_PARAM].GetString());
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("MCI index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }

    dim_ = basic_info["dim"].GetInt();
    this->total_count_.store(static_cast<uint64_t>(basic_info["total_count"].GetInt()));
    this->max_capacity_.store(static_cast<uint64_t>(basic_info["max_capacity"].GetInt()));
    this->total_clique_count_ = static_cast<uint64_t>(basic_info["total_clique_count"].GetInt());
    uint64_t hgraph_serialized_size = 0;
    if (basic_info.Contains("hgraph_serialized_size")) {
        hgraph_serialized_size =
            static_cast<uint64_t>(basic_info["hgraph_serialized_size"].GetInt());
    }

    this->base_codes_->Deserialize(buffer_reader);
    if (this->reorder_codes_ != nullptr) {
        this->reorder_codes_->Deserialize(buffer_reader);
    }
    this->label_table_->Deserialize(buffer_reader);
    StreamReader::ReadVector(buffer_reader, this->p_maxc_);
    StreamReader::ReadVector(buffer_reader, this->maxcs_);
    StreamReader::ReadVector(buffer_reader, this->p_node_to_cid_);
    StreamReader::ReadVector(buffer_reader, this->node_to_cids_);
    this->validate_clique_csr(this->total_count_.load());
    if (basic_info.Contains("lazy_delta_version")) {
        read_nested_vector(buffer_reader, this->delta_cliques_, this->allocator_);
        read_nested_vector(buffer_reader, this->delta_clique_extra_, this->allocator_);
        read_nested_vector(buffer_reader, this->delta_node_to_cids_, this->allocator_);
        this->ensure_delta_node_rows(this->total_count_.load());
    } else {
        this->reset_delta_clique_index(this->total_count_.load());
    }
    if (hgraph_serialized_size > 0) {
        const auto hgraph_begin = buffer_reader.GetCursor();
        if (this->hgraph_index_ != nullptr) {
            auto hgraph_reader = buffer_reader.Slice(hgraph_serialized_size);
            this->hgraph_index_->Deserialize(hgraph_reader);
        }
        buffer_reader.Seek(hgraph_begin + hgraph_serialized_size);
    }
    if (this->hgraph_index_ != nullptr and
        this->hgraph_index_->GetNumElements() != this->GetNumElements() and
        not this->hgraph_index_path_.empty()) {
        this->load_hgraph_index(this->hgraph_index_path_);
    }
    this->cal_memory_usage();
}

void
MCI::InitFeatures() {
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
    });

    auto name = this->base_codes_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16 and
        name != QUANTIZATION_TYPE_VALUE_FP16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    }
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_L2);
    } else if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_COSINE);
    }
}

int64_t
MCI::GetMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(this->memory_usage_mutex_);
    return this->current_memory_usage_.load();
}

void
MCI::resize(uint64_t new_size) {
    auto new_size_power_2 = next_multiple_of_power_of_two(new_size, 10);
    auto cur_size = this->max_capacity_.load();
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->base_codes_->Resize(new_size_power_2);
        if (this->reorder_codes_ != nullptr) {
            this->reorder_codes_->Resize(new_size_power_2);
        }
        this->label_table_->Resize(new_size_power_2);
        this->max_capacity_.store(new_size_power_2);
    }
}

const float*
MCI::get_float_vectors(const DatasetPtr& data) const {
    CHECK_ARGUMENT(data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   fmt::format("MCI currently supports only {} datatype", DATATYPE_FLOAT32));
    const auto* vectors = data->GetFloat32Vectors();
    CHECK_ARGUMENT(vectors != nullptr, "float vectors are nullptr");
    return vectors;
}

FlattenInterfacePtr
MCI::get_search_codes() const {
    if (this->reorder_codes_ != nullptr) {
        return this->reorder_codes_;
    }
    return this->base_codes_;
}

FlattenInterfacePtr
MCI::get_reorder_codes() const {
    if (this->use_reorder_ and not this->reorder_by_base_ and this->reorder_codes_ != nullptr) {
        return this->reorder_codes_;
    }
    return this->base_codes_;
}

void
MCI::cal_memory_usage() {
    uint64_t memory = sizeof(MCI);
    if (this->base_codes_ != nullptr) {
        memory += static_cast<uint64_t>(this->base_codes_->GetMemoryUsage());
    }
    if (this->reorder_codes_ != nullptr) {
        memory += static_cast<uint64_t>(this->reorder_codes_->GetMemoryUsage());
    }
    memory += static_cast<uint64_t>(this->label_table_->GetMemoryUsage());
    memory += this->p_maxc_.capacity() * sizeof(InnerIdType);
    memory += this->maxcs_.capacity() * sizeof(InnerIdType);
    memory += this->p_node_to_cid_.capacity() * sizeof(InnerIdType);
    memory += this->node_to_cids_.capacity() * sizeof(InnerIdType);
    memory += nested_vector_memory(this->delta_cliques_);
    memory += nested_vector_memory(this->delta_clique_extra_);
    memory += nested_vector_memory(this->delta_node_to_cids_);
    if (this->hgraph_index_ != nullptr) {
        memory += static_cast<uint64_t>(this->hgraph_index_->GetMemoryUsage());
    }
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

void
MCI::load_hgraph_index(const std::string& index_path) {
    CHECK_ARGUMENT(this->hgraph_index_ != nullptr, "mci hgraph hybrid sub-index is null");
    std::ifstream input(index_path, std::ios::binary);
    CHECK_ARGUMENT(input.good(), fmt::format("failed to open hgraph index: {}", index_path));
    IOStreamReader reader(input);
    this->hgraph_index_->Deserialize(reader);
    CHECK_ARGUMENT(this->hgraph_index_->GetNumElements() == this->GetNumElements() or  // NOLINT
                       this->GetNumElements() == 0,
                   fmt::format("hgraph index size {} does not match mci size {}",
                               this->hgraph_index_->GetNumElements(),
                               this->GetNumElements()));
}

ParamPtr
MCI::CheckAndMappingExternalParam(const JsonType& external_param,
                                  const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   fmt::format("MCI currently supports only {} datatype", DATATYPE_FLOAT32));
    ConstParamMap external_mapping = {
        {BASE_CODES_KEY, {BASE_CODES_KEY}},
        {PRECISE_CODES_KEY, {PRECISE_CODES_KEY}},
        {USE_REORDER_KEY, {USE_REORDER_KEY}},
        {REORDER_SOURCE_KEY, {REORDER_SOURCE_KEY}},
        {BUILD_THREAD_COUNT_KEY, {BUILD_THREAD_COUNT_KEY}},
        {LABEL_REMAP_TYPE_KEY, {LABEL_REMAP_TYPE_KEY}},
        {USE_ATTRIBUTE_FILTER_KEY, {USE_ATTRIBUTE_FILTER_KEY}},
        {ATTR_PARAMS_KEY, {ATTR_PARAMS_KEY}},
        {MCI_PARAMETER_MAX_DEGREE, {MCI_PARAMETER_MAX_DEGREE}},
        {MCI_PARAMETER_MCS, {MCI_PARAMETER_MCS}},
        {MCI_PARAMETER_CLIQUE_MAX, {MCI_PARAMETER_CLIQUE_MAX}},
        {MCI_PARAMETER_ALPHA, {MCI_PARAMETER_ALPHA}},
        {MCI_PARAMETER_JOIN_RATIO_THRESHOLD, {MCI_PARAMETER_JOIN_RATIO_THRESHOLD}},
        {MCI_PARAMETER_ADDED_MCT, {MCI_PARAMETER_ADDED_MCT}},
        {MCI_PARAMETER_KNNG_PATH, {MCI_PARAMETER_KNNG_PATH}},
        {MCI_PARAMETER_CLIQUE_PATH, {MCI_PARAMETER_CLIQUE_PATH}},
        {MCI_PARAMETER_USE_HGRAPH_HYBRID, {MCI_PARAMETER_USE_HGRAPH_HYBRID}},
        {MCI_PARAMETER_HGRAPH_VALID_RATIO_THRESHOLD, {MCI_PARAMETER_HGRAPH_VALID_RATIO_THRESHOLD}},
        {MCI_PARAMETER_HGRAPH_INDEX_PARAM, {MCI_PARAMETER_HGRAPH_INDEX_PARAM}},
        {MCI_PARAMETER_HGRAPH_INDEX_PATH, {MCI_PARAMETER_HGRAPH_INDEX_PATH}},
        {MCI_PARAMETER_HGRAPH_EF_SEARCH, {MCI_PARAMETER_HGRAPH_EF_SEARCH}},
        {"base_quantization_type", {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {"base_codes_type", {BASE_CODES_KEY, CODES_TYPE_KEY}},
        {"base_io_type", {BASE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {"base_file_path", {BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {"precise_quantization_type", {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {"precise_io_type", {PRECISE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {"precise_file_path", {PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {"rabitq_bits_per_dim_query",
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}},
        {"rabitq_bits_per_dim_base",
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}},
        {"rabitq_error_rate",
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_ERROR_RATE_KEY}},
        {"rabitq_version",
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_VERSION_KEY}},
    };

    auto inner_json = JsonType::Parse(MCI_PARAMS_TEMPLATE);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);
    if (external_param.Contains(MCI_PARAMETER_HGRAPH_INDEX_PARAM)) {
        auto hgraph_param = HGraph::CheckAndMappingExternalParam(
            external_param[MCI_PARAMETER_HGRAPH_INDEX_PARAM], common_param);
        inner_json[MCI_PARAMETER_HGRAPH_INDEX_PARAM].SetJson(hgraph_param->ToJson());
    }

    auto mci_parameter = std::make_shared<MCIParameter>();
    mci_parameter->FromJson(inner_json);
    return mci_parameter;
}

}  // namespace vsag
