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

#include <algorithm>
#include <limits>

#include "common.h"
#include "dataset_impl.h"
#include "hgraph.h"  // IWYU pragma: keep

namespace vsag {

uint32_t
HGraph::Feedback(const DatasetPtr& query,
                 int64_t k,
                 const std::string& parameters,
                 int64_t global_optimum_tag_id) {
    if (not this->use_conjugate_graph_) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "HGraph conjugate graph is not enabled");
    }
    CHECK_ARGUMENT(k > 0, "feedback k must be positive");
    this->validate_knn_args(query, k);
    if (this->GetNumElements() == 0) {
        return 0;
    }

    if (global_optimum_tag_id == std::numeric_limits<int64_t>::max()) {
        auto global_lock = this->acquire_global_read_lock();
        QueryContext ctx{.alloc = this->allocator_};
        auto exact = this->brute_force_search<InnerSearchMode::KNN_SEARCH>(
            this->get_data(query), nullptr, 1, 0.0F, &ctx);
        CHECK_ARGUMENT(not exact->Empty(), "feedback cannot find an exact nearest neighbor");
        std::shared_lock label_lock(this->label_lookup_mutex_);
        global_optimum_tag_id = this->label_table_->GetLabelById(exact->Top().second);
    }

    auto result = this->KnnSearch(query, k, parameters, nullptr);
    const auto* ids = result->GetIds();
    const auto result_size = result->GetDim();
    uint32_t inserted = 0;
    std::shared_lock label_lock(this->label_lookup_mutex_);
    const auto [optimum_found, unused] =
        this->label_table_->TryGetIdByLabel(global_optimum_tag_id, true);
    CHECK_ARGUMENT(optimum_found, "feedback global optimum id does not belong to the index");
    std::unique_lock graph_lock(this->conjugate_graph_mutex_);
    uint64_t added_memory = 0;
    for (int64_t i = 0; i < std::min(k, result_size); ++i) {
        const auto [found, unused] = this->label_table_->TryGetIdByLabel(ids[i], true);
        CHECK_ARGUMENT(found, "feedback search result does not belong to the index");
        const auto memory_before = this->conjugate_graph_->GetMemoryUsage();
        auto add_result = this->conjugate_graph_->AddNeighbor(ids[i], global_optimum_tag_id);
        if (not add_result) {
            throw VsagException(add_result.error().type, add_result.error().message);
        }
        if (add_result.value()) {
            ++inserted;
            added_memory += this->conjugate_graph_->GetMemoryUsage() - memory_before;
        }
    }
    if (added_memory > 0) {
        std::unique_lock memory_lock(this->memory_usage_mutex_);
        this->current_memory_usage_.fetch_add(added_memory);
    }
    return inserted;
}

uint32_t
HGraph::Pretrain(const std::vector<int64_t>& base_tag_ids,
                 uint32_t k,
                 const std::string& parameters) {
    if (not this->use_conjugate_graph_) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "HGraph conjugate graph is not enabled");
    }
    CHECK_ARGUMENT(k > 0, "pretrain k must be positive");
    CHECK_ARGUMENT(this->data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   "HGraph pretrain currently supports float32 vectors only");
    if (this->GetNumElements() == 0 or base_tag_ids.empty()) {
        return 0;
    }

    Vector<float> base_vector(this->dim_, this->allocator_);
    Vector<float> neighbor_vector(this->dim_, this->allocator_);
    Vector<float> generated_vector(this->dim_, this->allocator_);
    auto base = DatasetImpl::Make();
    base->Dim(this->dim_)->NumElements(1)->Float32Vectors(base_vector.data())->Owner(false);
    auto generated = DatasetImpl::Make();
    generated->Dim(this->dim_)
        ->NumElements(1)
        ->Float32Vectors(generated_vector.data())
        ->Owner(false);
    const auto generate_parameters = fmt::format(
        R"({{"hgraph":{{"ef_search":{},"use_conjugate_graph_search":false}}}})", GENERATE_SEARCH_L);

    uint32_t inserted = 0;
    for (const auto base_tag_id : base_tag_ids) {
        InnerIdType base_inner_id;
        {
            std::shared_lock label_lock(this->label_lookup_mutex_);
            const auto [found, inner_id] = this->label_table_->TryGetIdByLabel(base_tag_id, true);
            CHECK_ARGUMENT(
                found,
                fmt::format("pretrain base id {} does not belong to the index", base_tag_id));
            base_inner_id = inner_id;
        }
        this->GetVectorByInnerId(base_inner_id, base_vector.data());
        auto neighbors = this->KnnSearch(base, GENERATE_SEARCH_K, generate_parameters, nullptr);
        for (int64_t i = 0; i < neighbors->GetDim(); ++i) {
            const auto neighbor_tag_id = neighbors->GetIds()[i];
            if (neighbor_tag_id == base_tag_id) {
                continue;
            }
            InnerIdType neighbor_inner_id;
            {
                std::shared_lock label_lock(this->label_lookup_mutex_);
                const auto [found, inner_id] =
                    this->label_table_->TryGetIdByLabel(neighbor_tag_id, true);
                CHECK_ARGUMENT(found,
                               fmt::format("pretrain neighbor id {} does not belong to the index",
                                           neighbor_tag_id));
                neighbor_inner_id = inner_id;
            }
            this->GetVectorByInnerId(neighbor_inner_id, neighbor_vector.data());
            for (int64_t d = 0; d < this->dim_; ++d) {
                generated_vector[d] =
                    GENERATE_OMEGA * base_vector[d] + (1.0F - GENERATE_OMEGA) * neighbor_vector[d];
            }
            inserted += this->Feedback(generated, k, parameters, base_tag_id);
        }
    }
    return inserted;
}

bool
HGraph::UpdateId(int64_t old_id, int64_t new_id) {
    const auto updated = InnerIndexInterface::UpdateId(old_id, new_id);
    if (updated and this->use_conjugate_graph_) {
        std::unique_lock graph_lock(this->conjugate_graph_mutex_);
        (void)this->conjugate_graph_->UpdateId(old_id, new_id);
    }
    return updated;
}

}  // namespace vsag
