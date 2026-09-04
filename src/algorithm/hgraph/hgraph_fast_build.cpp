// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hgraph_fast_build.h"

#include <exception>
#include <utility>

#include "dataset_impl.h"
#include "hgraph.h"

namespace vsag {
namespace {

void
wait_all_futures(std::vector<std::future<void>>& futures) {
    std::exception_ptr first_exception = nullptr;
    for (auto& future : futures) {
        if (not future.valid()) {
            continue;
        }
        try {
            future.get();
        } catch (...) {
            if (not first_exception) {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

}  // namespace

HGraphOptimizedBuildSession::HGraphOptimizedBuildSession(HGraph& hgraph) : hgraph_(&hgraph) {
    if (hgraph.using_dedup_storage()) {
        return;
    }
    const bool build_uses_base_codes = not hgraph.has_precise_reorder() or hgraph.build_by_base_;
    if (not build_uses_base_codes) {
        return;
    }

    auto optimized_build_codes =
        std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(hgraph.basic_flatten_codes_);
    if (optimized_build_codes == nullptr) {
        return;
    }

    FlattenOptimizedBuildContext context{hgraph.thread_pool_, hgraph.build_thread_count_};
    if (not optimized_build_codes->BeginOptimizedBuild(context)) {
        return;
    }

    optimized_build_codes_ = std::move(optimized_build_codes);
    hgraph.optimized_build_codes_ = optimized_build_codes_;
}

HGraphOptimizedBuildSession::~HGraphOptimizedBuildSession() {
    if (optimized_build_codes_ != nullptr) {
        optimized_build_codes_->AbortOptimizedBuild();
        hgraph_->optimized_build_codes_.reset();
    }
}

void
HGraphOptimizedBuildSession::Commit() {
    if (optimized_build_codes_ == nullptr) {
        return;
    }
    optimized_build_codes_->FinalizeOptimizedBuild();
    hgraph_->optimized_build_codes_.reset();
    optimized_build_codes_.reset();
}

bool
HGraphOptimizedBuildSession::Active() const {
    return optimized_build_codes_ != nullptr;
}

HGraphBuildTaskGuard::HGraphBuildTaskGuard(std::vector<std::future<void>>& futures,
                                           uint64_t capacity)
    : futures_(futures) {
    if (capacity > 0) {
        futures_.reserve(capacity);
    }
}

HGraphBuildTaskGuard::~HGraphBuildTaskGuard() {
    try {
        wait_all_futures(futures_);
    } catch (...) {
    }
}

std::optional<std::vector<int64_t>>
HGraph::try_optimized_build(const DatasetPtr& data) {
    // Start the session before training so unsupported configurations fall through without
    // changing the training behavior of the normal build path.
    HGraphOptimizedBuildSession session(*this);
    if (not session.Active()) {
        return std::nullopt;
    }

    std::vector<int64_t> result;
    if (graph_type_ == GRAPH_TYPE_VALUE_NSW) {
        result = this->Add(data);
    } else {
        result = this->build_by_odescent(data);
    }
    session.Commit();
    return result;
}

bool
HGraph::need_temporary_sq8_build_data_for_add() const {
    return this->optimized_build_codes_ == nullptr and not this->has_precise_reorder() and
           this->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_RABITQ;
}

bool
HGraph::prepare_build_codes(const DatasetPtr& data,
                            const Vector<AddRow>& rows,
                            const AddContext& context) {
    if (this->optimized_build_codes_ != nullptr) {
        if (this->thread_pool_ == nullptr) {
            for (const auto& row : rows) {
                this->insert_persistent_codes(get_data(data, row.input_idx), row.inner_id);
            }
            return true;
        }

        std::vector<std::future<void>> futures;
        HGraphBuildTaskGuard task_guard(futures, static_cast<uint64_t>(rows.size()));
        // Parallel graph insertion may probe rows from the same batch. Make every scalar code
        // visible before any of those probes starts.
        for (const auto& row : rows) {
            const auto inner_id = row.inner_id;
            const auto input_idx = row.input_idx;
            futures.emplace_back(
                this->thread_pool_->GeneralEnqueue([this, data, inner_id, input_idx]() {
                    this->insert_persistent_codes(get_data(data, input_idx), inner_id);
                }));
        }
        wait_all_futures(futures);
        futures.clear();
        return true;
    }

    const auto row_count = static_cast<uint64_t>(rows.size());
    const bool use_parallel_encode =
        context.first_empty_add && not context.use_dedup_storage &&
        not this->support_force_remove() && this->thread_pool_ != nullptr &&
        this->build_thread_count_ > 1 && row_count > 1 && this->has_precise_reorder() &&
        not this->build_by_base_ && not this->create_new_raw_vector_ &&
        this->basic_flatten_codes_->InMemory() && this->high_precise_codes_->InMemory() &&
        this->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_RABITQ &&
        this->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SQ8;
    if (not use_parallel_encode) {
        return false;
    }

    const uint64_t worker_count = std::min(this->build_thread_count_, row_count);
    const uint64_t block_size = (row_count + worker_count - 1) / worker_count;
    std::vector<std::future<void>> futures;
    HGraphBuildTaskGuard task_guard(futures, worker_count);

    // prepare_add_batch() already resized both layouts. The workers only encode disjoint IDs;
    // these outer locks keep storage stable until every worker finishes.
    std::shared_lock<std::shared_mutex> add_lock(this->add_mutex_);
    std::unique_lock<std::shared_mutex> codes_lock(this->persistent_codes_mutex_);
    for (uint64_t begin = 0; begin < row_count; begin += block_size) {
        const uint64_t end = std::min(begin + block_size, row_count);
        futures.emplace_back(this->thread_pool_->GeneralEnqueue([this, data, &rows, begin, end]() {
            for (uint64_t offset = begin; offset < end; ++offset) {
                const auto& row = rows[offset];
                this->insert_persistent_codes_unlocked(get_data(data, row.input_idx), row.inner_id);
            }
        }));
    }
    wait_all_futures(futures);
    futures.clear();
    return true;
}

bool
HGraph::should_insert_codes_before_probe(bool use_dedup_storage) const {
    return not use_dedup_storage and this->optimized_build_codes_ == nullptr;
}

ComputerInterfacePtr
HGraph::make_build_computer(const void* query, InnerIdType inner_id) const {
    if (this->optimized_build_codes_ == nullptr) {
        return nullptr;
    }
    return this->optimized_build_codes_->FactoryComputerForBuild(query, inner_id);
}

DistHeapPtr
HGraph::search_graph_for_build(const void* query,
                               const GraphInterfacePtr& graph,
                               const FlattenInterfacePtr& flatten,
                               InnerSearchParam& inner_search_param,
                               const ComputerInterfacePtr& computer) const {
    if (computer == nullptr) {
        return this->search_one_graph(
            query, graph, flatten, inner_search_param, (VisitedListPtr) nullptr, nullptr);
    }

    auto visited_list = this->pool_->TakeOne();
    try {
        auto result = this->searcher_->SearchWithPresetComputer(graph,
                                                                flatten,
                                                                visited_list,
                                                                query,
                                                                inner_search_param,
                                                                this->label_table_,
                                                                nullptr,
                                                                nullptr,
                                                                computer);
        this->pool_->ReturnOne(visited_list);
        return result;
    } catch (...) {
        this->pool_->ReturnOne(visited_list);
        throw;
    }
}

}  // namespace vsag
