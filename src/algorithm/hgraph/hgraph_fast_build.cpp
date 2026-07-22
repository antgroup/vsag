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
WaitAllFutures(std::vector<std::future<void>>& futures) {
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

HGraphFastBuildGuard::HGraphFastBuildGuard(HGraph& hgraph) : hgraph_(&hgraph) {
    const bool build_uses_base_codes =
        hgraph.has_precise_reorder() ? hgraph.build_by_base_ : hgraph.raw_vector_ == nullptr;
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

HGraphFastBuildGuard::~HGraphFastBuildGuard() {
    if (optimized_build_codes_ != nullptr) {
        optimized_build_codes_->AbortOptimizedBuild();
        hgraph_->optimized_build_codes_.reset();
    }
}

void
HGraphFastBuildGuard::Finalize() {
    if (optimized_build_codes_ == nullptr) {
        return;
    }
    optimized_build_codes_->FinalizeOptimizedBuild();
    hgraph_->optimized_build_codes_.reset();
    optimized_build_codes_.reset();
}

HGraphFastBuildTaskGuard::HGraphFastBuildTaskGuard(std::vector<std::future<void>>& futures,
                                                   bool enabled,
                                                   uint64_t capacity)
    : futures_(futures), enabled_(enabled) {
    if (enabled_) {
        futures_.reserve(capacity);
    }
}

HGraphFastBuildTaskGuard::~HGraphFastBuildTaskGuard() {
    if (not enabled_) {
        return;
    }
    try {
        WaitAllFutures(futures_);
    } catch (...) {
    }
}

void
HGraph::prepare_optimized_build_codes(const DatasetPtr& data,
                                      const Vector<std::pair<InnerIdType, LabelType>>& inner_ids,
                                      std::vector<std::future<void>>& futures) {
    if (this->optimized_build_codes_ == nullptr) {
        return;
    }

    if (this->thread_pool_ == nullptr) {
        for (const auto& [inner_id, local_idx] : inner_ids) {
            this->insert_persistent_codes(get_data(data, local_idx), inner_id);
        }
        return;
    }

    for (const auto& [inner_id, local_idx] : inner_ids) {
        futures.emplace_back(
            this->thread_pool_->GeneralEnqueue([this, data, inner_id, local_idx]() {
                this->insert_persistent_codes(get_data(data, local_idx), inner_id);
            }));
    }
    WaitAllFutures(futures);
    futures.clear();
}

DistHeapPtr
HGraph::search_one_graph_for_optimized_build(const void* query,
                                             const GraphInterfacePtr& graph,
                                             const FlattenInterfacePtr& flatten,
                                             InnerSearchParam& inner_search_param,
                                             const ComputerInterfacePtr& computer) const {
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
