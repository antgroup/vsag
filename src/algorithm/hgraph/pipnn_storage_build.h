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

#include <algorithm>
#include <exception>
#include <future>
#include <limits>
#include <utility>
#include <vector>

#include "common.h"
#include "container_types.h"
#include "datacell/code_slot_map.h"
#include "datacell/graph_interface.h"
#include "impl/thread_pool/safe_thread_pool.h"

namespace vsag {

// Encode representative input rows, then publish every logical ID's shared storage slot.
// Capacity and encoding callbacks must not allocate/publish slots; this function commits them
// only after all workers succeed. The caller keeps graph, slot_map and allocator alive.
template <typename EnsureCapacity, typename EncodeRow>
inline void
materialize_pipnn_storage(const Vector<InnerIdType>& inner_ids,
                          const Vector<int64_t>& valid_indices,
                          const GraphInterfacePtr& graph,
                          CodeSlotMap& slot_map,
                          Allocator* allocator,
                          SafeThreadPool* thread_pool,
                          uint64_t thread_count,
                          EnsureCapacity&& ensure_capacity,
                          EncodeRow&& encode_row) {
    struct representative_row {
        InnerIdType inner_id;
        CodeSlotIdType slot;
        int64_t row;
    };
    const auto unassigned = std::numeric_limits<uint64_t>::max();
    UnorderedMap<InnerIdType, uint64_t> row_by_id(allocator);
    row_by_id.reserve(inner_ids.size());
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        row_by_id.emplace(inner_ids[i], i);
    }
    Vector<uint64_t> group_rows(inner_ids.size(), unassigned, allocator);
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        if (group_rows[i] != unassigned) {
            continue;
        }
        const auto group = graph->GetGroupId(inner_ids[i]);
        const auto group_row = row_by_id.at(group);
        group_rows[group_row] = group_row;
        for (auto member : graph->GetDuplicateIds(group)) {
            if (auto found = row_by_id.find(member); found != row_by_id.end()) {
                group_rows[found->second] = group_row;
            }
        }
    }
    Vector<representative_row> representatives(allocator);
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        if (group_rows[i] == i) {
            representatives.push_back({inner_ids[i], 0, valid_indices[i]});
        }
    }
    // The caller excludes concurrent slot allocators. Encode into provisional slots so failure
    // does not consume slot IDs or change mappings; allocated buffers remain reusable.
    const auto first_slot = slot_map.PhysicalCount();
    CHECK_ARGUMENT(
        representatives.size() <= std::numeric_limits<CodeSlotIdType>::max() - first_slot,
        "PiPNN storage exceeds the code slot limit");
    for (uint64_t i = 0; i < representatives.size(); ++i) {
        representatives[i].slot = first_slot + static_cast<CodeSlotIdType>(i);
    }
    ensure_capacity(first_slot + static_cast<CodeSlotIdType>(representatives.size()));

    // Capacity is fixed before workers write. Wait for every scheduled task before publishing
    // mappings or letting representative rows go out of scope, even if scheduling/encoding fails.
    const uint64_t workers = std::min<uint64_t>(
        representatives.size(), thread_pool != nullptr ? std::max<uint64_t>(1, thread_count) : 1);
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    std::exception_ptr failure;
    try {
        for (uint64_t worker = 0; worker < workers; ++worker) {
            const auto begin = representatives.size() * worker / workers;
            const auto end = representatives.size() * (worker + 1) / workers;
            auto encode = [&representatives, encode_row, begin, end]() {
                for (auto i = begin; i < end; ++i) {
                    const auto& representative = representatives[i];
                    encode_row(representative.row, representative.slot);
                }
            };
            if (workers > 1) {
                futures.emplace_back(thread_pool->GeneralEnqueue(std::move(encode)));
            } else {
                encode();
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (failure == nullptr) {
                failure = std::current_exception();
            }
        }
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    for (const auto& representative : representatives) {
        slot_map.PublishSlot(representative.inner_id, slot_map.AllocateSlot());
    }
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        if (group_rows[i] != i) {
            slot_map.PublishSlot(inner_ids[i], slot_map.Resolve(inner_ids[group_rows[i]]));
        }
    }
}

}  // namespace vsag
