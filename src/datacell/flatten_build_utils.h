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
#include <numeric>
#include <vector>

#include "datacell/flatten_interface.h"
#include "impl/thread_pool/safe_thread_pool.h"

namespace vsag {

template <typename VectorAt>
inline void
ParallelBatchInsertVector(const FlattenInterfacePtr& flatten,
                          InnerIdType count,
                          InnerIdType* ids,
                          SafeThreadPool* thread_pool,
                          uint64_t thread_count,
                          VectorAt&& vector_at) {
    constexpr InnerIdType BATCH_SIZE = 4096;
    std::vector<std::future<void>> futures;
    std::exception_ptr first_exception = nullptr;
    for (InnerIdType begin = 0; begin < count; begin += BATCH_SIZE) {
        const auto batch_count = std::min<InnerIdType>(BATCH_SIZE, count - begin);
        auto insert_batch = [=]() {
            if (ids != nullptr) {
                flatten->BatchInsertVector(vector_at(begin), batch_count, ids + begin);
                return;
            }
            std::vector<InnerIdType> contiguous_ids(batch_count);
            std::iota(contiguous_ids.begin(), contiguous_ids.end(), begin);
            flatten->BatchInsertVector(vector_at(begin), batch_count, contiguous_ids.data());
        };
        if (thread_pool != nullptr and thread_count > 1) {
            try {
                futures.emplace_back(thread_pool->GeneralEnqueue(std::move(insert_batch)));
            } catch (...) {
                first_exception = std::current_exception();
                break;
            }
        } else {
            insert_batch();
        }
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (first_exception == nullptr) {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

}  // namespace vsag
