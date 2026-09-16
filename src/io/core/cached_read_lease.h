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

#pragma once

#include <utility>
#include <variant>

#include "io/core/read_lease.h"

namespace vsag {

template <typename BackendLease>
class CachedReadLease {
public:
    CachedReadLease() : storage_(std::in_place_index<0>) {
    }

    explicit CachedReadLease(BackendLease lease)
        : data_(lease.Data()),
          size_(lease.Size()),
          storage_(std::in_place_index<0>, std::move(lease)) {
    }

    explicit CachedReadLease(AllocatorLease lease)
        : data_(lease.Data()),
          size_(lease.Size()),
          storage_(std::in_place_index<1>, std::move(lease)) {
    }

    CachedReadLease(const CachedReadLease&) = delete;
    CachedReadLease&
    operator=(const CachedReadLease&) = delete;

    CachedReadLease(CachedReadLease&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)),
          storage_(std::move(other.storage_)) {
    }

    CachedReadLease&
    operator=(CachedReadLease&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    [[nodiscard]] explicit operator bool() const {
        return data_ != nullptr;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return data_;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_;
    }

private:
    const uint8_t* data_{nullptr};
    uint64_t size_{0};
    std::variant<BackendLease, AllocatorLease> storage_;
};

}  // namespace vsag
