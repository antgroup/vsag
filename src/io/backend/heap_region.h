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

#include <algorithm>
#include <cstdint>

#include "vsag/allocator.h"
#include "vsag_exception.h"

namespace vsag {

class HeapRegion {
public:
    static constexpr bool InMemory = true;

    explicit HeapRegion(Allocator* allocator) : allocator_(allocator) {
        data_ = static_cast<uint8_t*>(allocator_->Allocate(1));
        if (data_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "failed to allocate initial HeapRegion buffer");
        }
        capacity_ = 1;
    }

    ~HeapRegion() {
        if (data_ != nullptr) {
            allocator_->Deallocate(data_);
        }
    }

    HeapRegion(const HeapRegion&) = delete;
    HeapRegion&
    operator=(const HeapRegion&) = delete;

    HeapRegion(HeapRegion&& other) noexcept
        : allocator_(other.allocator_), data_(other.data_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }

    HeapRegion&
    operator=(HeapRegion&&) = delete;

    [[nodiscard]] uint8_t*
    Data() {
        return data_;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return data_;
    }

    [[nodiscard]] uint64_t
    Capacity() const {
        return capacity_;
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return 0;
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t logical_size) const {
        return static_cast<int64_t>(logical_size);
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

    void
    EnsureCapacity(uint64_t size) {
        if (size <= capacity_) {
            return;
        }
        auto* new_data = static_cast<uint8_t*>(allocator_->Reallocate(data_, size));
        if (new_data == nullptr) {
            throw VsagException(
                ErrorType::NO_ENOUGH_MEMORY, "failed to grow HeapRegion, requested size: ", size);
        }
        data_ = new_data;
        capacity_ = size;
    }

    void
    ResizePhysical(uint64_t size) {
        EnsureCapacity(size);
    }

    void
    ShrinkPhysical(uint64_t size) {
        uint64_t allocation_size = std::max<uint64_t>(size, 1);
        if (allocation_size >= capacity_) {
            return;
        }
        auto* new_data = static_cast<uint8_t*>(allocator_->Reallocate(data_, allocation_size));
        if (new_data == nullptr) {
            throw VsagException(
                ErrorType::NO_ENOUGH_MEMORY, "failed to shrink HeapRegion, requested size: ", size);
        }
        data_ = new_data;
        capacity_ = allocation_size;
    }

private:
    Allocator* allocator_{nullptr};
    uint8_t* data_{nullptr};
    uint64_t capacity_{0};
};

}  // namespace vsag
