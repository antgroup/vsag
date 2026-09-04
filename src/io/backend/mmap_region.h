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

#include <cstdint>
#include <string>

#include "vsag/allocator.h"

namespace vsag {

class MMapRegion {
public:
    static constexpr bool InMemory = false;
    static constexpr uint64_t MINIMUM_MAPPING_SIZE = 4096;

    MMapRegion(std::string filename, Allocator* allocator);
    ~MMapRegion();

    MMapRegion(const MMapRegion&) = delete;
    MMapRegion&
    operator=(const MMapRegion&) = delete;

    [[nodiscard]] uint8_t*
    Data() {
        return mapped_data_;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return mapped_data_;
    }

    [[nodiscard]] uint64_t
    Capacity() const {
        return mapped_capacity_;
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return initial_logical_size_;
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
    EnsureCapacity(uint64_t size);

    void
    ResizePhysical(uint64_t size);

    void
    ShrinkPhysical(uint64_t size);

private:
    void
    Remap(uint64_t mapped_size);

    void
    CleanupFailedConstruction() noexcept;

    Allocator* allocator_{nullptr};
    std::string filepath_;
    int fd_{-1};
    uint8_t* mapped_data_{nullptr};
    uint64_t mapped_capacity_{0};
    uint64_t file_size_{0};
    uint64_t initial_logical_size_{0};
    bool existed_{false};
};

}  // namespace vsag
