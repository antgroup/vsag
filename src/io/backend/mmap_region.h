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
    static constexpr bool CanResizeForOverwrite = true;
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

    /**
     * @brief Grows the mapping like ResizePhysical, for an extent the caller
     * overwrites in full.
     *
     * In addition to growing the file it reserves the blocks up front, so a
     * later concurrent store cannot hit SIGBUS once the filesystem is full and
     * the extent is allocated in one step instead of through scattered page
     * faults. The kernel still zero-fills each page on its first write fault;
     * that cost is inherent to a file-backed mapping and is not removed here.
     *
     * The reservation uses Linux fallocate(2); macOS has no equivalent
     * (IOSyscall::Fallocate reports ENOTSUP there), so the pre-allocation
     * benefit is Linux-only and growth proceeds without the reservation.
     *
     * @param size The new size of the mapped region.
     * @param previous_logical_size Unused; freshly mapped file pages read as
     *        zero, so there is no recycled content to mask.
     */
    void
    ResizePhysicalForOverwrite(uint64_t size, uint64_t previous_logical_size);

    void
    ShrinkPhysical(uint64_t size);

    void
    Prefetch(uint64_t offset, uint64_t size);

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
