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
#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "container_types.h"
#include "io/common/io_parameter.h"
#include "io/core/read_lease.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"
#include "io/noncontinuous_io/noncontinuous_allocator.h"
#include "vsag/allocator.h"

namespace vsag {

namespace noncontinuous_backend_detail {

template <typename IO, typename = void>
struct HasScatterRead : std::false_type {};

template <typename IO>
struct HasScatterRead<IO,
                      std::void_t<decltype(std::declval<const IO&>().ReadMany(
                          std::declval<const ReadRequest*>(), std::declval<uint64_t>()))>>
    : std::true_type {};

template <typename IO, typename = void>
struct HasPrevalidatedScatterRead : std::false_type {};

template <typename IO>
struct HasPrevalidatedScatterRead<
    IO,
    std::void_t<decltype(std::declval<const IO&>().ReadManyPrevalidated(
        std::declval<const ReadRequest*>(), std::declval<uint64_t>()))>> : std::true_type {};

class ReadRequestBuffer {
public:
    explicit ReadRequestBuffer(Allocator* allocator, uint64_t expected_requests)
        : overflow_(allocator) {
        if (expected_requests > INLINE_CAPACITY) {
            UseOverflow(expected_requests);
        }
    }

    void
    Emplace(uint8_t* destination, uint64_t offset, uint64_t size) {
        if (using_overflow_) {
            overflow_.emplace_back(ReadRequest{destination, offset, size});
            return;
        }
        if (size_ < INLINE_CAPACITY) {
            inline_[size_++] = ReadRequest{destination, offset, size};
            return;
        }
        UseOverflow(INLINE_CAPACITY * 2);
        overflow_.emplace_back(ReadRequest{destination, offset, size});
    }

    [[nodiscard]] const ReadRequest*
    Data() const {
        return using_overflow_ ? overflow_.data() : inline_.data();
    }

    [[nodiscard]] uint64_t
    Size() const {
        return using_overflow_ ? static_cast<uint64_t>(overflow_.size()) : size_;
    }

private:
    void
    UseOverflow(uint64_t capacity) {
        overflow_.reserve(capacity);
        overflow_.insert(overflow_.end(), inline_.begin(), inline_.begin() + size_);
        using_overflow_ = true;
    }

    static constexpr uint64_t INLINE_CAPACITY = 128;
    std::array<ReadRequest, INLINE_CAPACITY> inline_{};
    Vector<ReadRequest> overflow_;
    uint64_t size_{0};
    bool using_overflow_{false};
};

}  // namespace noncontinuous_backend_detail

template <typename PhysicalIO>
struct NonContinuousBackendCapabilities {
    static constexpr bool InMemory = PhysicalIO::InMemory;
    static constexpr bool RequiresInitialization = true;
    static constexpr bool CanBindSerializedRange = false;
    static constexpr bool LegacyBatchRangeThrows = false;
    static constexpr bool LegacyUncheckedReadable = false;
    static constexpr bool BorrowedReadable = false;
    static constexpr bool BatchReadable = true;
    static constexpr bool AsyncReadable = false;
    static constexpr bool Writable = true;
    static constexpr bool Resizable = true;
};

/**
 * Maps a contiguous logical address space onto extents in a shared physical IO address space.
 * Query-time reads are lock-free after the caller freezes the extent table.
 */
template <typename PhysicalIO>
class NonContinuousBackend {
public:
    using Capabilities = NonContinuousBackendCapabilities<PhysicalIO>;
    using Lease = AllocatorLease;
    using Operation = ImmediateOperation;

    template <typename... Args>
    NonContinuousBackend(NonContinuousAllocator* extent_allocator,
                         Allocator* allocator,
                         Args&&... args)
        : extent_allocator_(extent_allocator),
          allocator_(allocator),
          physical_io_(std::make_unique<PhysicalIO>(std::forward<Args>(args)...)),
          extents_(allocator) {
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return 0;
    }

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        if (size == 0) {
            return true;
        }
        if constexpr (noncontinuous_backend_detail::HasScatterRead<PhysicalIO>::value) {
            ReadRequest request{destination, offset, size};
            return ReadMany(&request, 1);
        }
        Vector<uint64_t> sizes(allocator_);
        Vector<uint64_t> offsets(allocator_);
        PlanContiguous(offset, size, sizes, offsets);
        return physical_io_->MultiRead(
            destination, sizes.data(), offsets.data(), static_cast<uint64_t>(sizes.size()));
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        if constexpr (noncontinuous_backend_detail::HasScatterRead<PhysicalIO>::value) {
            noncontinuous_backend_detail::ReadRequestBuffer physical_requests(allocator_, count);
            for (uint64_t i = 0; i < count; ++i) {
                PlanScatter(requests[i], physical_requests);
            }
            if (physical_requests.Size() == 0) {
                return true;
            }
            return ReadPhysicalMany(physical_requests.Data(), physical_requests.Size());
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                if (not ReadAt(requests[i].offset, requests[i].size, requests[i].destination)) {
                    return false;
                }
            }
            return true;
        }
    }

    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        if constexpr (noncontinuous_backend_detail::HasScatterRead<PhysicalIO>::value) {
            noncontinuous_backend_detail::ReadRequestBuffer physical_requests(allocator_, count);
            for (uint64_t i = 0; i < count; ++i) {
                PlanScatter(ReadRequest{destination, offsets[i], sizes[i]}, physical_requests);
                if (sizes[i] > 0) {
                    destination += sizes[i];
                }
            }
            if (physical_requests.Size() == 0) {
                return true;
            }
            return ReadPhysicalMany(physical_requests.Data(), physical_requests.Size());
        }
        Vector<uint64_t> physical_sizes(allocator_);
        Vector<uint64_t> physical_offsets(allocator_);
        physical_sizes.reserve(count);
        physical_offsets.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            PlanContiguous(offsets[i], sizes[i], physical_sizes, physical_offsets);
        }
        if (physical_sizes.empty()) {
            return true;
        }
        return physical_io_->MultiRead(destination,
                                       physical_sizes.data(),
                                       physical_offsets.data(),
                                       static_cast<uint64_t>(physical_sizes.size()));
    }

    [[nodiscard]] Operation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        return Operation(ReadMany(requests, count));
    }

    [[nodiscard]] Lease
    Acquire(uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return Lease{};
        }
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "NonContinuousBackend acquire allocation failed");
        }
        AllocatorOwner owner(allocator_, data);
        if (not ReadAt(offset, size, data)) {
            return Lease{};
        }
        return Lease(data, size, std::move(owner));
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(uint64_t offset, uint64_t size, bool& need_release) const {
        if (size == 0) {
            need_release = false;
            return nullptr;
        }
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            need_release = false;
            return nullptr;
        }
        try {
            if (ReadAt(offset, size, data)) {
                need_release = true;
                return data;
            }
        } catch (...) {
            allocator_->Deallocate(data);
            throw;
        }
        allocator_->Deallocate(data);
        need_release = false;
        return nullptr;
    }

    void
    Release(const uint8_t* data) const {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        if (size == 0) {
            return;
        }
        EnsureLogicalCapacity(offset + size);
        auto extent = FindExtent(offset);
        uint64_t written = 0;
        while (written < size) {
            uint64_t logical_start = extent->second - extent->first.size;
            uint64_t physical_offset = extent->first.offset + offset + written - logical_start;
            uint64_t current_size = std::min(size - written, extent->second - offset - written);
            physical_io_->Write(source + written, current_size, physical_offset);
            written += current_size;
            ++extent;
        }
    }

    void
    ResizePhysical(uint64_t size) {
        EnsureLogicalCapacity(size);
    }

    void
    ShrinkPhysical(uint64_t) {
        // Extents are allocated from a shared monotonic address space and may be interleaved with
        // other users. The allocator has no release operation, so shrinking is logical only; the
        // retained extents remain available if this IO grows again.
    }

    void
    Prefetch(uint64_t offset, uint64_t cache_line) {
        auto extent = FindExtent(offset);
        uint64_t logical_start = extent->second - extent->first.size;
        physical_io_->Prefetch(extent->first.offset + offset - logical_start, cache_line);
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t logical_size) const {
        return Capabilities::InMemory ? static_cast<int64_t>(logical_size) : 0;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return nullptr;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

    uint64_t
    Initialize(const IOParamPtr& io_param, bool, uint64_t, uint64_t current_logical_size) {
        physical_io_->InitIO(io_param);
        return current_logical_size;
    }

private:
    using ExtentEntry = std::pair<NonContinuousArea, uint64_t>;

    [[nodiscard]] bool
    ReadPhysicalMany(const ReadRequest* requests, uint64_t count) const {
        if constexpr (noncontinuous_backend_detail::HasPrevalidatedScatterRead<PhysicalIO>::value) {
            return physical_io_->ReadManyPrevalidated(requests, count);
        }
        return physical_io_->ReadMany(requests, count);
    }

    [[nodiscard]] auto
    FindExtent(uint64_t logical_offset) const {
        return std::upper_bound(
            extents_.begin(),
            extents_.end(),
            logical_offset,
            [](uint64_t offset, const ExtentEntry& extent) { return offset < extent.second; });
    }

    void
    EnsureLogicalCapacity(uint64_t required_size) {
        uint64_t capacity = extents_.empty() ? 0 : extents_.back().second;
        if (required_size <= capacity) {
            return;
        }
        auto extent = extent_allocator_->Require(required_size - capacity);
        extents_.emplace_back(extent, capacity + extent.size);
    }

    void
    PlanContiguous(uint64_t logical_offset,
                   uint64_t size,
                   Vector<uint64_t>& sizes,
                   Vector<uint64_t>& offsets) const {
        if (size == 0) {
            return;
        }
        auto extent = FindExtent(logical_offset);
        uint64_t planned = 0;
        while (planned < size) {
            uint64_t logical_start = extent->second - extent->first.size;
            uint64_t current_size =
                std::min(size - planned, extent->second - logical_offset - planned);
            sizes.emplace_back(current_size);
            offsets.emplace_back(extent->first.offset + logical_offset + planned - logical_start);
            planned += current_size;
            ++extent;
        }
    }

    void
    PlanScatter(const ReadRequest& request,
                noncontinuous_backend_detail::ReadRequestBuffer& physical_requests) const {
        if (request.size == 0) {
            return;
        }
        auto extent = FindExtent(request.offset);
        uint64_t planned = 0;
        while (planned < request.size) {
            uint64_t logical_start = extent->second - extent->first.size;
            uint64_t current_size =
                std::min(request.size - planned, extent->second - request.offset - planned);
            physical_requests.Emplace(
                request.destination + planned,
                extent->first.offset + request.offset + planned - logical_start,
                current_size);
            planned += current_size;
            ++extent;
        }
    }

    NonContinuousAllocator* extent_allocator_{nullptr};
    Allocator* allocator_{nullptr};
    std::unique_ptr<PhysicalIO> physical_io_;
    Vector<ExtentEntry> extents_;
};

}  // namespace vsag
