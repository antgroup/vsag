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
#include <memory>
#include <vector>

#include "io/core/io_utils.h"
#include "io/core/read_lease.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"
#include "io/reader_io/reader_io_parameter.h"
#include "vsag/allocator.h"
#include "vsag/readerset.h"
#include "vsag_exception.h"

namespace vsag {

struct ExternalReaderBackendCapabilities {
    static constexpr bool InMemory = false;
    static constexpr bool RequiresInitialization = true;
    static constexpr bool CanBindSerializedRange = true;
    static constexpr bool LegacyBatchRangeThrows = true;
    static constexpr bool LegacyUncheckedReadable = false;
    static constexpr bool BorrowedReadable = false;
    static constexpr bool BatchReadable = true;
    static constexpr bool AsyncReadable = true;
    static constexpr bool Writable = false;
    static constexpr bool Resizable = false;
};

class ExternalReaderBackend {
public:
    using Capabilities = ExternalReaderBackendCapabilities;
    using Lease = AllocatorLease;
    using Operation = ImmediateOperation;

    explicit ExternalReaderBackend(Allocator* allocator) : allocator_(allocator) {
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return 0;
    }

    [[nodiscard]] uint64_t
    Initialize(const IOParamPtr& io_param,
               bool has_deserialized,
               uint64_t serialized_start,
               uint64_t current_logical_size) {
        auto reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(io_param);
        if (reader_param == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "ReaderIOParam is required for ReaderIO initialization");
        }
        if (reader_param->reader == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "ReaderIO requires a non-null reader");
        }
        reader_ = reader_param->reader;
        if (has_deserialized) {
            BindSerializedRange(serialized_start, current_logical_size);
            ValidateBoundRange();
            return current_logical_size;
        }
        base_offset_ = 0;
        logical_size_ = reader_->Size();
        return logical_size_;
    }

    void
    BindSerializedRange(uint64_t start, uint64_t size) {
        base_offset_ = start;
        logical_size_ = size;
        if (reader_ != nullptr) {
            ValidateBoundRange();
        }
    }

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        EnsureInitialized();
        reader_->Read(CheckedEnd(base_offset_, offset), size, destination);
        return true;
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        EnsureInitialized();
        if (count == 0) {
            return true;
        }
        std::vector<uint64_t> sizes;
        std::vector<uint64_t> offsets;
        sizes.reserve(count);
        offsets.reserve(count);
        uint8_t* first_destination = nullptr;
        uint8_t* expected_destination = nullptr;
        bool contiguous = true;
        for (uint64_t i = 0; i < count; ++i) {
            if (requests[i].size == 0) {
                continue;
            }
            if (first_destination == nullptr) {
                first_destination = requests[i].destination;
                expected_destination = first_destination;
            }
            contiguous &= requests[i].destination == expected_destination;
            expected_destination += requests[i].size;
            sizes.emplace_back(requests[i].size);
            offsets.emplace_back(CheckedEnd(base_offset_, requests[i].offset));
        }
        if (sizes.empty()) {
            return true;
        }
        if (contiguous) {
            return reader_->MultiRead(
                first_destination, sizes.data(), offsets.data(), sizes.size());
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (requests[i].size > 0) {
                reader_->Read(CheckedEnd(base_offset_, requests[i].offset),
                              requests[i].size,
                              requests[i].destination);
            }
        }
        return true;
    }

    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        EnsureInitialized();
        std::vector<uint64_t> non_empty_sizes;
        std::vector<uint64_t> real_offsets;
        non_empty_sizes.reserve(count);
        real_offsets.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            if (sizes[i] > 0) {
                non_empty_sizes.emplace_back(sizes[i]);
                real_offsets.emplace_back(CheckedEnd(base_offset_, offsets[i]));
            }
        }
        if (non_empty_sizes.empty()) {
            return true;
        }
        return reader_->MultiRead(
            destination, non_empty_sizes.data(), real_offsets.data(), non_empty_sizes.size());
    }

    [[nodiscard]] Operation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        return Operation(ReadMany(requests, count));
    }

    [[nodiscard]] Lease
    Acquire(uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return {};
        }
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "ExternalReaderBackend allocation failed");
        }
        AllocatorOwner owner(allocator_, data);
        static_cast<void>(ReadAt(offset, size, data));
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
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "ExternalReaderBackend legacy allocation failed");
        }
        try {
            static_cast<void>(ReadAt(offset, size, data));
        } catch (...) {
            allocator_->Deallocate(data);
            throw;
        }
        need_release = true;
        return data;
    }

    void
    Release(const uint8_t* data) const {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    void
    WriteAt(uint64_t, const uint8_t*, uint64_t) {
    }

    void
    ResizePhysical(uint64_t) {
    }

    void
    ShrinkPhysical(uint64_t) {
    }

    void
    Prefetch(uint64_t, uint64_t) {
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t logical_size) const {
        return logical_size;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return nullptr;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

private:
    void
    EnsureInitialized() const {
        if (reader_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "ReaderIO is not initialized, please call InitIO first");
        }
    }

    void
    ValidateBoundRange() const {
        if (base_offset_ > reader_->Size() or logical_size_ > reader_->Size() - base_offset_) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "ReaderIO serialized range exceeds reader size");
        }
    }

    Allocator* allocator_{nullptr};
    std::shared_ptr<Reader> reader_;
    uint64_t base_offset_{0};
    uint64_t logical_size_{0};
};

}  // namespace vsag
