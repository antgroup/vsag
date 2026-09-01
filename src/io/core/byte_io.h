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
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "io/common/io_parameter.h"
#include "io/core/io_utils.h"
#include "io/core/read_request.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/byte_buffer.h"

namespace vsag {

class PageCache;

template <typename Backend, typename CachePolicy>
class ByteIO {
public:
    using BackendType = Backend;
    using CacheType = CachePolicy;
    using Lease = typename CachePolicy::template Lease<Backend>;
    using Operation = typename CachePolicy::template Operation<Backend>;

    static constexpr bool InMemory = Backend::Capabilities::InMemory;
    static constexpr bool SkipDeserialize = Backend::Capabilities::CanBindSerializedRange;

    template <typename... Args>
    explicit ByteIO(Args&&... args) : backend_(std::forward<Args>(args)...), cache_() {
        size_.store(backend_.InitialLogicalSize(), std::memory_order_relaxed);
    }

    ByteIO(const ByteIO&) = delete;
    ByteIO&
    operator=(const ByteIO&) = delete;

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        const auto published_size = Size();
        if (not IsValidRange(offset, size, published_size)) {
            return false;
        }
        if (size == 0) {
            return true;
        }
        if (destination == nullptr) {
            return false;
        }
        return cache_.ReadAt(backend_, published_size, ReadRequest{destination, offset, size});
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        if (count == 0) {
            return true;
        }
        if (requests == nullptr) {
            return false;
        }
        const auto published_size = Size();
        for (uint64_t i = 0; i < count; ++i) {
            if (not IsValidRange(requests[i].offset, requests[i].size, published_size)) {
                return false;
            }
            if (requests[i].size > 0 and requests[i].destination == nullptr) {
                return false;
            }
        }
        return cache_.ReadMany(backend_, published_size, requests, count);
    }

    /**
     * Executes a batch whose ranges and destinations were already validated by a composing IO.
     * This avoids repeating per-request checks in trusted adapters such as NonContinuousIO.
     */
    [[nodiscard]] bool
    ReadManyPrevalidated(const ReadRequest* requests, uint64_t count) const {
        return cache_.ReadMany(backend_, Size(), requests, count);
    }

    [[nodiscard]] Operation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        const auto published_size = Size();
        if (not ValidateRequests(requests, count, published_size)) {
            return Operation(false);
        }
        return cache_.SubmitReads(backend_, published_size, requests, count);
    }

    [[nodiscard]] Lease
    Acquire(uint64_t offset, uint64_t size) const {
        const auto published_size = Size();
        if (not IsValidRange(offset, size, published_size)) {
            return Lease{};
        }
        return cache_.Acquire(backend_, published_size, offset, size);
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        if (size > 0 and source == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "IO write source is null");
        }
        uint64_t end = CheckedEnd(offset, size);
        backend_.WriteAt(offset, source, size);
        PublishSize(end);
        cache_.Invalidate(backend_, offset, size);
    }

    void
    Resize(uint64_t size) {
        uint64_t old_size = Size();
        backend_.ResizePhysical(size);
        size_.store(size, std::memory_order_release);
        cache_.Clear(old_size);
    }

    void
    Shrink(uint64_t size) {
        uint64_t old_size = Size();
        if (size >= old_size) {
            return;
        }
        backend_.ShrinkPhysical(size);
        size_.store(size, std::memory_order_release);
        cache_.Clear(old_size);
    }

    void
    Prefetch(uint64_t offset, uint64_t cache_line = 64) {
        if (offset < Size()) {
            backend_.Prefetch(offset, cache_line);
        }
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_.load(std::memory_order_acquire);
    }

    [[nodiscard]] int64_t
    MemoryUsage() const {
        return backend_.MemoryUsage(Size());
    }

    void
    Serialize(StreamWriter& writer) const {
        const auto size = Size();
        StreamWriter::WriteObj(writer, size);
        ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, AllocatorPtr());
        uint64_t offset = 0;
        while (offset < size) {
            uint64_t current_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
            if (not ReadAt(offset, current_size, buffer.data)) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read valid IO range during serialization");
            }
            writer.Write(reinterpret_cast<const char*>(buffer.data), current_size);
            offset += current_size;
        }
    }

    void
    Deserialize(StreamReader& reader) {
        uint64_t size = 0;
        StreamReader::ReadObj(reader, size);
        const uint64_t serialized_start = reader.GetCursor();
        if constexpr (SkipDeserialize) {
            const uint64_t serialized_end = CheckedEnd(serialized_start, size);
            uint64_t old_size = Size();
            reader.Seek(serialized_end);
            backend_.BindSerializedRange(serialized_start, size);
            start_ = serialized_start;
            has_deserialized_ = true;
            size_.store(size, std::memory_order_release);
            cache_.Clear(old_size);
            return;
        }
        start_ = serialized_start;
        has_deserialized_ = true;
        Resize(size);
        ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, AllocatorPtr());
        uint64_t offset = 0;
        while (offset < size) {
            uint64_t current_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
            reader.Read(reinterpret_cast<char*>(buffer.data), current_size);
            WriteAt(offset, buffer.data, current_size);
            offset += current_size;
        }
    }

    void
    Write(const uint8_t* data, uint64_t size, uint64_t offset) {
        WriteAt(offset, data, size);
    }

    bool
    Read(uint64_t size, uint64_t offset, uint8_t* data) const {
        // BufferIO's historical copy API delegated bounds handling to pread. Preserve that
        // behavior when no cache is installed; ReadAt remains the range-checked canonical API.
        if constexpr (Backend::Capabilities::LegacyUncheckedReadable) {
            if (not cache_.Enabled()) {
                if (size == 0) {
                    return true;
                }
                if (data == nullptr) {
                    return false;
                }
                return backend_.ReadAt(offset, size, data);
            }
        }
        return ReadAt(offset, size, data);
    }

    [[nodiscard]] const uint8_t*
    Read(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = false;
        const auto published_size = Size();
        if (not IsValidRange(offset, size, published_size)) {
            return nullptr;
        }
        return cache_.LegacyRead(backend_, published_size, offset, size, need_release);
    }

    bool
    MultiRead(uint8_t* data, const uint64_t* sizes, const uint64_t* offsets, uint64_t count) const {
        if (count == 0) {
            return true;
        }
        if (sizes == nullptr or offsets == nullptr) {
            return false;
        }
        if constexpr (Backend::Capabilities::LegacyUncheckedReadable) {
            if (not cache_.Enabled()) {
                if (data == nullptr) {
                    bool has_data = false;
                    for (uint64_t i = 0; i < count; ++i) {
                        has_data = has_data or sizes[i] > 0;
                    }
                    if (has_data) {
                        return false;
                    }
                    return true;
                }
                return backend_.ReadManyContiguous(data, sizes, offsets, count);
            }
        }
        const auto published_size = Size();
        uint64_t total_size = 0;
        for (uint64_t i = 0; i < count; ++i) {
            if (not IsValidRange(offsets[i], sizes[i], published_size)) {
                if constexpr (Backend::Capabilities::LegacyBatchRangeThrows) {
                    if (not cache_.Enabled()) {
                        throw VsagException(ErrorType::INTERNAL_ERROR,
                                            "Reader MultiRead size mismatch");
                    }
                }
                return false;
            }
            total_size = CheckedEnd(total_size, sizes[i]);
        }
        if (total_size > 0 and data == nullptr) {
            return false;
        }
        return cache_.ReadManyContiguous(backend_, published_size, data, sizes, offsets, count);
    }

    void
    Release(const uint8_t* data) const {
        cache_.Release(backend_, data);
    }

    void
    InitIO(const IOParamPtr& io_param) {
        const auto old_size = Size();
        cache_.Clear(old_size);
        cache_.Configure(io_param);
        if constexpr (Backend::Capabilities::RequiresInitialization) {
            const auto initialized_size =
                backend_.Initialize(io_param, has_deserialized_, start_, old_size);
            size_.store(initialized_size, std::memory_order_release);
        }
    }

    void
    EnableReadCache(const IOParamPtr& io_param) {
        cache_.Configure(io_param);
    }

    void
    SetReadCache(const std::shared_ptr<PageCache>& cache, uint64_t page_id_base = 0) {
        cache_.SetShared(cache, page_id_base);
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const {
        return MemoryUsage();
    }

    [[nodiscard]] bool
    HasDeserialized() const {
        return has_deserialized_;
    }

    [[nodiscard]] const uint8_t*
    GetReadOnlyRawData() const {
        return backend_.Data();
    }

private:
    void
    PublishSize(uint64_t size) {
        auto current = size_.load(std::memory_order_relaxed);
        while (current < size and
               not size_.compare_exchange_weak(
                   current, size, std::memory_order_release, std::memory_order_relaxed)) {
        }
    }

    [[nodiscard]] bool
    ValidateRequests(const ReadRequest* requests, uint64_t count, uint64_t published_size) const {
        if (count == 0) {
            return true;
        }
        if (requests == nullptr) {
            return false;
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (not IsValidRange(requests[i].offset, requests[i].size, published_size)) {
                return false;
            }
            if (requests[i].size > 0 and requests[i].destination == nullptr) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return backend_.AllocatorPtr();
    }

    static constexpr uint64_t SERIALIZE_BUFFER_SIZE = 2 * 1024 * 1024;

    Backend backend_;
    CachePolicy cache_;
    // Writers publish newly initialized storage through size_; readers acquire that publication
    // before validating offsets. Some in-memory IO implementations append concurrently with reads.
    std::atomic<uint64_t> size_{0};
    uint64_t start_{0};
    bool has_deserialized_{false};
};

}  // namespace vsag
