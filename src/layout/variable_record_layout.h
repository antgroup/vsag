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
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "layout/byte_range_layout.h"
#include "layout/fixed_layout.h"
#include "vsag_exception.h"

namespace vsag {

/** Default location policy for records whose offset and length are persisted. */
struct OffsetAndLengthLocationPolicy {
#pragma pack(push, 1)
    struct Entry {
        uint64_t offset{0};
        uint32_t length{0};
    };
#pragma pack(pop)

    static uint64_t
    GetOffset(const Entry& entry) {
        return entry.offset;
    }

    template <typename PayloadLayout>
    static uint64_t
    GetLength(const Entry& entry, const PayloadLayout&) {
        return entry.length;
    }

    static Entry
    Make(uint64_t offset, uint64_t length) {
        if (length > std::numeric_limits<uint32_t>::max()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "variable record length exceeds uint32_t limit");
        }
        return {offset, static_cast<uint32_t>(length)};
    }
};

/** Location policy for records whose byte length is derived from a uint32 header. */
struct HeaderLengthLocationPolicy {
    using Entry = uint64_t;

    uint64_t bytes_per_element{0};

    static uint64_t
    GetOffset(const Entry& entry) {
        return entry;
    }

    template <typename PayloadLayout>
    uint64_t
    GetLength(const Entry& entry, const PayloadLayout& payload) const {
        if (bytes_per_element == 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "variable record bytes per element must be positive");
        }
        uint32_t element_count = 0;
        if (not payload.Read(
                entry, sizeof(element_count), reinterpret_cast<uint8_t*>(&element_count))) {
            throw VsagException(ErrorType::READ_ERROR,
                                "failed to read variable record length header");
        }
        if (element_count >
            (std::numeric_limits<uint64_t>::max() - sizeof(element_count)) / bytes_per_element) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "variable record header length overflow");
        }
        return sizeof(element_count) + static_cast<uint64_t>(element_count) * bytes_per_element;
    }

    static Entry
    Make(uint64_t offset, uint64_t) {
        return offset;
    }
};

/** Maps a logical record ID to an opaque variable-length payload. */
template <typename LocationPolicy, typename LocationIO, typename PayloadIO>
class VariableRecordLayout {
public:
    using LocationEntry = typename LocationPolicy::Entry;
    using LocationLayout = FixedLayout<LocationIO>;
    using PayloadLayout = ByteRangeLayout<PayloadIO>;

    VariableRecordLayout() = default;

    VariableRecordLayout(std::shared_ptr<BasicIO<LocationIO>> location_io,
                         std::shared_ptr<BasicIO<PayloadIO>> payload_io) {
        SetIO(std::move(location_io), std::move(payload_io));
    }

    void
    SetIO(std::shared_ptr<BasicIO<LocationIO>> location_io,
          std::shared_ptr<BasicIO<PayloadIO>> payload_io) {
        locations_.SetCodeSize(sizeof(LocationEntry));
        locations_.SetIO(std::move(location_io));
        payload_.SetIO(std::move(payload_io));
    }

    void
    SetLocationPolicy(LocationPolicy policy) {
        location_policy_ = std::move(policy);
    }

    void
    Write(InnerIdType id, const uint8_t* data, uint64_t length) {
        std::lock_guard lock(append_mutex_);
        const uint64_t offset = next_offset_;
        if (length > std::numeric_limits<uint64_t>::max() - offset) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "variable record payload offset overflow");
        }
        const uint64_t required_size = offset + length;
        const auto location = location_policy_.Make(offset, length);
        if (required_size > payload_.GetByteSize()) {
            payload_.Resize(required_size);
        }
        payload_.Write(offset, data, length);
        next_offset_ = required_size;
        locations_.Write(id, reinterpret_cast<const uint8_t*>(&location));
    }

    [[nodiscard]] LocationEntry
    ReadLocation(InnerIdType id) const {
        LocationEntry location{};
        if (not locations_.Read(id, reinterpret_cast<uint8_t*>(&location))) {
            throw VsagException(ErrorType::READ_ERROR, "failed to read variable record location");
        }
        return location;
    }

    [[nodiscard]] uint64_t
    GetRecordLength(const LocationEntry& location) const {
        return location_policy_.GetLength(location, payload_);
    }

    [[nodiscard]] bool
    IsValidLocation(const LocationEntry& location) const {
        uint64_t offset = 0;
        uint64_t length = 0;
        return TryResolveRange(location, offset, length);
    }

    [[nodiscard]] const uint8_t*
    Read(InnerIdType id, bool& need_release) const {
        const auto location = ReadLocation(id);
        return Read(location, need_release);
    }

    [[nodiscard]] const uint8_t*
    Read(const LocationEntry& location, bool& need_release) const {
        uint64_t offset = 0;
        uint64_t length = 0;
        if (not TryResolveRange(location, offset, length)) {
            need_release = false;
            return nullptr;
        }
        return payload_.Read(offset, length, need_release);
    }

    bool
    MultiRead(const LocationEntry* locations,
              uint64_t count,
              uint8_t* data,
              Allocator* allocator) const {
        if (count > 0 && locations == nullptr) {
            return false;
        }
        Vector<uint64_t> offsets(count, allocator);
        Vector<uint64_t> lengths(count, allocator);
        for (uint64_t i = 0; i < count; ++i) {
            if (not TryResolveRange(locations[i], offsets[i], lengths[i])) {
                return false;
            }
        }
        return payload_.MultiRead(offsets.data(), lengths.data(), count, data);
    }

    void
    Release(const uint8_t* data) const {
        payload_.Release(data);
    }

    void
    ResizeLocations(uint64_t capacity) {
        locations_.Resize(capacity);
    }

    void
    ReservePayload(uint64_t byte_size) {
        if (byte_size > payload_.GetByteSize()) {
            payload_.Resize(byte_size);
        }
    }

    void
    SetNextOffset(uint64_t offset) {
        next_offset_ = offset;
    }

    [[nodiscard]] uint64_t
    GetNextOffset() const {
        return next_offset_;
    }

    LocationLayout&
    Locations() {
        return locations_;
    }

    PayloadLayout&
    Payload() {
        return payload_;
    }

    [[nodiscard]] const PayloadLayout&
    Payload() const {
        return payload_;
    }

    [[nodiscard]] const LocationLayout&
    Locations() const {
        return locations_;
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        return locations_.GetMemoryUsage() + payload_.GetMemoryUsage();
    }

private:
    [[nodiscard]] bool
    TryResolveRange(const LocationEntry& location, uint64_t& offset, uint64_t& length) const {
        try {
            offset = location_policy_.GetOffset(location);
            length = GetRecordLength(location);
            const uint64_t payload_size = payload_.GetByteSize();
            return offset <= payload_size && length <= payload_size - offset;
        } catch (...) {
            return false;
        }
    }

    LocationLayout locations_{};
    PayloadLayout payload_{};
    uint64_t next_offset_{0};
    LocationPolicy location_policy_{};
    std::mutex append_mutex_;
};

}  // namespace vsag
