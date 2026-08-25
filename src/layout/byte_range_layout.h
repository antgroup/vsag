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
#include <utility>

#include "index_common_param.h"
#include "io/common/basic_io.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

/** Organizes opaque bytes addressed by an explicit byte offset and length. */
template <typename IOTmpl>
class ByteRangeLayout {
public:
    using IOType = IOTmpl;
    static constexpr bool InMemory = IOTmpl::InMemory;

    ByteRangeLayout() = default;

    ByteRangeLayout(const IOParamPtr& io_param, const IndexCommonParam& common_param)
        : io_(std::make_shared<IOTmpl>(io_param, common_param)) {
    }

    void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        io_ = std::move(io);
    }

    void
    Write(uint64_t offset, const uint8_t* data, uint64_t length) {
        if (length > std::numeric_limits<uint64_t>::max() - offset) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "byte range write offset overflow");
        }
        if (not IsValidRange(offset, length)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "byte range write exceeds allocated size");
        }
        io_->Write(data, length, offset);
    }

    bool
    Read(uint64_t offset, uint64_t length, uint8_t* data) const {
        if (not IsValidRange(offset, length)) {
            return false;
        }
        return io_->Read(length, offset, data);
    }

    [[nodiscard]] const uint8_t*
    Read(uint64_t offset, uint64_t length, bool& need_release) const {
        if (not IsValidRange(offset, length)) {
            need_release = false;
            return nullptr;
        }
        return io_->Read(length, offset, need_release);
    }

    bool
    MultiRead(uint64_t* offsets, uint64_t* lengths, uint64_t count, uint8_t* data) const {
        if (count > 0 && (offsets == nullptr || lengths == nullptr)) {
            return false;
        }
        uint64_t total_length = 0;
        for (uint64_t i = 0; i < count; ++i) {
            if (not IsValidRange(offsets[i], lengths[i]) ||
                lengths[i] > std::numeric_limits<uint64_t>::max() - total_length) {
                return false;
            }
            total_length += lengths[i];
        }
        if (total_length > 0 && data == nullptr) {
            return false;
        }
        return io_->MultiRead(data, lengths, offsets, count);
    }

    void
    Release(const uint8_t* data) const {
        if (data != nullptr) {
            io_->Release(data);
        }
    }

    void
    Prefetch(uint64_t offset, uint64_t length) {
        io_->Prefetch(offset, length);
    }

    void
    Resize(uint64_t byte_size) {
        io_->Resize(byte_size);
    }

    void
    Shrink(uint64_t byte_size) {
        io_->Shrink(byte_size);
    }

    void
    InitIO(const IOParamPtr& io_param) {
        io_->InitIO(io_param);
    }

    void
    Serialize(StreamWriter& writer) {
        io_->Serialize(writer);
    }

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) {
        io_->Deserialize(reader);
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        if constexpr (InMemory) {
            const int64_t memory_usage = io_->GetMemoryUsage();
            return memory_usage > 0 ? static_cast<uint64_t>(memory_usage) : 0;
        }
        return 0;
    }

    [[nodiscard]] uint64_t
    GetByteSize() const {
        return io_->size_;
    }

private:
    [[nodiscard]] bool
    IsValidRange(uint64_t offset, uint64_t length) const {
        const uint64_t byte_size = GetByteSize();
        return offset <= byte_size && length <= byte_size - offset;
    }

    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};
};

}  // namespace vsag
