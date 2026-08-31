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
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "common.h"
#include "index_common_param.h"
#include "io/common/basic_io.h"
#include "io/memory_io/memory_io.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

/**
 * A fixed-size record layout backed by a concrete CRTP IO type.
 *
 * FixedLayout maps a logical record ID to the byte range
 * `[id * code_size, (id + 1) * code_size)`. It only organizes and transports
 * opaque bytes; interpretation of those bytes remains in the DataCell.
 */
template <typename IOTmpl>
class FixedLayout {
public:
    using IOType = IOTmpl;
    static constexpr bool InMemory = IOTmpl::InMemory;

    FixedLayout() = default;

    FixedLayout(const IOParamPtr& io_param, const IndexCommonParam& common_param)
        : FixedLayout(0, io_param, common_param) {
    }

    FixedLayout(uint64_t code_size,
                const IOParamPtr& io_param,
                const IndexCommonParam& common_param)
        : code_size_(code_size), io_(std::make_shared<IOTmpl>(io_param, common_param)) {
    }

    [[nodiscard]] uint64_t
    GetCodeSize() const {
        return code_size_;
    }

    void
    SetCodeSize(uint64_t code_size) {
        code_size_ = code_size;
    }

    void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        io_ = std::move(io);
    }

    void
    Write(InnerIdType id, const uint8_t* code) {
        io_->Write(code, code_size_, GetOffset(id));
    }

    void
    WriteRange(InnerIdType begin_id, const uint8_t* codes, uint64_t count) {
        io_->Write(codes, GetByteSize(count), GetOffset(begin_id));
    }

    bool
    Read(InnerIdType id, uint8_t* code) const {
        return io_->Read(code_size_, GetOffset(id), code);
    }

    [[nodiscard]] const uint8_t*
    Read(InnerIdType id, bool& need_release) const {
        return io_->Read(code_size_, GetOffset(id), need_release);
    }

    [[nodiscard]] const uint8_t*
    ReadRange(InnerIdType begin_id, uint64_t count, bool& need_release) const {
        return io_->Read(GetByteSize(count), GetOffset(begin_id), need_release);
    }

    bool
    MultiRead(const InnerIdType* ids, uint64_t count, uint8_t* codes, Allocator* allocator) const {
        Vector<uint64_t> sizes(count, code_size_, allocator);
        Vector<uint64_t> offsets(count, 0, allocator);
        for (uint64_t i = 0; i < count; ++i) {
            offsets[i] = GetOffset(ids[i]);
        }
        return io_->MultiRead(codes, sizes.data(), offsets.data(), count);
    }

    void
    Release(const uint8_t* code) const {
        if (code != nullptr) {
            io_->Release(code);
        }
    }

    void
    Prefetch(InnerIdType id, uint64_t bytes) {
        io_->Prefetch(GetOffset(id), bytes);
    }

    void
    Resize(uint64_t capacity) {
        io_->Resize(GetByteSize(capacity));
    }

    void
    Shrink(uint64_t capacity) {
        io_->Shrink(GetByteSize(capacity));
    }

    void
    Move(InnerIdType from, InnerIdType to) {
        bool need_release = false;
        const auto* code = Read(from, need_release);
        if (code == nullptr) {
            throw VsagException(ErrorType::READ_ERROR, "failed to read fixed layout record");
        }
        try {
            Write(to, code);
        } catch (...) {
            if (need_release and code != nullptr) {
                Release(code);
            }
            throw;
        }
        if (need_release and code != nullptr) {
            Release(code);
        }
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
            return static_cast<uint64_t>(io_->GetMemoryUsage());
        }
        return 0;
    }

    [[nodiscard]] const uint8_t*
    TryGetContiguousData() const {
        if constexpr (std::is_same_v<IOTmpl, MemoryIO>) {
            return std::static_pointer_cast<MemoryIO>(io_)->GetReadOnlyRawData();
        }
        return nullptr;
    }

private:
    [[nodiscard]] uint64_t
    GetOffset(InnerIdType id) const {
        return GetByteSize(static_cast<uint64_t>(id));
    }

    [[nodiscard]] uint64_t
    GetByteSize(uint64_t count) const {
        if (code_size_ != 0 and count > std::numeric_limits<uint64_t>::max() / code_size_) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "fixed layout byte size overflow");
        }
        return count * code_size_;
    }

    uint64_t code_size_{0};
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};
};

}  // namespace vsag
