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

#include "index_common_param.h"
#include "io/common/basic_io.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

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
        io_->Write(data, length, offset);
    }

    bool
    Read(uint64_t offset, uint64_t length, uint8_t* data) const {
        return io_->Read(length, offset, data);
    }

    [[nodiscard]] const uint8_t*
    Read(uint64_t offset, uint64_t length, bool& need_release) const {
        return io_->Read(length, offset, need_release);
    }

    bool
    MultiRead(uint64_t* offsets, uint64_t* lengths, uint64_t count, uint8_t* data) const {
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
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        io_->Deserialize(reader);
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        if constexpr (InMemory) {
            return io_->GetMemoryUsage();
        }
        return 0;
    }

    [[nodiscard]] uint64_t
    GetByteSize() const {
        return io_->size_;
    }

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};
};

}  // namespace vsag
