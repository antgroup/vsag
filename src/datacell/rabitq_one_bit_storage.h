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
#include <memory>

#include "basic_types.h"
#include "common.h"
#include "index_common_param_fwd.h"
#include "io/basic_io.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "type_helpers.h"

namespace vsag {

/**
 * One-bit storage component for split RaBitQ.
 *
 * Owns a single BasicIO instance whose entries are sized by the one-bit record
 * layout reported by the RaBitQuantizer (plane bytes + per-record meta such as
 * norms and error bounds). It is intentionally kept simple and IO-only so that
 * the higher-level RaBitQSplitDataCell can drive graph traversal exclusively
 * through this object during the lower-bound search phase.
 */
template <typename IOTmpl>
class RaBitQOneBitStorage {
public:
    static constexpr bool InMemory = IOTmpl::InMemory;

    RaBitQOneBitStorage(const IOParamPtr& io_param, const IndexCommonParam& common_param)
        : io_(std::make_shared<IOTmpl>(io_param, common_param)) {
    }

    void
    SetCodeSize(uint64_t code_size) {
        code_size_ = code_size;
    }

    [[nodiscard]] uint64_t
    GetCodeSize() const {
        return code_size_;
    }

    void
    Resize(uint64_t new_capacity) {
        io_->Resize(new_capacity * code_size_);
    }

    void
    Shrink(uint64_t new_capacity) {
        io_->Shrink(new_capacity * code_size_);
    }

    void
    Write(const uint8_t* code, InnerIdType id) {
        io_->Write(code, code_size_, static_cast<uint64_t>(id) * code_size_);
    }

    bool
    Read(InnerIdType id, uint8_t* dst) const {
        return io_->Read(code_size_, static_cast<uint64_t>(id) * code_size_, dst);
    }

    [[nodiscard]] const uint8_t*
    Read(InnerIdType id, bool& need_release) const {
        return io_->Read(code_size_, static_cast<uint64_t>(id) * code_size_, need_release);
    }

    void
    Release(const uint8_t* code) const {
        if (code != nullptr) {
            io_->Release(code);
        }
    }

    void
    Prefetch(InnerIdType id, uint64_t bytes) const {
        io_->Prefetch(static_cast<uint64_t>(id) * code_size_,
                      std::min<uint64_t>(bytes, code_size_));
    }

    void
    MultiRead(uint8_t* dst, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        io_->MultiRead(dst, sizes, offsets, count);
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

    [[nodiscard]] int64_t
    GetMemoryUsage() const {
        if constexpr (IOTmpl::InMemory) {
            return io_->GetMemoryUsage();
        }
        return 0;
    }

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};
    uint64_t code_size_{0};
};

}  // namespace vsag
