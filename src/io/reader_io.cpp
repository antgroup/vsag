
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

#include "reader_io.h"

#include <fmt/format.h>

#include <future>

namespace vsag {

void
ReaderIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    // ReaderIO is read-only, so we do nothing here. Just for deserialization.
    this->size_ += size;
}

void
ReaderIO::InitIOImpl(const vsag::IOParamPtr& io_param) {
    auto reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(io_param);
    if (not reader_param) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "ReaderIOParam is required for ReaderIO initialization.");
    }
    reader_ = reader_param->reader;
}

bool
ReaderIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    if (not reader_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "ReaderIO is not initialized, please call Init() first.");
    }
    bool ret = check_valid_offset(size + offset);
    if (ret) {
        reader_->Read(start_ + offset, size, data);
    }
    return ret;
}

[[nodiscard]] const uint8_t*
ReaderIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    if (not reader_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "ReaderIO is not initialized, please call Init() first.");
    }
    if (check_valid_offset(size + offset)) {
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        need_release = true;
        reader_->Read(start_ + offset, size, data);
        return data;
    }
    return nullptr;
}

void
ReaderIO::ReleaseImpl(const uint8_t* data) const {
    allocator_->Deallocate((void*)data);
}

bool
ReaderIO::MultiReadImpl(uint8_t* datas,
                        const uint64_t* sizes,
                        const uint64_t* offsets,
                        uint64_t count) const {
    if (not reader_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "ReaderIO is not initialized, please call Init() first.");
    }

    std::vector<uint64_t> real_offsets(count);
    for (uint64_t i = 0; i < count; ++i) {
        real_offsets[i] = start_ + offsets[i];
    }
    return reader_->MultiRead(datas, sizes, real_offsets.data(), count);
}

}  // namespace vsag
