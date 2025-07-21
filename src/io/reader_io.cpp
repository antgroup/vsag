
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
    std::atomic<bool> succeed(true);
    std::string error_message;
    std::atomic<uint64_t> counter(count);
    std::promise<void> total_promise;
    uint8_t* dest = datas;
    auto total_future = total_promise.get_future();
    for (int i = 0; i < count; ++i) {
        uint64_t offset = offsets[i];
        uint64_t size = sizes[i];
        if (not check_valid_offset(size + offset)) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("ReaderIO MultiReadImpl size mismatch: "
                                            "offset {}, size {}, total size {}",
                                            offset,
                                            size,
                                            size_ + start_));
        }
        auto callback = [&counter, &total_promise, &succeed, &error_message](
                            IOErrorCode code, const std::string& message) {
            if (code != vsag::IOErrorCode::IO_SUCCESS) {
                bool expected = true;
                if (succeed.compare_exchange_strong(expected, false)) {
                    error_message = message;
                }
            }
            if (--counter == 0) {
                total_promise.set_value();
            }
        };
        reader_->AsyncRead(start_ + offset, size, dest, callback);
        dest += size;
    }
    total_future.wait();
    if (not succeed) {
        throw VsagException(ErrorType::READ_ERROR, "failed to read diskann index");
    }
    return true;
}

void
ReaderIO::PrefetchImpl(uint64_t offset, uint64_t cache_line) {
}

}  // namespace vsag
