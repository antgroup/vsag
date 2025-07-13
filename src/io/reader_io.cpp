
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
}

bool
ReaderIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    bool ret = check_valid_offset(size + offset);
    if (ret) {
        reader_->Read(offset, size, data);
    }
    return ret;
}

const uint8_t*
ReaderIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    if (check_valid_offset(size + offset)) {
        uint8_t* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        need_release = true;
        reader_->Read(offset, size, data);
        return data;
    }
    return nullptr;
}

bool
ReaderIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    std::atomic<bool> succeed(true);
    std::string error_message;
    std::atomic<uint64_t> counter(count);
    std::promise<void> total_promise;
    uint8_t* dest = datas;
    auto total_future = total_promise.get_future();
    for (int i = 0; i < count; ++i) {
        uint64_t offset = offsets[i];
        uint64_t size = sizes[i];
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
        reader_->AsyncRead(offset, size, dest, callback);
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

void
ReaderIO::ReleaseImpl(const uint8_t* data) const {
    allocator_->Deallocate((void *)data);
}

void
ReaderIO::check(uint64_t size) {
    if (size <= this->size_) {
        return;
    }
    throw VsagException(ErrorType::INTERNAL_ERROR,
                        fmt::format("ReaderIO size is not enough"
                                    " size: {}, reader size: {}",
                                    size,
                                    this->size_));
}

}  // namespace vsag
