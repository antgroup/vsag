
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

#include "async_io.h"

#include <filesystem>

#include "direct_io_object.h"
#include "io_context.h"

namespace vsag {
std::unique_ptr<IOContextPool> AsyncIO::io_context_pool =
    std::make_unique<IOContextPool>(10, nullptr);

AsyncIO::AsyncIO(std::string filename, Allocator* allocator)
    : BasicIO<AsyncIO>(allocator), filepath_(std::move(filename)) {
    this->exist_file_ = std::filesystem::exists(this->filepath_);
    if (std::filesystem::is_directory(this->filepath_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", this->filepath_));
    }
    this->rfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (this->rfd_ < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("open file {} error {}", this->filepath_, strerror(errno)));
    }
    this->wfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (this->wfd_ < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("open file {} error {}", this->filepath_, strerror(errno)));
    }
}

AsyncIO::AsyncIO(const AsyncIOParameterPtr& io_param, const IndexCommonParam& common_param)
    : AsyncIO(io_param->path_, common_param.allocator_.get()){};

AsyncIO::AsyncIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : AsyncIO(std::dynamic_pointer_cast<AsyncIOParameter>(param), common_param){};

AsyncIO::~AsyncIO() {
    close(this->wfd_);
    close(this->rfd_);
    // remove file
    if (not this->exist_file_) {
        std::filesystem::remove(this->filepath_);
    }
}

void
AsyncIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    auto ret = pwrite64(this->wfd_, data, size, static_cast<int64_t>(offset));
    if (ret != size) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("write bytes {} less than {}", ret, size));
    }
    if (size + offset > this->size_) {
        this->size_ = size + offset;
    }
    fsync(wfd_);
}

bool
AsyncIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    bool need_release = true;
    const auto* ptr = DirectReadImpl(size, offset, need_release);
    memcpy(data, ptr, size);
    AsyncIO::ReleaseImpl(ptr);
    return true;
}

const uint8_t*
AsyncIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    need_release = true;
    if (size == 0) {
        return nullptr;
    }
    DirectIOObject obj(size, offset);
    auto ret = pread64(this->rfd_, obj.align_data, obj.size, static_cast<int64_t>(obj.offset));
    if (ret < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, fmt::format("pread64 error {}", ret));
    }
    return obj.data;
}

void
AsyncIO::ReleaseImpl(const uint8_t* data) {
    auto* ptr = const_cast<uint8_t*>(data);
    uint64_t align_bit = Options::Instance().direct_IO_object_align_bit();
    auto raw = reinterpret_cast<uintptr_t>(ptr);
    raw &= ~((1ULL << align_bit) - 1);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    free(reinterpret_cast<void*>(raw));
}

bool
AsyncIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    auto context = io_context_pool->TakeOne();
    uint8_t* cur_data = datas;
    auto all_count = static_cast<int64_t>(count);
    while (all_count > 0) {
        count = std::min(IOContext::DEFAULT_REQUEST_COUNT, all_count);
        auto* cb = context->cb_;
        std::vector<DirectIOObject> objs(count);
        for (int64_t i = 0; i < count; ++i) {
            objs[i].Set(sizes[i], offsets[i]);
            auto& obj = objs[i];
            io_prep_pread(cb[i], rfd_, obj.align_data, obj.size, static_cast<int64_t>(obj.offset));
            cb[i]->data = &(objs[i]);
        }

        int submitted = io_submit(context->ctx_, static_cast<int64_t>(count), cb);
        if (submitted < 0) {
            io_context_pool->ReturnOne(context);
            for (auto& obj : objs) {
                obj.Release();
            }
            throw VsagException(ErrorType::INTERNAL_ERROR, "io submit failed");
        }

        struct timespec timeout = {1, 0};
        auto num_events = io_getevents(context->ctx_,
                                       static_cast<int64_t>(count),
                                       static_cast<int64_t>(count),
                                       context->events_,
                                       &timeout);
        if (num_events != count) {
            io_context_pool->ReturnOne(context);
            for (auto& obj : objs) {
                obj.Release();
            }
            throw VsagException(ErrorType::INTERNAL_ERROR, "io async read failed");
        }

        for (int64_t i = 0; i < count; ++i) {
            memcpy(cur_data, objs[i].data, sizes[i]);
            cur_data += sizes[i];
            this->ReleaseImpl(objs[i].data);
        }

        sizes += count;
        offsets += count;
        all_count -= static_cast<int64_t>(count);
    }
    io_context_pool->ReturnOne(context);
    return true;
}

}  // namespace vsag
