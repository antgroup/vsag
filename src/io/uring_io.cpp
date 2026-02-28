
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

#if HAVE_LIBURING

#include "uring_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>

#include "direct_io_object.h"

namespace vsag {

std::unique_ptr<UringIOContextPool> UringIO::io_context_pool_ =
    std::make_unique<UringIOContextPool>(10, nullptr);

UringIO::UringIO(std::string filename, Allocator* allocator)
    : BasicIO<UringIO>(allocator), filepath_(std::move(filename)) {
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

UringIO::UringIO(const UringIOParameterPtr& io_param, const IndexCommonParam& common_param)
    : UringIO(io_param->path_, common_param.allocator_.get()) {
}

UringIO::UringIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : UringIO(std::dynamic_pointer_cast<UringIOParameter>(param), common_param) {
}

UringIO::~UringIO() {
    close(this->wfd_);
    close(this->rfd_);
    // remove file
    if (not this->exist_file_) {
        std::filesystem::remove(this->filepath_);
    }
}

void
UringIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    auto ret = pwrite64(this->wfd_, data, size, static_cast<int64_t>(offset));
    if (ret != size) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("write bytes {} less than {}", ret, size));
    }
    if (size + offset > this->size_) {
        this->size_ = size + offset;
    }
    fsync(this->wfd_);
}

bool
UringIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    bool need_release = true;
    const uint8_t* ptr = DirectReadImpl(size, offset, need_release);
    if (ptr == nullptr)
        return false;
    memcpy(data, ptr, size);
    ReleaseImpl(ptr);
    return true;
}

const uint8_t*
UringIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    if (not check_valid_offset(size + offset)) {
        return nullptr;
    }
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
UringIO::ReleaseImpl(const uint8_t* data) {
    auto* ptr = const_cast<uint8_t*>(data);
    uint64_t align_bit = Options::Instance().direct_IO_object_align_bit();
    auto raw = reinterpret_cast<uintptr_t>(ptr);
    raw &= ~((1ULL << align_bit) - 1);
    free(reinterpret_cast<void*>(raw));
}

class UringDirectIOObject : public DirectIOObject {
public:
    explicit UringDirectIOObject(uint64_t size, uint64_t offset, uint8_t* dist_data)
        : DirectIOObject(size, offset), dist_data(dist_data), dist_size(size) {
    }

    UringDirectIOObject() = default;

    void
    Set(uint64_t size1, uint64_t offset1, uint8_t* dist_data1) {
        DirectIOObject::Set(size1, offset1);
        this->dist_data = dist_data1;
        this->dist_size = size1;
    }

    uint8_t* dist_data{nullptr};
    uint64_t dist_size{0};
};

bool
UringIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    if (count == 0) {
        return true;
    }

    auto ctx = io_context_pool_->TakeOne();
    auto* ring = ctx->ring();

    auto all_count = static_cast<int64_t>(count);

    while (all_count > 0) {
        count = std::min(static_cast<uint64_t>(UringIOContext::RING_SIZE),
                         static_cast<uint64_t>(all_count));
        std::vector<UringDirectIOObject> objs(count);

        for (uint64_t i = 0; i < count; ++i) {
            objs[i].Set(sizes[i], offsets[i], datas);
            datas += sizes[i];

            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) {
                io_context_pool_->ReturnOne(ctx);
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "io_uring_get_sqe failed in multi-read");
            }
            io_uring_prep_read(sqe,
                               this->rfd_,
                               objs[i].align_data,
                               objs[i].size,
                               static_cast<int64_t>(objs[i].offset));
            sqe->user_data = reinterpret_cast<uint64_t>(&objs[i]);
        }

        int submitted = io_uring_submit(ring);
        if (submitted < static_cast<int>(count)) {
            io_context_pool_->ReturnOne(ctx);
            throw VsagException(ErrorType::INTERNAL_ERROR, "failed to submit all reads");
        }

        uint64_t completed = 0;
        while (completed < count) {
            struct io_uring_cqe* cqe;
            int ret = io_uring_wait_cqe(ring, &cqe);
            if (ret < 0) {
                io_context_pool_->ReturnOne(ctx);
                throw VsagException(ErrorType::INTERNAL_ERROR, "wait cqe failed");
            }

            if (cqe->res < 0) {
                io_context_pool_->ReturnOne(ctx);
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("multi-read failed: {}", strerror(-cqe->res)));
            }

            auto* obj = reinterpret_cast<UringDirectIOObject*>(cqe->user_data);

            memcpy(obj->dist_data, obj->data, obj->dist_size);
            ReleaseImpl(obj->data);
            completed++;

            io_uring_cqe_seen(ring, cqe);
        }
        sizes += count;
        offsets += count;
        all_count -= static_cast<int64_t>(count);
    }

    io_context_pool_->ReturnOne(ctx);
    return true;
}

}  // namespace vsag

#endif  // HAVE_LIBURING
