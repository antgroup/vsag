
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

#include "io_syscall.h"

namespace vsag {

std::unique_ptr<UringIOContextPool> UringIO::io_context_pool_ =
    std::make_unique<UringIOContextPool>(0, nullptr);

UringIO::UringIO(std::string filename, Allocator* allocator)
    : BasicIO<UringIO>(allocator), filepath_(std::move(filename)) {
    this->exist_file_ = std::filesystem::exists(this->filepath_);
    if (std::filesystem::is_directory(this->filepath_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", this->filepath_));
    }
    this->rfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (this->rfd_ < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("open file {} error {}", this->filepath_, strerror(errno)));
    }
    this->wfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (this->wfd_ < 0) {
        close(this->rfd_);
        if (not this->exist_file_) {
            std::filesystem::remove(this->filepath_);
        }
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
    auto ret = IOSyscall::PWrite(this->wfd_, data, size, offset);
    if (ret != static_cast<ssize_t>(size)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("write bytes {} less than {}", ret, size));
    }
    if (size + offset > this->size_) {
        this->size_ = size + offset;
    }
}

void
UringIO::ResizeImpl(uint64_t size) {
    auto ret = IOSyscall::FTruncate(this->wfd_, size);
    if (ret == -1) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate failed");
    }
    this->size_ = size;
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
    auto* data = static_cast<uint8_t*>(malloc(size));
    if (data == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "UringIO allocation failed");
    }
    auto ret = IOSyscall::PRead(this->rfd_, data, size, offset);
    if (ret != static_cast<ssize_t>(size)) {
        free(data);
        if (ret < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("pread error {}", strerror(errno)));
        }
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("read bytes {} less than {}", ret, size));
    }
    return data;
}

void
UringIO::ReleaseImpl(const uint8_t* data) {
    free(const_cast<uint8_t*>(data));
}

class UringReadObject {
public:
    UringReadObject() = default;

    void
    Set(uint64_t size1, uint64_t offset1, uint8_t* dist_data1) {
        if (data != nullptr) {
            free(data);
        }
        this->data = static_cast<uint8_t*>(malloc(size1));
        if (this->data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "UringIO multi-read allocation failed");
        }
        this->size = size1;
        this->offset = offset1;
        this->dist_data = dist_data1;
    }

    uint8_t* data{nullptr};
    uint64_t size{0};
    uint64_t offset{0};
    uint8_t* dist_data{nullptr};
    bool released{true};
};

static void
ReleaseUringReadObject(UringReadObject& obj) {
    if (not obj.released) {
        free(obj.data);
        obj.data = nullptr;
        obj.released = true;
    }
}

class UringIOContextGuard {
public:
    explicit UringIOContextGuard(std::shared_ptr<UringIOContext> ctx) : ctx_(std::move(ctx)) {
    }

    ~UringIOContextGuard() {
        if (ctx_ != nullptr) {
            UringIO::io_context_pool_->ReturnOne(ctx_);
        }
    }

    UringIOContextGuard(const UringIOContextGuard&) = delete;
    UringIOContextGuard&
    operator=(const UringIOContextGuard&) = delete;

    void
    Abandon() {
        // Keep the context alive but do not return it to the pool when the ring
        // may still have outstanding kernel references after a fatal wait error.
        (void)new std::shared_ptr<UringIOContext>(ctx_);
        ctx_ = nullptr;
    }

private:
    std::shared_ptr<UringIOContext> ctx_{nullptr};
};

bool
UringIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    if (count == 0) {
        return true;
    }

    std::shared_ptr<UringIOContext> ctx;
    try {
        ctx = io_context_pool_->TakeOne();
    } catch (const VsagException&) {
        for (uint64_t i = 0; i < count; ++i) {
            if (not this->ReadImpl(sizes[i], offsets[i], datas)) {
                return false;
            }
            datas += sizes[i];
        }
        return true;
    }
    UringIOContextGuard ctx_guard(ctx);
    auto* ring = ctx->ring();

    auto all_count = static_cast<int64_t>(count);

    while (all_count > 0) {
        count = std::min(static_cast<uint64_t>(UringIOContext::RING_SIZE),
                         static_cast<uint64_t>(all_count));
        std::vector<UringReadObject> objs(count);

        try {
            for (uint64_t i = 0; i < count; ++i) {
                objs[i].Set(sizes[i], offsets[i], datas);
                objs[i].released = false;
                datas += sizes[i];

                struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
                if (!sqe) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "io_uring_get_sqe failed in multi-read");
                }
                io_uring_prep_read(
                    sqe, this->rfd_, objs[i].data, objs[i].size, static_cast<int64_t>(offsets[i]));
                sqe->user_data = reinterpret_cast<uint64_t>(&objs[i]);
            }
        } catch (...) {
            for (auto& obj : objs) {
                ReleaseUringReadObject(obj);
            }
            ctx_guard.Abandon();
            throw;
        }

        int submitted = io_uring_submit(ring);
        if (submitted < 0) {
            for (auto& obj : objs) {
                ReleaseUringReadObject(obj);
            }
            ctx_guard.Abandon();
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("io_uring_submit failed: {}", strerror(-submitted)));
        }

        int first_error = 0;
        bool has_short_read = false;
        uint64_t completed = 0;
        while (completed < static_cast<uint64_t>(submitted)) {
            struct io_uring_cqe* cqe;
            int ret = io_uring_wait_cqe(ring, &cqe);
            if (ret < 0) {
                if (ret == -EINTR) {
                    continue;
                }
                ctx_guard.Abandon();
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("io_uring_wait_cqe failed: {}", strerror(-ret)));
            }

            auto* obj = reinterpret_cast<UringReadObject*>(cqe->user_data);
            if (cqe->res < 0) {
                first_error = first_error == 0 ? cqe->res : first_error;
            } else if (cqe->res != static_cast<int>(obj->size)) {
                has_short_read = true;
            } else {
                memcpy(obj->dist_data, obj->data, obj->size);
            }

            ReleaseUringReadObject(*obj);
            completed++;

            io_uring_cqe_seen(ring, cqe);
        }

        if (submitted != static_cast<int>(count)) {
            for (auto& obj : objs) {
                ReleaseUringReadObject(obj);
            }
            ctx_guard.Abandon();
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format(
                    "io_uring_submit partial: requested {} but submitted {}", count, submitted));
        }

        if (first_error < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("multi-read failed: {}", strerror(-first_error)));
        }
        if (has_short_read) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "multi-read short read");
        }
        sizes += count;
        offsets += count;
        all_count -= static_cast<int64_t>(count);
    }

    return true;
}

}  // namespace vsag

#endif  // HAVE_LIBURING
