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

#include "io/uring_io/uring_io.h"

#include <fcntl.h>
#include <fmt/format.h>
#include <liburing.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <vector>

#include "io/common/io_syscall.h"
#include "io/uring_io/uring_io_context_guard.h"
#include "io/uring_io/uring_io_read_object.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {

std::unique_ptr<UringIOContextPool> UringIO::io_context_pool =
    std::make_unique<UringIOContextPool>(0, nullptr);

UringIO::UringIO(std::string filename, Allocator* allocator, bool direct_read)
    : BasicIO<UringIO>(allocator), filepath_(std::move(filename)), direct_read_(direct_read) {
    this->exist_file_ = std::filesystem::exists(this->filepath_);
    if (std::filesystem::is_directory(this->filepath_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", this->filepath_));
    }
    int rflags = this->direct_read_ ? (O_CREAT | O_RDWR | O_DIRECT) : (O_CREAT | O_RDWR);
    this->rfd_ = open(filepath_.c_str(), rflags, 0644);
    if (this->rfd_ < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("open file {} error {}", this->filepath_, strerror(errno)));
    }
    this->wfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (this->wfd_ < 0) {
        close(this->rfd_);
        if (not this->exist_file_) {
            std::error_code ec;
            std::filesystem::remove(this->filepath_, ec);
        }
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("open file {} error {}", this->filepath_, strerror(errno)));
    }
}

UringIO::UringIO(const UringIOParameterPtr& io_param, const IndexCommonParam& common_param)
    : UringIO(io_param->path_, common_param.allocator_.get(), io_param->direct_read_) {
}

UringIO::UringIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : UringIO(
          [&]() -> UringIOParameterPtr {
              auto p = std::dynamic_pointer_cast<UringIOParameter>(param);
              if (p == nullptr) {
                  throw VsagException(ErrorType::INVALID_ARGUMENT,
                                      "invalid IO parameter type for UringIO");
              }
              return p;
          }(),
          common_param) {
}

UringIO::~UringIO() {
    close(this->wfd_);
    close(this->rfd_);
    if (not this->exist_file_) {
        std::error_code ec;
        std::filesystem::remove(this->filepath_, ec);
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
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("ftruncate failed for {}: {}", this->filepath_, strerror(errno)));
    }
    this->size_ = size;
}

bool
UringIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    if (size == 0) {
        return true;
    }
    if (offset > this->size_ or size > this->size_ - offset) {
        return false;
    }
    if (this->direct_read_) {
        uint64_t align = 1ULL << Options::Instance().direct_IO_object_align_bit();
        uint64_t align_mask = align - 1;
        uint64_t aligned_offset = offset & ~align_mask;
        uint64_t prefix = offset - aligned_offset;
        uint64_t aligned_size = (prefix + size + align_mask) & ~align_mask;
        auto* buf = static_cast<uint8_t*>(AlignedAlloc(aligned_size, align));
        if (buf == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "UringIO aligned alloc failed");
        }
        auto ret = IOSyscall::PRead(this->rfd_, buf, aligned_size, aligned_offset);
        if (ret < 0) {
            AlignedFree(buf);
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("pread error {}", strerror(errno)));
        }
        if (static_cast<uint64_t>(ret) < prefix + size) {
            AlignedFree(buf);
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("read bytes {} less than {}", ret, prefix + size));
        }
        memcpy(data, buf + prefix, size);
        AlignedFree(buf);
    } else {
        auto ret = IOSyscall::PRead(this->rfd_, data, size, offset);
        if (ret != static_cast<ssize_t>(size)) {
            if (ret < 0) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("pread error {}", strerror(errno)));
            }
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("read bytes {} less than {}", ret, size));
        }
    }
    return true;
}

const uint8_t*
UringIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    need_release = false;
    if (size == 0) {
        return nullptr;
    }
    if (offset > this->size_ or size > this->size_ - offset) {
        return nullptr;
    }
    need_release = true;
    uint8_t* data = nullptr;
    if (this->direct_read_) {
        uint64_t align = 1ULL << Options::Instance().direct_IO_object_align_bit();
        uint64_t align_mask = align - 1;
        uint64_t aligned_offset = offset & ~align_mask;
        uint64_t prefix = offset - aligned_offset;
        uint64_t aligned_size = (prefix + size + align_mask) & ~align_mask;
        data = static_cast<uint8_t*>(AlignedAlloc(aligned_size, align));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "UringIO alloc failed");
        }
        auto ret = IOSyscall::PRead(this->rfd_, data, aligned_size, aligned_offset);
        if (ret < 0) {
            AlignedFree(data);
            need_release = false;
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("pread error {}", strerror(errno)));
        }
        if (static_cast<uint64_t>(ret) < prefix + size) {
            AlignedFree(data);
            need_release = false;
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("read bytes {} less than {}", ret, prefix + size));
        }
        if (prefix != 0) {
            memmove(data, data + prefix, size);
        }
    } else {
        data = static_cast<uint8_t*>(this->allocator_->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "UringIO alloc failed");
        }
        auto ret = IOSyscall::PRead(this->rfd_, data, size, offset);
        if (ret != static_cast<ssize_t>(size)) {
            this->allocator_->Deallocate(data);
            need_release = false;
            if (ret < 0) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("pread error {}", strerror(errno)));
            }
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("read bytes {} less than {}", ret, size));
        }
    }
    return data;
}

void
UringIO::ReleaseImpl(const uint8_t* data) const {
    if (this->direct_read_) {
        AlignedFree(const_cast<uint8_t*>(data));
    } else {
        this->allocator_->Deallocate(const_cast<uint8_t*>(data));
    }
}

bool
UringIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    if (count == 0) {
        return true;
    }

    for (uint64_t i = 0; i < count; ++i) {
        if (sizes[i] > static_cast<uint64_t>(INT32_MAX)) {
            return false;
        }
        if (sizes[i] == 0) {
            continue;
        }
        if (offsets[i] > this->size_ or sizes[i] > this->size_ - offsets[i]) {
            return false;
        }
    }

    static std::atomic<bool> ring_permanently_failed{false};
    std::shared_ptr<UringIOContext> ctx;
    if (not ring_permanently_failed.load(std::memory_order_relaxed)) {
        try {
            ctx = io_context_pool->TakeOne();
        } catch (const VsagException& e) {
            auto msg = std::string(e.what());
            if (msg.find("ENOSYS") != std::string::npos or msg.find("EPERM") != std::string::npos or
                msg.find("EACCES") != std::string::npos) {
                ring_permanently_failed.store(true, std::memory_order_relaxed);
            }
        }
    }
    if (ctx == nullptr) {
        for (uint64_t i = 0; i < count; ++i) {
            if (not this->ReadImpl(sizes[i], offsets[i], datas)) {
                return false;
            }
            datas += sizes[i];
        }
        return true;
    }
    auto* ring = ctx->ring();
    UringIOContextGuard ctx_guard(std::move(ctx));

    uint64_t dio_align = 1ULL << Options::Instance().direct_IO_object_align_bit();
    auto all_count = count;

    while (all_count > 0) {
        uint64_t batch = std::min(static_cast<uint64_t>(UringIOContext::RING_SIZE), all_count);
        std::vector<UringReadObject> objs(batch);

        uint64_t prepared = 0;
        try {
            for (uint64_t i = 0; i < batch; ++i) {
                objs[i].Set(sizes[i], offsets[i], datas, this->direct_read_, dio_align);
                objs[i].released = false;
                datas += sizes[i];

                if (sizes[i] == 0) {
                    continue;
                }

                if (this->direct_read_) {
                    if (not objs[i].AllocAligned(dio_align)) {
                        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                            "UringIO multi-read aligned alloc failed");
                    }
                }

                uint8_t* read_buf = this->direct_read_ ? objs[i].aligned_buf : objs[i].dest_data;
                struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
                if (!sqe) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "io_uring_get_sqe failed in multi-read");
                }
                io_uring_prep_read(sqe,
                                   this->rfd_,
                                   read_buf,
                                   static_cast<uint32_t>(objs[i].submit_size),
                                   objs[i].submit_offset);
                sqe->user_data = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&objs[i]));
                ++prepared;
            }
        } catch (...) {
            for (auto& obj : objs) {
                ReleaseUringReadObject(obj);
            }
            ctx_guard.Drop();
            throw;
        }

        if (prepared == 0) {
            sizes += batch;
            offsets += batch;
            all_count -= batch;
            continue;
        }

        int submitted = io_uring_submit(ring);
        if (submitted < 0) {
            for (auto& obj : objs) {
                ReleaseUringReadObject(obj);
            }
            ctx_guard.Drop();
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("io_uring_submit failed: {}", strerror(-submitted)));
        }

        int remaining_to_submit = static_cast<int>(prepared) - submitted;
        while (remaining_to_submit > 0) {
            int ret = io_uring_submit(ring);
            if (ret <= 0) {
                break;
            }
            submitted += ret;
            remaining_to_submit -= ret;
        }
        if (remaining_to_submit > 0) {
            uint64_t drain_count = 0;
            while (drain_count < static_cast<uint64_t>(submitted)) {
                struct io_uring_cqe* drain_cqe;
                int drain_ret = io_uring_wait_cqe(ring, &drain_cqe);
                if (drain_ret < 0) {
                    if (drain_ret == -EINTR) {
                        continue;
                    }
                    break;
                }
                auto* drain_obj = reinterpret_cast<UringReadObject*>(
                    static_cast<uintptr_t>(drain_cqe->user_data));
                ReleaseUringReadObject(*drain_obj);
                io_uring_cqe_seen(ring, drain_cqe);
                ++drain_count;
            }
            // NOTE: submitted buffers intentionally NOT freed - may be in-flight.
            // Leak is bounded (only on fatal submit failure).
            ctx_guard.Abandon();
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("io_uring_submit could not submit all {} prepared SQEs ({} pending)",
                            prepared,
                            remaining_to_submit));
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
                uint64_t outstanding = static_cast<uint64_t>(submitted) - completed;
                uint64_t drained = 0;
                while (drained < outstanding) {
                    struct io_uring_cqe* drain_cqe;
                    int drain_ret = io_uring_wait_cqe(ring, &drain_cqe);
                    if (drain_ret < 0) {
                        if (drain_ret == -EINTR) {
                            continue;
                        }
                        break;
                    }
                    auto* drain_obj = reinterpret_cast<UringReadObject*>(
                        static_cast<uintptr_t>(drain_cqe->user_data));
                    ReleaseUringReadObject(*drain_obj);
                    io_uring_cqe_seen(ring, drain_cqe);
                    ++drained;
                }
                // NOTE: buffers from drained CQEs released in loop above.
                // If drain fails, remaining buffers intentionally leaked (use-after-free prevention).
                ctx_guard.Abandon();
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("io_uring_wait_cqe failed: {}", strerror(-ret)));
            }

            auto* obj = reinterpret_cast<UringReadObject*>(static_cast<uintptr_t>(cqe->user_data));
            if (cqe->res < 0) {
                first_error = first_error == 0 ? cqe->res : first_error;
            } else if (static_cast<uint64_t>(cqe->res) < obj->prefix + obj->orig_size) {
                has_short_read = true;
            } else {
                obj->CopyToDest();
            }

            ReleaseUringReadObject(*obj);
            completed++;

            io_uring_cqe_seen(ring, cqe);
        }

        if (first_error < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("multi-read failed: {}", strerror(-first_error)));
        }
        if (has_short_read) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "multi-read short read");
        }
        sizes += batch;
        offsets += batch;
        all_count -= batch;
    }

    return true;
}

}  // namespace vsag

#endif  // HAVE_LIBURING
