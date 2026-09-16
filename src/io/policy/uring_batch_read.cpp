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

#include "io/policy/uring_batch_read.h"

#if HAVE_LIBURING

#include <liburing.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "io/policy/direct_single_read.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

class UringContextLease {
public:
    UringContextLease(UringIOContextPool* pool, std::shared_ptr<UringIOContext> context)
        : pool_(pool), context_(std::move(context)) {
    }

    ~UringContextLease() noexcept {
        if (reusable_ and context_ != nullptr) {
            try {
                pool_->ReturnOne(context_);
            } catch (...) {
            }
        }
    }

    [[nodiscard]] UringIOContext*
    Get() const {
        return context_.get();
    }

    void
    Discard() {
        reusable_ = false;
        context_.reset();
    }

private:
    UringIOContextPool* pool_{nullptr};
    std::shared_ptr<UringIOContext> context_;
    bool reusable_{true};
};

bool
is_permanent_ring_failure(int error_code) {
    return error_code == ENOSYS or error_code == EPERM or error_code == EACCES;
}

std::shared_ptr<UringIOContext>
take_context(UringIOContextPool* pool) {
    // These errors indicate a process-wide kernel or sandbox restriction, so avoid repeatedly
    // attempting ring setup. Resource-pressure failures remain retryable because they do not set
    // this circuit breaker.
    static std::atomic<bool> permanently_unavailable{false};
    if (permanently_unavailable.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    try {
        return pool->TakeOne();
    } catch (const UringSetupException& exception) {
        if (is_permanent_ring_failure(exception.ErrorCode())) {
            permanently_unavailable.store(true, std::memory_order_relaxed);
        }
        return nullptr;
    } catch (const VsagException&) {
        return nullptr;
    }
}

template <bool direct_read>
class pending_read;

template <>
class pending_read<false> {
public:
    pending_read(uint64_t size, uint64_t offset, uint8_t* destination, uint64_t direct_alignment)
        : destination_(destination), size_(size), offset_(offset) {
        (void)direct_alignment;
    }

    [[nodiscard]] uint8_t*
    SubmitBuffer() const {
        return destination_;
    }

    [[nodiscard]] uint64_t
    SubmitSize() const {
        return size_;
    }

    [[nodiscard]] uint64_t
    SubmitOffset() const {
        return offset_;
    }

    [[nodiscard]] uint64_t
    MinimumResultSize() const {
        return size_;
    }

    void
    CopyToDestination() const {
    }

    void
    ReleaseBuffer() {
    }

private:
    uint8_t* destination_{nullptr};
    uint64_t size_{0};
    uint64_t offset_{0};
};

template <>
class pending_read<true> {
public:
    pending_read(uint64_t size, uint64_t offset, uint8_t* destination, uint64_t direct_alignment)
        : direct_buffer_(size, offset, direct_alignment), destination_(destination) {
    }

    [[nodiscard]] uint8_t*
    SubmitBuffer() const {
        return direct_buffer_.Base();
    }

    [[nodiscard]] uint64_t
    SubmitSize() const {
        return direct_buffer_.SubmitSize();
    }

    [[nodiscard]] uint64_t
    SubmitOffset() const {
        return direct_buffer_.SubmitOffset();
    }

    [[nodiscard]] uint64_t
    MinimumResultSize() const {
        return direct_buffer_.MinimumResultSize();
    }

    void
    CopyToDestination() const {
        std::memcpy(destination_, direct_buffer_.Data(), direct_buffer_.RequestedSize());
    }

    void
    ReleaseBuffer() {
        direct_buffer_.Release();
    }

private:
    DirectReadBuffer direct_buffer_;
    uint8_t* destination_{nullptr};
};

bool
drain_submitted(UringContextLease& context_lease, uint64_t submitted) {
    io_uring* ring = context_lease.Get()->ring();
    uint64_t completed = 0;
    while (completed < submitted) {
        io_uring_cqe* cqe = nullptr;
        int result = io_uring_wait_cqe(ring, &cqe);
        if (result == -EINTR) {
            continue;
        }
        if (result < 0) {
            context_lease.Discard();
            return false;
        }
        io_uring_cqe_seen(ring, cqe);
        ++completed;
    }
    return true;
}

template <bool direct_read, typename Destination>
UringBatchRead::ReadStatus
execute(UringIOContextPool* pool,
        const PosixFile& file,
        uint64_t count,
        Destination&& destination) {
    for (uint64_t i = 0; i < count; ++i) {
        if (destination.Size(i) > static_cast<uint64_t>(INT32_MAX)) {
            return UringBatchRead::ReadStatus::Invalid;
        }
    }
    std::shared_ptr<UringIOContext> context = take_context(pool);
    if (context == nullptr) {
        return UringBatchRead::ReadStatus::Unavailable;
    }
    UringContextLease context_lease(pool, std::move(context));
    io_uring* ring = context_lease.Get()->ring();
    uint64_t direct_alignment = direct_read ? DirectReadBuffer::Alignment() : 1;

    uint64_t completed_requests = 0;
    while (completed_requests < count) {
        uint64_t slice_size =
            std::min<uint64_t>(UringIOContext::RING_SIZE, count - completed_requests);
        std::vector<pending_read<direct_read>> pending_reads;
        pending_reads.reserve(slice_size);
        try {
            for (uint64_t i = 0; i < slice_size; ++i) {
                uint64_t request_index = completed_requests + i;
                uint64_t request_size = destination.Size(request_index);
                uint8_t* request_destination = destination.Address(request_index);
                if (request_size == 0) {
                    continue;
                }
                pending_reads.emplace_back(request_size,
                                           destination.Offset(request_index),
                                           request_destination,
                                           direct_alignment);
                pending_read<direct_read>& pending = pending_reads.back();
                if (pending.SubmitSize() > static_cast<uint64_t>(INT32_MAX)) {
                    context_lease.Discard();
                    return UringBatchRead::ReadStatus::Invalid;
                }
                io_uring_sqe* sqe = io_uring_get_sqe(ring);
                if (sqe == nullptr) {
                    context_lease.Discard();
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "io_uring_get_sqe failed while preparing batch");
                }
                io_uring_prep_read(sqe,
                                   file.ReadFd(),
                                   pending.SubmitBuffer(),
                                   static_cast<uint32_t>(pending.SubmitSize()),
                                   pending.SubmitOffset());
                io_uring_sqe_set_data(sqe, &pending);
            }
        } catch (...) {
            if (context_lease.Get() != nullptr) {
                context_lease.Discard();
            }
            throw;
        }

        if (pending_reads.empty()) {
            completed_requests += slice_size;
            continue;
        }

        auto requested_submit = static_cast<int64_t>(pending_reads.size());
        int64_t submitted = 0;
        while (submitted < requested_submit) {
            int result = io_uring_submit(ring);
            if (result == -EINTR) {
                continue;
            }
            if (result <= 0) {
                if (submitted > 0) {
                    (void)drain_submitted(context_lease, static_cast<uint64_t>(submitted));
                }
                if (context_lease.Get() != nullptr) {
                    context_lease.Discard();
                }
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    fmt::format("io_uring partial submit: requested {} but submitted {}",
                                requested_submit,
                                submitted));
            }
            submitted += result;
        }

        int first_error = 0;
        bool short_read = false;
        int64_t completed = 0;
        while (completed < submitted) {
            io_uring_cqe* cqe = nullptr;
            int result = io_uring_wait_cqe(ring, &cqe);
            if (result == -EINTR) {
                continue;
            }
            if (result < 0) {
                context_lease.Discard();
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("io_uring_wait_cqe failed with errno {}", -result));
            }

            auto* pending = static_cast<pending_read<direct_read>*>(io_uring_cqe_get_data(cqe));
            if (cqe->res < 0) {
                if (first_error == 0) {
                    first_error = cqe->res;
                }
            } else if (static_cast<uint64_t>(cqe->res) < pending->MinimumResultSize()) {
                short_read = true;
            } else {
                pending->CopyToDestination();
            }
            pending->ReleaseBuffer();
            io_uring_cqe_seen(ring, cqe);
            ++completed;
        }

        if (first_error < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("io_uring read failed with errno {}", -first_error));
        }
        if (short_read) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "io_uring read completed short");
        }
        completed_requests += slice_size;
    }
    return UringBatchRead::ReadStatus::Completed;
}

template <typename Destination>
UringBatchRead::ReadStatus
execute_configured(UringIOContextPool* pool,
                   bool direct_read,
                   const PosixFile& file,
                   uint64_t count,
                   Destination destination) {
    if (direct_read) {
        return execute<true>(pool, file, count, destination);
    }
    return execute<false>(pool, file, count, destination);
}

class ScatterDestination {
public:
    explicit ScatterDestination(const ReadRequest* requests) : requests_(requests) {
    }

    [[nodiscard]] uint64_t
    Size(uint64_t index) const {
        return requests_[index].size;
    }

    [[nodiscard]] uint64_t
    Offset(uint64_t index) const {
        return requests_[index].offset;
    }

    [[nodiscard]] uint8_t*
    Address(uint64_t index) const {
        return requests_[index].destination;
    }

private:
    const ReadRequest* requests_{nullptr};
};

class ContiguousDestination {
public:
    ContiguousDestination(uint8_t* destination, const uint64_t* sizes, const uint64_t* offsets)
        : destination_(destination), sizes_(sizes), offsets_(offsets) {
    }

    [[nodiscard]] uint64_t
    Size(uint64_t index) const {
        return sizes_[index];
    }

    [[nodiscard]] uint64_t
    Offset(uint64_t index) const {
        return offsets_[index];
    }

    [[nodiscard]] uint8_t*
    Address(uint64_t index) {
        uint8_t* address = destination_;
        destination_ += sizes_[index];
        return address;
    }

private:
    uint8_t* destination_{nullptr};
    const uint64_t* sizes_{nullptr};
    const uint64_t* offsets_{nullptr};
};

}  // namespace

UringBatchRead::ReadStatus
UringBatchRead::ReadScatter(const PosixFile& file,
                            const ReadRequest* requests,
                            uint64_t count) const {
    if (count == 0) {
        return ReadStatus::Completed;
    }
    return execute_configured(
        context_pool_, direct_read_, file, count, ScatterDestination(requests));
}

UringBatchRead::ReadStatus
UringBatchRead::ReadContiguous(const PosixFile& file,
                               uint8_t* destination,
                               const uint64_t* sizes,
                               const uint64_t* offsets,
                               uint64_t count) const {
    if (count == 0) {
        return ReadStatus::Completed;
    }
    return execute_configured(context_pool_,
                              direct_read_,
                              file,
                              count,
                              ContiguousDestination(destination, sizes, offsets));
}

}  // namespace vsag

#endif
