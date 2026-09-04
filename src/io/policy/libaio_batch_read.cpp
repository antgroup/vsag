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

#include "io/policy/libaio_batch_read.h"

#if HAVE_LIBAIO

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "io/policy/direct_single_read.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

class AioContextLease {
public:
    explicit AioContextLease(IOContextPool* pool) : pool_(pool), context_(pool_->TakeOne()) {
    }

    ~AioContextLease() noexcept {
        if (reusable_) {
            try {
                pool_->ReturnOne(context_);
            } catch (...) {
            }
        }
    }

    [[nodiscard]] const std::shared_ptr<IOContext>&
    Get() const {
        return context_;
    }

    void
    MarkUnusable() {
        reusable_ = false;
    }

private:
    IOContextPool* pool_{nullptr};
    std::shared_ptr<IOContext> context_;
    bool reusable_{true};
};

void
wait_for_submitted(AioContextLease& context_lease, int64_t submitted) {
    const auto& context = context_lease.Get();
    int64_t completed = 0;
    while (completed < submitted) {
        int result = io_getevents(context->ctx_,
                                  submitted - completed,
                                  submitted - completed,
                                  context->events_ + completed,
                                  nullptr);
        if (result == -EINTR) {
            continue;
        }
        if (result < 0) {
            (void)io_destroy(context->ctx_);
            context->ctx_ = nullptr;
            if (io_setup(IOContext::DEFAULT_REQUEST_COUNT, &context->ctx_) != 0) {
                context_lease.MarkUnusable();
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "io_getevents failed and libaio context recreation failed");
            }
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "io_getevents failed with errno " + std::to_string(-result));
        }
        completed += result;
    }
}

template <typename Destination>
bool
execute(IOContextPool* pool, const PosixFile& file, uint64_t count, Destination&& destination) {
    AioContextLease context_lease(pool);
    const auto& context = context_lease.Get();
    uint64_t completed_requests = 0;
    while (completed_requests < count) {
        uint64_t slice_size =
            std::min<uint64_t>(IOContext::DEFAULT_REQUEST_COUNT, count - completed_requests);
        struct PendingRead {
            DirectReadBuffer buffer;
            uint8_t* destination{nullptr};

            PendingRead(uint64_t size, uint64_t offset, uint8_t* destination_value)
                : buffer(size, offset), destination(destination_value) {
            }
        };
        std::vector<PendingRead> pending_reads;
        pending_reads.reserve(slice_size);
        for (uint64_t i = 0; i < slice_size; ++i) {
            uint64_t request_index = completed_requests + i;
            uint64_t request_size = destination.Size(request_index);
            uint8_t* request_destination = destination.Address(request_index);
            if (request_size == 0) {
                continue;
            }
            pending_reads.emplace_back(
                request_size, destination.Offset(request_index), request_destination);
            uint64_t pending_index = pending_reads.size() - 1;
            io_prep_pread(context->cb_[pending_index],
                          file.ReadFd(),
                          pending_reads[pending_index].buffer.Base(),
                          pending_reads[pending_index].buffer.SubmitSize(),
                          static_cast<int64_t>(pending_reads[pending_index].buffer.SubmitOffset()));
            context->cb_[pending_index]->data = &pending_reads[pending_index];
        }
        if (pending_reads.empty()) {
            completed_requests += slice_size;
            continue;
        }

        const auto requested_submit = static_cast<int64_t>(pending_reads.size());
        int64_t submitted_total = 0;
        while (submitted_total < requested_submit) {
            int submitted = io_submit(
                context->ctx_, requested_submit - submitted_total, context->cb_ + submitted_total);
            if (submitted == -EINTR) {
                continue;
            }
            if (submitted <= 0) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "io_submit failed with result " + std::to_string(submitted));
            }
            wait_for_submitted(context_lease, submitted);
            for (int64_t i = 0; i < submitted; ++i) {
                const struct io_event& event = context->events_[i];
                auto* pending = static_cast<PendingRead*>(event.data);
                if (event.res < 0 or event.res2 != 0 or
                    static_cast<uint64_t>(event.res) < pending->buffer.MinimumResultSize()) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "libaio read completed with short or failed result");
                }
                std::memcpy(
                    pending->destination, pending->buffer.Data(), pending->buffer.RequestedSize());
            }
            submitted_total += submitted;
        }
        completed_requests += slice_size;
    }
    return true;
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
    // The two arrays are intentionally kept separate to match the legacy MultiRead contract.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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

bool
LibAioBatchRead::ReadScatter(const PosixFile& file,
                             const ReadRequest* requests,
                             uint64_t count) const {
    if (count == 0) {
        return true;
    }
    return execute(context_pool_, file, count, ScatterDestination(requests));
}

bool
LibAioBatchRead::ReadContiguous(const PosixFile& file,
                                uint8_t* destination,
                                const uint64_t* sizes,
                                const uint64_t* offsets,
                                uint64_t count) const {
    if (count == 0) {
        return true;
    }
    return execute(context_pool_, file, count, ContiguousDestination(destination, sizes, offsets));
}

}  // namespace vsag

#endif
