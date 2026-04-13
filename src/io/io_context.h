
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

#if HAVE_LIBAIO

#include "libaio.h"
#include "utils/resource_object.h"
#include "utils/resource_object_pool.h"

namespace vsag {

/**
 * @brief Linux libaio context for asynchronous IO operations.
 *
 * This class manages the io_context_t and associated control blocks (iocb)
 * for performing asynchronous read/write operations using the Linux AIO subsystem.
 * It is pooled via IOContextPool for efficient reuse across AsyncIO instances.
 */
class IOContext : public ResourceObject {
public:
    /**
     * @brief Constructs an IOContext with default request count.
     *
     * Initializes the libaio context and allocates iocb structures for requests.
     */
    IOContext() {
        memset(&ctx_, 0, sizeof(ctx_));
        io_setup(DEFAULT_REQUEST_COUNT, &this->ctx_);
        for (int i = 0; i < DEFAULT_REQUEST_COUNT; ++i) {
            this->cb_[i] = static_cast<iocb*>(malloc(sizeof(struct iocb)));
        }
    }

    /**
     * @brief Destructor that destroys the libaio context and frees iocb structures.
     */
    ~IOContext() override {
        io_destroy(this->ctx_);
        for (int i = 0; i < DEFAULT_REQUEST_COUNT; ++i) {
            free(this->cb_[i]);
        }
    };

    /**
     * @brief Resets the context for reuse (no state to reset).
     */
    void
    Reset() override{};

    /**
     * @brief Returns the memory usage of this context.
     *
     * Only counts heap-allocated iocb structures, as events_ array is inline member.
     *
     * @return Memory usage in bytes (sizeof(IOContext) plus heap-allocated iocb pointers).
     */
    int64_t
    GetMemoryUsage() const override {
        return sizeof(IOContext) + DEFAULT_REQUEST_COUNT * sizeof(struct iocb);
    }

public:
    /// Default number of concurrent AIO requests supported.
    static constexpr int64_t DEFAULT_REQUEST_COUNT = 100;

    /// The libaio context handle.
    io_context_t ctx_;

    /// Array of IO control blocks for submitting requests.
    struct iocb* cb_[DEFAULT_REQUEST_COUNT];

    /// Array of IO events for receiving completion notifications.
    struct io_event events_[DEFAULT_REQUEST_COUNT];
};

/// Pool type for managing reusable IOContext objects.
using IOContextPool = ResourceObjectPool<IOContext>;

}  // namespace vsag

#endif  // HAVE_LIBAIO
