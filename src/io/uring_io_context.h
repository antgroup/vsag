
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
#if HAVE_LIBURING

#include <fmt/format.h>

#include <cstring>

#include "liburing.h"
#include "utils/resource_object.h"
#include "utils/resource_object_pool.h"
#include "vsag_exception.h"
namespace vsag {
class UringIOContext : public ResourceObject {
public:
    static constexpr uint32_t RING_SIZE = 512;

    UringIOContext() {
        int ret = io_uring_queue_init(RING_SIZE, &ring_, 0);
        if (ret < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("io_uring_queue_init failed: {}", strerror(-ret)));
        }
    }

    ~UringIOContext() {
        io_uring_queue_exit(&ring_);
    }

    void
    Reset() override{};

    int64_t
    MemoryUsage() const override {
        return sizeof(UringIOContext);
    }

    io_uring*
    ring() {
        return &ring_;
    }

private:
    io_uring ring_;
};

using UringIOContextPool = ResourceObjectPool<UringIOContext>;

}  // namespace vsag

#endif  // HAVE_LIBURING
