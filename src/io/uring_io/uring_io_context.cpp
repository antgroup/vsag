
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

#include "io/uring_io/uring_io_context.h"

#include <fmt/format.h>
#include <liburing.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "vsag_exception.h"

namespace vsag {

UringIOContext::UringIOContext() {
    ring_ = new io_uring;
    int ret = io_uring_queue_init(RING_SIZE, ring_, 0);
    if (ret < 0) {
        delete ring_;
        ring_ = nullptr;
        int err = -ret;
        std::string err_name;
        if (err == ENOSYS)
            err_name = "ENOSYS";
        else if (err == EPERM)
            err_name = "EPERM";
        else if (err == EACCES)
            err_name = "EACCES";
        else if (err == ENOMEM)
            err_name = "ENOMEM";
        else if (err == EMFILE)
            err_name = "EMFILE";
        else if (err == EAGAIN)
            err_name = "EAGAIN";
        else
            err_name = "errno_" + std::to_string(err);
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("io_uring_queue_init failed: {} ({})", strerror(err), err_name));
    }
}

UringIOContext::~UringIOContext() {
    if (ring_ != nullptr) {
        io_uring_queue_exit(ring_);
        delete ring_;
        ring_ = nullptr;
    }
}

uint64_t
UringIOContext::GetMemoryUsage() const {
    return sizeof(UringIOContext) + sizeof(struct io_uring) +
           RING_SIZE * sizeof(struct io_uring_sqe) + RING_SIZE * sizeof(struct io_uring_cqe);
}

io_uring*
UringIOContext::ring() {
    return ring_;
}

}  // namespace vsag

#endif  // HAVE_LIBURING
