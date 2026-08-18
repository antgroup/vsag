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

#include "io/uring_io/uring_io_context_guard.h"

#include <mutex>
#include <vector>

#include "io/uring_io/uring_io.h"

namespace vsag {

UringIOContextGuard::UringIOContextGuard(std::shared_ptr<UringIOContext> ctx)
    : ctx_(std::move(ctx)) {
}

UringIOContextGuard::~UringIOContextGuard() {
    if (ctx_ != nullptr) {
        UringIO::io_context_pool->ReturnOne(ctx_);
    }
}

void
UringIOContextGuard::Abandon() {
    static std::mutex abandon_mutex;
    static std::vector<std::shared_ptr<UringIOContext>> abandoned;
    std::lock_guard<std::mutex> lock(abandon_mutex);
    abandoned.push_back(std::move(ctx_));
    ctx_ = nullptr;
}

void
UringIOContextGuard::Drop() {
    ctx_.reset();
}

}  // namespace vsag

#endif  // HAVE_LIBURING
