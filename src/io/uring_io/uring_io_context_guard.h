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

#include <memory>

#include "io/uring_io/uring_io_context.h"

namespace vsag {

class UringIOContextGuard {
public:
    explicit UringIOContextGuard(std::shared_ptr<UringIOContext> ctx);

    ~UringIOContextGuard();

    UringIOContextGuard(const UringIOContextGuard&) = delete;
    UringIOContextGuard&
    operator=(const UringIOContextGuard&) = delete;

    void
    Abandon();

    void
    Drop();

private:
    std::shared_ptr<UringIOContext> ctx_{nullptr};
};

}  // namespace vsag

#endif  // HAVE_LIBURING
