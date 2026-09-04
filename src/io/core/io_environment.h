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

#pragma once

#include "vsag/allocator.h"

#if HAVE_LIBAIO
#include "io/async_io/aio_context.h"
#endif

#if HAVE_LIBURING
#include "io/uring_io/uring_io_context.h"
#endif

namespace vsag {

struct IOEnvironment {
    Allocator* allocator{nullptr};
#if HAVE_LIBAIO
    IOContextPool* aio_context_pool{nullptr};
#endif
#if HAVE_LIBURING
    UringIOContextPool* uring_context_pool{nullptr};
#endif
    bool direct_read{false};
};

#if HAVE_LIBAIO
inline IOContextPool&
DefaultAioContextPool() {
    static IOContextPool pool(10, nullptr);
    return pool;
}
#endif

#if HAVE_LIBURING
inline UringIOContextPool&
DefaultUringContextPool() {
    static UringIOContextPool pool(0, nullptr);
    return pool;
}
#endif

inline IOEnvironment
MakeDefaultIOEnvironment(Allocator* allocator) {
    IOEnvironment environment;
    environment.allocator = allocator;
#if HAVE_LIBAIO
    environment.aio_context_pool = &DefaultAioContextPool();
#endif
#if HAVE_LIBURING
    environment.uring_context_pool = &DefaultUringContextPool();
#endif
    return environment;
}

}  // namespace vsag
