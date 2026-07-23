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

#include <cstdint>

#include "utils/resource_object.h"
#include "utils/resource_object_pool.h"

struct io_uring;

namespace vsag {

class UringIOContext : public ResourceObject {
public:
    static constexpr uint32_t RING_SIZE = 512;

    UringIOContext();
    ~UringIOContext() override;

    void
    Reset() override {
    }

    uint64_t
    GetMemoryUsage() const override;

    io_uring*
    ring();

private:
    // Pointer storage + forward declaration keeps liburing.h (and its heavy
    // Linux kernel uapi headers) out of every TU that only includes this
    // header. The ring is allocated/freed in uring_io_context.cpp.
    io_uring* ring_{nullptr};
};

using UringIOContextPool = ResourceObjectPool<UringIOContext>;

}  // namespace vsag

#endif  // HAVE_LIBURING
