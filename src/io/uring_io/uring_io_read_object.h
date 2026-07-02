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

namespace vsag {

static constexpr uint64_t DIRECT_IO_ALIGN = 4096;

void*
AlignedAlloc(uint64_t size, uint64_t alignment = DIRECT_IO_ALIGN);

void
AlignedFree(void* ptr);

class UringReadObject {
public:
    UringReadObject() = default;

    void
    Set(uint64_t size1,
        uint64_t offset1,
        uint8_t* dest_data1,
        bool do_align,
        uint64_t align = DIRECT_IO_ALIGN);

    bool
    AllocAligned(uint64_t align = DIRECT_IO_ALIGN);

    void
    CopyToDest();

    uint8_t* dest_data{nullptr};
    uint8_t* aligned_buf{nullptr};
    uint64_t orig_size{0};
    uint64_t orig_offset{0};
    uint64_t submit_offset{0};
    uint64_t submit_size{0};
    uint64_t prefix{0};
    bool released{true};
};

void
ReleaseUringReadObject(UringReadObject& obj);

}  // namespace vsag

#endif  // HAVE_LIBURING
