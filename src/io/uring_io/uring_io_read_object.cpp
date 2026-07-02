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

#include "io/uring_io/uring_io_read_object.h"

#include <cstdlib>
#include <cstring>

namespace vsag {

void*
AlignedAlloc(uint64_t size, uint64_t alignment) {
    uint64_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
        return nullptr;
    }
    return ptr;
}

void
AlignedFree(void* ptr) {
    free(ptr);
}

void
UringReadObject::Set(
    uint64_t size1, uint64_t offset1, uint8_t* dest_data1, bool do_align, uint64_t align) {
    this->dest_data = dest_data1;
    this->orig_size = size1;
    this->orig_offset = offset1;
    this->aligned_buf = nullptr;
    if (do_align && size1 > 0) {
        uint64_t align_mask = align - 1;
        this->submit_offset = offset1 & ~align_mask;
        uint64_t prefix = offset1 - this->submit_offset;
        this->submit_size = (prefix + size1 + align_mask) & ~align_mask;
        this->prefix = prefix;
    } else {
        this->submit_offset = offset1;
        this->submit_size = size1;
        this->prefix = 0;
    }
}

bool
UringReadObject::AllocAligned(uint64_t align) {
    this->aligned_buf = static_cast<uint8_t*>(AlignedAlloc(this->submit_size, align));
    return this->aligned_buf != nullptr;
}

void
UringReadObject::CopyToDest() {
    if (this->aligned_buf != nullptr && this->dest_data != nullptr) {
        memcpy(this->dest_data, this->aligned_buf + this->prefix, this->orig_size);
    }
}

void
ReleaseUringReadObject(UringReadObject& obj) {
    if (not obj.released) {
        if (obj.aligned_buf != nullptr) {
            AlignedFree(obj.aligned_buf);
            obj.aligned_buf = nullptr;
        }
        obj.released = true;
    }
}

}  // namespace vsag

#endif  // HAVE_LIBURING
