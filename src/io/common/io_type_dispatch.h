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

#include <type_traits>
#include <utility>

#include "io/async_io/async_io.h"
#include "io/buffer_io/buffer_io.h"
#include "io/common/io_parameter.h"
#include "io/memory_block_io/memory_block_io.h"
#include "io/memory_io/memory_io.h"
#include "io/mmap_io/mmap_io.h"
#include "io/reader_io/reader_io.h"
#include "io/uring_io/uring_io.h"
#include "io/user_defined_io/user_defined_io.h"
#include "vsag_exception.h"

namespace vsag {

template <typename IO>
struct IOTypeTag {
    using Type = IO;
};

// NOLINTBEGIN(readability-identifier-naming)
template <typename Visitor>
decltype(auto)
VisitIOKind(IOKind kind, Visitor&& visitor) {
    switch (kind) {
        case IOKind::MEMORY:
            return std::forward<Visitor>(visitor)(IOTypeTag<MemoryIO>{});
        case IOKind::BLOCK_MEMORY:
            return std::forward<Visitor>(visitor)(IOTypeTag<MemoryBlockIO>{});
        case IOKind::MMAP:
            return std::forward<Visitor>(visitor)(IOTypeTag<MMapIO>{});
        case IOKind::BUFFER:
            return std::forward<Visitor>(visitor)(IOTypeTag<BufferIO>{});
        case IOKind::ASYNC:
            return std::forward<Visitor>(visitor)(IOTypeTag<AsyncIO>{});
        case IOKind::URING:
            return std::forward<Visitor>(visitor)(IOTypeTag<UringIO>{});
        case IOKind::READER:
            return std::forward<Visitor>(visitor)(IOTypeTag<ReaderIO>{});
        // Intentionally not instantiable through generic visitors: user defined IO is constructed
        // by the precise FP32 factory path only, to avoid instantiating unrelated graph/quantizer
        // combinations. Failing loudly here surfaces accidental generic use instead of a silent void.
        case IOKind::USER_DEFINED:
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "user_defined_io is not instantiable through generic IO dispatch");
        case IOKind::UNKNOWN:
            return std::forward<Visitor>(visitor)(IOTypeTag<void>{});
    }
    return std::forward<Visitor>(visitor)(IOTypeTag<void>{});
}

}  // namespace vsag
// NOLINTEND(readability-identifier-naming)
