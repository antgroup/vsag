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

#include "io/common/io_type_dispatch.h"

#include <type_traits>

#include "unittest.h"

namespace vsag {

TEST_CASE("VisitIOKind maps every stable IO kind", "[ut][IOTypeDispatch]") {
    auto mapped_kind = [](auto tag) {
        using IO = typename decltype(tag)::Type;
        if constexpr (std::is_same_v<IO, MemoryIO>) {
            return IOKind::MEMORY;
        } else if constexpr (std::is_same_v<IO, MemoryBlockIO>) {
            return IOKind::BLOCK_MEMORY;
        } else if constexpr (std::is_same_v<IO, MMapIO>) {
            return IOKind::MMAP;
        } else if constexpr (std::is_same_v<IO, BufferIO>) {
            return IOKind::BUFFER;
        } else if constexpr (std::is_same_v<IO, AsyncIO>) {
            return IOKind::ASYNC;
        } else if constexpr (std::is_same_v<IO, UringIO>) {
            return IOKind::URING;
        } else if constexpr (std::is_same_v<IO, ReaderIO>) {
            return IOKind::READER;
        } else {
            return IOKind::UNKNOWN;
        }
    };

    for (const auto kind : {IOKind::MEMORY,
                            IOKind::BLOCK_MEMORY,
                            IOKind::MMAP,
                            IOKind::BUFFER,
                            IOKind::ASYNC,
                            IOKind::URING,
                            IOKind::READER,
                            IOKind::UNKNOWN}) {
        REQUIRE(VisitIOKind(kind, mapped_kind) == kind);
    }
}

}  // namespace vsag
