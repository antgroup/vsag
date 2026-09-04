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

#include "io/common/io_parameter.h"

#include "inner_string_params.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("IOParameter maps profile names to stable kinds", "[ut][IOParameter]") {
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_MEMORY_IO) == IOKind::MEMORY);
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_BLOCK_MEMORY_IO) == IOKind::BLOCK_MEMORY);
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_MMAP_IO) == IOKind::MMAP);
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_BUFFER_IO) == IOKind::BUFFER);
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_ASYNC_IO) == IOKind::ASYNC);
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_URING_IO) == IOKind::URING);
    REQUIRE(IOParameter::KindFromName(IO_TYPE_VALUE_READER_IO) == IOKind::READER);
    REQUIRE(IOParameter::KindFromName("unknown_io") == IOKind::UNKNOWN);
}
