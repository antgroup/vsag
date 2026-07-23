
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
#include "index_common_param.h"
#include "io/common/basic_io.h"
#include "io/uring_io/uring_io_context.h"
#include "io/uring_io/uring_io_parameter.h"
namespace vsag {

class UringIO : public BasicIO<UringIO> {
public:
    static constexpr bool InMemory = false;
    static constexpr bool SkipDeserialize = false;

public:
    explicit UringIO(std::string filename, Allocator* allocator, bool direct_read = false);

    explicit UringIO(const UringIOParameterPtr& io_param, const IndexCommonParam& common_param);

    explicit UringIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    ~UringIO();

public:
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    void
    ResizeImpl(uint64_t size);

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    void
    ReleaseImpl(const uint8_t* data) const;

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

public:
    static std::unique_ptr<UringIOContextPool> io_context_pool;

private:
    std::string filepath_{};

    int rfd_{-1};

    int wfd_{-1};

    bool exist_file_{false};

    bool direct_read_{false};
};

}  // namespace vsag

#else
#include "io/buffer_io/buffer_io.h"
namespace vsag {
using UringIO = BufferIO;
}  // namespace vsag
#endif  // HAVE_LIBURING
