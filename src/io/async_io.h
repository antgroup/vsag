
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

#include <filesystem>

#include "basic_io.h"
#include "io_context.h"

namespace vsag {

class AsyncIOParameter;
class IndexCommonParam;
class Allocator;

using AsyncIOParameterPtr = std::shared_ptr<AsyncIOParameter>;

#if HAVE_LIBAIO
class AsyncIO : public BasicIO<AsyncIO> {
public:
    static constexpr bool InMemory = false;
    static constexpr bool SkipDeserialize = false;

public:
    explicit AsyncIO(std::filesystem::path filepath, Allocator* allocator);

    explicit AsyncIO(const AsyncIOParameterPtr& io_param, const IndexCommonParam& common_param);

    explicit AsyncIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    ~AsyncIO() override;

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

private:
    std::filesystem::path filepath_;

    int rfd_{-1};

    int wfd_{-1};

    bool exist_file_{false};

    static std::unique_ptr<IOContextPool> io_context_pool;
};

#else
#define AsyncIO BufferIO
#endif

}  // namespace vsag
