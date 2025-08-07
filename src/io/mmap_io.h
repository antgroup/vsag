
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

#include <sys/mman.h>

#include <filesystem>
#include <utility>

#include "basic_io.h"
#include "index/index_common_param.h"
#include "mmap_io_parameter.h"

namespace vsag {

class MMapIO : public BasicIO<MMapIO> {
public:
    static constexpr bool InMemory = false;

public:
    MMapIO(std::string filename, Allocator* allocator);

    explicit MMapIO(const MMapIOParameterPtr& io_param, const IndexCommonParam& common_param);

    explicit MMapIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    ~MMapIO() override;

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    static constexpr int64_t DEFAULT_INIT_MMAP_SIZE = 4096;

private:
    std::string filepath_{};

    int fd_{-1};

    uint8_t* start_{nullptr};

    bool exist_file_{false};
};
}  // namespace vsag
