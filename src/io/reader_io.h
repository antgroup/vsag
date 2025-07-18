
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

#include "basic_io.h"
#include "index/index_common_param.h"
#include "reader_io_parameter.h"
#include "vsag/allocator.h"

namespace vsag {

class ReaderIO : public BasicIO<ReaderIO> {
public:
    explicit ReaderIO(Allocator* allocator) : BasicIO<ReaderIO>(allocator) {
    }

    explicit ReaderIO(const ReaderIOParamPtr& param, const IndexCommonParam& common_param)
        : ReaderIO(common_param.allocator_.get()) {
    }

    explicit ReaderIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : ReaderIO(std::dynamic_pointer_cast<ReaderIOParameter>(param), common_param) {
    }

    ~ReaderIO() override = default;

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    void
    InitIOImpl(const IOParamPtr& io_param);

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    void
    ReleaseImpl(const uint8_t* data) const;

    bool
    MultiReadImpl(uint8_t* datas,
                  const uint64_t* sizes,
                  const uint64_t* offsets,
                  uint64_t count) const;

    void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

    static inline bool
    InMemoryImpl() {
        return false;
    }

private:
    std::shared_ptr<Reader> reader_{nullptr};
};

}  // namespace vsag
