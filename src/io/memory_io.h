
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

#include <cstring>
#include <nlohmann/json.hpp>

#include "basic_io.h"
#include "index/index_common_param.h"
#include "memory_io_parameter.h"
#include "prefetch.h"
#include "vsag/allocator.h"

namespace vsag {

class MemoryIO : public BasicIO<MemoryIO> {
public:
    explicit MemoryIO(Allocator* allocator) : BasicIO<MemoryIO>(allocator) {
        start_ = static_cast<uint8_t*>(allocator->Allocate(1));
    }

    explicit MemoryIO(const MemoryIOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryIO(common_param.allocator_.get()) {
    }

    explicit MemoryIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryIO(std::dynamic_pointer_cast<MemoryIOParameter>(param), common_param) {
    }

    ~MemoryIO() override {
        this->allocator_->Deallocate(start_);
    }

    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    inline void
    ReleaseImpl(const uint8_t* data) const {};

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

    static inline bool
    InMemoryImpl() {
        return true;
    }

    void
    InitIOImpl(const IOParamPtr& io_param) {
    }

private:
    void
    check_and_realloc(uint64_t size) {
        if (size <= this->size_) {
            return;
        }
        start_ = reinterpret_cast<uint8_t*>(this->allocator_->Reallocate(start_, size));
        this->size_ = size;
    }

private:
    uint8_t* start_{nullptr};
};

void
MemoryIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    check_and_realloc(size + offset);
    memcpy(start_ + offset, data, size);
}

bool
MemoryIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    bool ret = check_valid_offset(size + offset);
    if (ret) {
        memcpy(data, start_ + offset, size);
    }
    return ret;
}

const uint8_t*
MemoryIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    need_release = false;
    if (check_valid_offset(size + offset)) {
        return start_ + offset;
    }
    return nullptr;
}
bool
MemoryIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    bool ret = true;
    for (uint64_t i = 0; i < count; ++i) {
        ret &= this->ReadImpl(sizes[i], offsets[i], datas);
        datas += sizes[i];
    }
    return ret;
}
void
MemoryIO::PrefetchImpl(uint64_t offset, uint64_t cache_line) {
    PrefetchLines(this->start_ + offset, cache_line);
}
}  // namespace vsag
