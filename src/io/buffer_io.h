
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

#include <utility>

#include "basic_io.h"
#include "buffer_io_parameter.h"
#include "index/index_common_param.h"

namespace vsag {

class BufferIO : public BasicIO<BufferIO> {
public:
    BufferIO(std::string filename, Allocator* allocator)
        : BasicIO<BufferIO>(allocator), filepath_(std::move(filename)) {
        this->fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    }

    explicit BufferIO(const BufferIOParameterPtr& io_param, const IndexCommonParam& common_param)
        : BufferIO(io_param->path_, common_param.allocator_.get()){};

    explicit BufferIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : BufferIO(std::dynamic_pointer_cast<BufferIOParameter>(param), common_param){};

    ~BufferIO() override = default;

    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        auto ret = pwrite64(this->fd_, data, size, offset);
        if (ret != size) {
            throw std::runtime_error(fmt::format("write bytes {} less than {}", ret, size));
        }
        if (size + offset > this->size_) {
            this->size_ = size + offset;
        }
    }

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        if (size == 0) {
            return true;
        }
        auto ret = pread64(this->fd_, data, size, offset);
        if (ret != size) {
            throw std::runtime_error(fmt::format("read bytes {} less than {}", ret, size));
        }
        return true;
    }

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = true;
        auto* buf = reinterpret_cast<uint8_t*>(allocator_->Allocate(size));
        ReadImpl(size, offset, buf);
        return buf;
    }

    inline void
    ReleaseImpl(const uint8_t* data) const {
        auto ptr = const_cast<uint8_t*>(data);
        allocator_->Deallocate(ptr);
    }

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        bool ret = true;
        for (uint64_t i = 0; i < count; ++i) {
            ret &= ReadImpl(sizes[i], offsets[i], datas);
            datas += sizes[i];
        }
        return ret;
    }

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64){};

    static inline bool
    InMemoryImpl() {
        return false;
    }

private:
    std::string filepath_{};

    int fd_{-1};
};
}  // namespace vsag
