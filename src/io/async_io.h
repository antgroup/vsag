
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

#include "async_io_parameter.h"
#include "basic_io.h"
#include "index/index_common_param.h"
#include "io_context.h"

namespace vsag {

class DirectIOObject {
public:
    DirectIOObject() = default;

    DirectIOObject(uint64_t size, uint64_t offset) {
        this->Set(size, offset);
    }

    void
    Set(uint64_t size1, uint64_t offset1) {
        this->size = size1;
        this->offset = offset1;
        if (align_data) {
            free(align_data);
        }
        auto new_offset = (offset >> ALIGN_BIT) << ALIGN_BIT;
        auto inner_offset = offset & ALIGN_MASK;
        auto new_size = (((size + inner_offset) + ALIGN_MASK) >> ALIGN_BIT) << ALIGN_BIT;
        this->align_data = static_cast<uint8_t*>(std::aligned_alloc(ALIGN_SIZE, new_size));
        this->data = align_data + inner_offset;
        this->size = new_size;
        this->offset = new_offset;
    }

public:
    uint8_t* data{nullptr};
    uint64_t size;
    uint64_t offset;
    uint8_t* align_data{nullptr};

    static constexpr int64_t ALIGN_BIT = 9;

    static constexpr int64_t ALIGN_SIZE = 1 << ALIGN_BIT;

    static constexpr int64_t ALIGN_MASK = (1 << ALIGN_BIT) - 1;
};

class AsyncIO : public BasicIO<AsyncIO> {
public:
    AsyncIO(std::string filename, Allocator* allocator)
        : BasicIO<AsyncIO>(allocator), filepath_(std::move(filename)) {
        this->rfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
        this->wfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    }

    explicit AsyncIO(const AsyncIOParameterPtr& io_param, const IndexCommonParam& common_param)
        : AsyncIO(io_param->path_, common_param.allocator_.get()){};

    explicit AsyncIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : AsyncIO(std::dynamic_pointer_cast<AsyncIOParameter>(param), common_param){};

    ~AsyncIO() override = default;

public:
    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        auto ret = pwrite64(this->wfd_, data, size, offset);
        if (ret != size) {
            throw std::runtime_error(fmt::format("write bytes {} less than {}", ret, size));
        }
        if (size + offset > this->size_) {
            this->size_ = size + offset;
        }
        fsync(wfd_);
    }

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        bool need_release = true;
        auto ptr = DirectReadImpl(size, offset, need_release);
        memcpy(data, ptr, size);
        this->ReleaseImpl(ptr);
        return true;
    }

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = true;
        if (size == 0) {
            return nullptr;
        }
        DirectIOObject obj(size, offset);
        auto ret = pread64(this->rfd_, obj.align_data, obj.size, obj.offset);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("pread64 error {}", ret));
        }
        return obj.data;
    }

    inline void
    ReleaseImpl(const uint8_t* data) const {
        auto ptr = const_cast<uint8_t*>(data);
        constexpr auto ALIGN_BIT = DirectIOObject::ALIGN_BIT;
        free(reinterpret_cast<void*>((reinterpret_cast<uint64_t>(ptr) >> ALIGN_BIT) << ALIGN_BIT));
    }

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        auto context = io_context_pool->TakeOne();
        uint8_t* cur_data = datas;
        int64_t all_count = count;
        while (all_count > 0) {
            count = std::min(IOContext::DEFAULT_REQUEST_COUNT, all_count);
            auto* cb = context->cb_;
            std::vector<DirectIOObject> objs(count);
            for (int64_t i = 0; i < count; ++i) {
                objs[i].Set(sizes[i], offsets[i]);
                auto& obj = objs[i];
                io_prep_pread(cb[i], rfd_, obj.align_data, obj.size, obj.offset);
                cb[i]->data = &(objs[i]);
            }

            int submitted = io_submit(context->ctx_, count, cb);
            if (submitted < 0) {
                io_context_pool->ReturnOne(context);
                throw std::runtime_error("io submit failed");
            }

            struct timespec timeout = {1, 0};
            auto num_events = io_getevents(context->ctx_, count, count, context->events_, &timeout);
            if (num_events != count) {
                io_context_pool->ReturnOne(context);
                throw std::runtime_error("io async read failed");
            }

            for (int64_t i = 0; i < count; ++i) {
                memcpy(cur_data, objs[i].data, sizes[i]);
                cur_data += sizes[i];
                this->ReleaseImpl(objs[i].data);
            }

            sizes += count;
            offsets += count;
            all_count -= count;
        }
        io_context_pool->ReturnOne(context);
        return true;
    }

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64){};

    static inline bool
    InMemoryImpl() {
        return false;
    }

public:
    static std::unique_ptr<IOContextPool> io_context_pool;

private:
    std::string filepath_{};

    int rfd_{-1};

    int wfd_{-1};
};
}  // namespace vsag
