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

#include <algorithm>
#include <limits>
#include <memory>

#include "byte_buffer.h"
#include "extra_info_interface.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "quantization/quantizer.h"

namespace vsag {
/*
* thread unsafe
*/
template <typename IOTmpl>
class ExtraInfoDataCell : public ExtraInfoInterface {
public:
    ExtraInfoDataCell() = default;

    explicit ExtraInfoDataCell(const IOParamPtr& io_param, const IndexCommonParam& common_param);

    void
    InsertExtraInfo(const char* extra_info, InnerIdType idx) override;

    void
    BatchInsertExtraInfo(const char* extra_infos, InnerIdType count, InnerIdType* idx) override;

    void
    SetMaxCapacity(InnerIdType capacity) override {
        this->max_capacity_ = std::max(capacity, this->total_count_);
    }

    InnerIdType
    GetMaxCapacity() override {
        this->max_capacity_ = std::max(this->max_capacity_, this->total_count_);
        return this->max_capacity_;
    };

    void
    Prefetch(InnerIdType id) override {
        if (this->force_in_memory_) {
            force_in_memory_io_->Prefetch(id * extra_info_size_, extra_info_size_);
        } else {
            io_->Prefetch(id * extra_info_size_, extra_info_size_);
        }
    };

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->max_capacity_ = new_capacity;
        uint64_t io_size =
            static_cast<uint64_t>(new_capacity) * static_cast<uint64_t>(extra_info_size_);
        uint8_t end_flag =
            127;  // the value is meaningless, only to occupy the position for io allocate
        this->io_->Write(&end_flag, 1, io_size);
        if (force_in_memory_) {
            this->force_in_memory_io_->Write(&end_flag, 1, io_size);
        }
    }

    void
    Release(const char* extra_info) override {
        if (extra_info == nullptr) {
            return;
        }
        if (this->force_in_memory_) {
            force_in_memory_io_->Release(reinterpret_cast<const uint8_t*>(extra_info));
        } else {
            io_->Release(reinterpret_cast<const uint8_t*>(extra_info));
        }
    }

    [[nodiscard]] bool
    InMemory() const override;

    void
    EnableForceInMemory() override;

    void
    DisableForceInMemory() override;

    bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const override;

    const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

public:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    void
    trans_from_memory_io();

private:
    bool force_in_memory_{false};

    std::shared_ptr<MemoryBlockIO> force_in_memory_io_{};
};

template <typename IOTmpl>
ExtraInfoDataCell<IOTmpl>::ExtraInfoDataCell(const IOParamPtr& io_param,
                                             const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->extra_info_size_ = common_param.extra_info_size_;
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->force_in_memory_io_ =
        std::make_shared<MemoryBlockIO>(allocator_, Options::Instance().block_size_limit());
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::InsertExtraInfo(const char* extra_info, InnerIdType idx) {
    if (idx == std::numeric_limits<InnerIdType>::max()) {
        idx = total_count_;
        ++total_count_;
    } else {
        total_count_ = std::max(total_count_, idx + 1);
    }

    if (this->force_in_memory_) {
        force_in_memory_io_->Write(
            reinterpret_cast<const uint8_t*>(extra_info),
            extra_info_size_,
            static_cast<uint64_t>(idx) * static_cast<uint64_t>(extra_info_size_));
    } else {
        io_->Write(reinterpret_cast<const uint8_t*>(extra_info),
                   extra_info_size_,
                   static_cast<uint64_t>(idx) * static_cast<uint64_t>(extra_info_size_));
    }
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::BatchInsertExtraInfo(const char* extra_infos,
                                                InnerIdType count,
                                                InnerIdType* idx) {
    if (idx == nullptr) {
        // length of extra info is fixed currently
        if (this->force_in_memory_) {
            force_in_memory_io_->Write(
                reinterpret_cast<const uint8_t*>(extra_infos),
                static_cast<uint64_t>(count) * static_cast<uint64_t>(extra_info_size_),
                static_cast<uint64_t>(total_count_) * static_cast<uint64_t>(extra_info_size_));
        } else {
            io_->Write(
                reinterpret_cast<const uint8_t*>(extra_infos),
                static_cast<uint64_t>(count) * static_cast<uint64_t>(extra_info_size_),
                static_cast<uint64_t>(total_count_) * static_cast<uint64_t>(extra_info_size_));
        }
        total_count_ += count;
    } else {
        for (int64_t i = 0; i < count; ++i) {
            this->InsertExtraInfo(extra_infos + extra_info_size_ * i, idx[i]);
        }
    }
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::trans_from_memory_io() {
    int64_t max_size = static_cast<int64_t>(this->extra_info_size_) * this->total_count_;
    constexpr uint64_t block_size = 1024 * 1024;
    uint64_t offset = 0;
    bool need_release = false;
    while (max_size > block_size) {
        auto data = this->force_in_memory_io_->Read(block_size, offset, need_release);
        this->io_->Write(data, block_size, offset);
        max_size -= block_size;
        offset += block_size;
        if (need_release) {
            this->force_in_memory_io_->Release(data);
        }
    }
    if (max_size > 0) {
        auto data = this->force_in_memory_io_->Read(max_size, offset, need_release);
        this->io_->Write(data, max_size, offset);
        if (need_release) {
            this->force_in_memory_io_->Release(data);
        }
    }
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::EnableForceInMemory() {
    if (this->TotalCount() != 0) {
        throw std::runtime_error("EnableForceInMemory must with empty extra info datacell");
    }
    this->force_in_memory_ = true;
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::DisableForceInMemory() {
    this->force_in_memory_ = false;
    this->trans_from_memory_io();
}

template <typename IOTmpl>
bool
ExtraInfoDataCell<IOTmpl>::InMemory() const {
    return this->io_->InMemory();
}

template <typename IOTmpl>
bool
ExtraInfoDataCell<IOTmpl>::GetExtraInfoById(InnerIdType id, char* extra_info) const {
    if (force_in_memory_) {
        return force_in_memory_io_->Read(
            extra_info_size_,
            static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
            reinterpret_cast<uint8_t*>(extra_info));
    } else {
        return io_->Read(extra_info_size_,
                         static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
                         reinterpret_cast<uint8_t*>(extra_info));
    }
}

template <typename IOTmpl>
const char*
ExtraInfoDataCell<IOTmpl>::GetExtraInfoById(InnerIdType id, bool& need_release) const {
    if (force_in_memory_) {
        return reinterpret_cast<const char*>(force_in_memory_io_->Read(
            extra_info_size_,
            static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
            need_release));
    } else {
        return reinterpret_cast<const char*>(
            io_->Read(extra_info_size_,
                      static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
                      need_release));
    }
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::Serialize(StreamWriter& writer) {
    ExtraInfoInterface::Serialize(writer);
    this->io_->Serialize(writer);
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::Deserialize(StreamReader& reader) {
    ExtraInfoInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
}
}  // namespace vsag
