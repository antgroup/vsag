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

#include "extra_info_interface.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "quantization/quantizer.h"
#include "utils/byte_buffer.h"

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
    Prefetch(InnerIdType id) override {
        io_->Prefetch(id * extra_info_size_, extra_info_size_);
    };

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->max_capacity_ = new_capacity;
        uint64_t io_size =
            static_cast<uint64_t>(new_capacity) * static_cast<uint64_t>(extra_info_size_);
        this->io_->Resize(io_size);
    }

    void
    Release(const char* extra_info) override {
        if (extra_info == nullptr) {
            return;
        }
        io_->Release(reinterpret_cast<const uint8_t*>(extra_info));
    }

    [[nodiscard]] bool
    InMemory() const override;

    bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const override;

    const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    int64_t
    GetMemoryUsage() const override;

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

public:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};
};

template <typename IOTmpl>
ExtraInfoDataCell<IOTmpl>::ExtraInfoDataCell(const IOParamPtr& io_param,
                                             const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->extra_info_size_ = common_param.extra_info_size_;
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
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
    io_->Write(reinterpret_cast<const uint8_t*>(extra_info),
               extra_info_size_,
               static_cast<uint64_t>(idx) * static_cast<uint64_t>(extra_info_size_));
}

template <typename IOTmpl>
void
ExtraInfoDataCell<IOTmpl>::BatchInsertExtraInfo(const char* extra_infos,
                                                InnerIdType count,
                                                InnerIdType* idx) {
    if (idx == nullptr) {
        // length of extra info is fixed currently
        io_->Write(reinterpret_cast<const uint8_t*>(extra_infos),
                   static_cast<uint64_t>(count) * static_cast<uint64_t>(extra_info_size_),
                   static_cast<uint64_t>(total_count_) * static_cast<uint64_t>(extra_info_size_));

        total_count_ += count;
    } else {
        for (int64_t i = 0; i < count; ++i) {
            this->InsertExtraInfo(extra_infos + extra_info_size_ * i, idx[i]);
        }
    }
}

template <typename IOTmpl>
bool
ExtraInfoDataCell<IOTmpl>::InMemory() const {
    return IOTmpl::InMemory;
}

template <typename IOTmpl>
bool
ExtraInfoDataCell<IOTmpl>::GetExtraInfoById(InnerIdType id, char* extra_info) const {
    return io_->Read(extra_info_size_,
                     static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
                     reinterpret_cast<uint8_t*>(extra_info));
}

template <typename IOTmpl>
const char*
ExtraInfoDataCell<IOTmpl>::GetExtraInfoById(InnerIdType id, bool& need_release) const {
    return reinterpret_cast<const char*>(
        io_->Read(extra_info_size_,
                  static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
                  need_release));
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

template <typename IOTmpl>
int64_t
ExtraInfoDataCell<IOTmpl>::GetMemoryUsage() const {
    int64_t memory = sizeof(ExtraInfoDataCell<IOTmpl>);
    if (IOTmpl::InMemory) {
        memory += this->io_->GetMemoryUsage();
    }
    return memory;
}
}  // namespace vsag
