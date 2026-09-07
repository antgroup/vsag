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
#include <utility>

#include "extra_info_interface.h"
#include "layout/fixed_layout.h"

namespace vsag {
/*
* thread unsafe
*/
template <typename LayoutTmpl>
class ExtraInfoDataCell : public ExtraInfoInterface {
public:
    ExtraInfoDataCell() : layout_(std::make_shared<LayoutTmpl>()) {
    }

    explicit ExtraInfoDataCell(const IOParamPtr& io_param, const IndexCommonParam& common_param);

    void
    InsertExtraInfo(const char* extra_info, InnerIdType idx) override;

    void
    BatchInsertExtraInfo(const char* extra_infos, InnerIdType count, InnerIdType* idx) override;

    void
    Prefetch(InnerIdType id) override {
        layout_->Prefetch(id, extra_info_size_);
    };

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->layout_->Resize(new_capacity);
        this->max_capacity_ = new_capacity;
    }

    void
    Release(const char* extra_info) const override {
        if (extra_info == nullptr) {
            return;
        }
        layout_->Release(reinterpret_cast<const uint8_t*>(extra_info));
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

    uint64_t
    GetMemoryUsage() const override;

    void
    Move(InnerIdType from, InnerIdType to) override;

    void
    ShrinkToFit(InnerIdType capacity) override {
        this->layout_->Shrink(capacity);
        this->max_capacity_ = capacity;
        this->total_count_ = std::min(this->total_count_, capacity);
    }

    inline void
    SetIO(std::shared_ptr<typename LayoutTmpl::IOType> io) {
        this->layout_->SetIO(std::move(io));
    }

public:
    std::shared_ptr<LayoutTmpl> layout_{nullptr};

    Allocator* const allocator_{nullptr};
};

template <typename LayoutTmpl>
ExtraInfoDataCell<LayoutTmpl>::ExtraInfoDataCell(const IOParamPtr& io_param,
                                                 const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->extra_info_size_ = common_param.extra_info_size_;
    this->layout_ = std::make_shared<LayoutTmpl>(this->extra_info_size_, io_param, common_param);
}

template <typename LayoutTmpl>
void
ExtraInfoDataCell<LayoutTmpl>::InsertExtraInfo(const char* extra_info, InnerIdType idx) {
    if (idx == std::numeric_limits<InnerIdType>::max()) {
        idx = total_count_;
        ++total_count_;
    } else {
        total_count_ = std::max(total_count_, idx + 1);
    }
    layout_->Write(idx, reinterpret_cast<const uint8_t*>(extra_info));
}

template <typename LayoutTmpl>
void
ExtraInfoDataCell<LayoutTmpl>::BatchInsertExtraInfo(const char* extra_infos,
                                                    InnerIdType count,
                                                    InnerIdType* idx) {
    if (idx == nullptr) {
        layout_->WriteRange(total_count_, reinterpret_cast<const uint8_t*>(extra_infos), count);
        total_count_ += count;
    } else {
        for (int64_t i = 0; i < count; ++i) {
            this->InsertExtraInfo(extra_infos + extra_info_size_ * i, idx[i]);
        }
    }
}

template <typename LayoutTmpl>
bool
ExtraInfoDataCell<LayoutTmpl>::InMemory() const {
    return LayoutTmpl::InMemory;
}

template <typename LayoutTmpl>
bool
ExtraInfoDataCell<LayoutTmpl>::GetExtraInfoById(InnerIdType id, char* extra_info) const {
    return layout_->Read(id, reinterpret_cast<uint8_t*>(extra_info));
}

template <typename LayoutTmpl>
const char*
ExtraInfoDataCell<LayoutTmpl>::GetExtraInfoById(InnerIdType id, bool& need_release) const {
    return reinterpret_cast<const char*>(layout_->Read(id, need_release));
}

template <typename LayoutTmpl>
void
ExtraInfoDataCell<LayoutTmpl>::Serialize(StreamWriter& writer) {
    ExtraInfoInterface::Serialize(writer);
    this->layout_->Serialize(writer);
}

template <typename LayoutTmpl>
void
ExtraInfoDataCell<LayoutTmpl>::Deserialize(StreamReader& reader) {
    ExtraInfoInterface::Deserialize(reader);
    this->layout_->SetCodeSize(this->extra_info_size_);
    this->layout_->Deserialize(reader);
}

template <typename LayoutTmpl>
uint64_t
ExtraInfoDataCell<LayoutTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(ExtraInfoDataCell<LayoutTmpl>);
    if (LayoutTmpl::InMemory) {
        memory += this->layout_->GetMemoryUsage();
    }
    return memory;
}

template <typename LayoutTmpl>
void
ExtraInfoDataCell<LayoutTmpl>::Move(InnerIdType from, InnerIdType to) {
    this->layout_->Move(from, to);
}
}  // namespace vsag
