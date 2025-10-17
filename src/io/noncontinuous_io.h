
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
#include "io_array.h"
#include "noncontinuous_allocator.h"

namespace vsag {

class IndexCommonParam;
class Allocator;

template <typename IOTmpl>
class NonContiguousIO : public BasicIO<NonContiguousIO<IOTmpl>> {
public:
    static constexpr bool InMemory = IOTmpl::InMemory;
    static constexpr bool SkipDeserialize = false;

    ~NonContiguousIO() override = default;

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        auto capacity = this->get_cur_max_size();
        if (size + offset > capacity) {
            auto area = this->non_contiguous_allocator_->Require(size + offset - capacity);
            areas_.emplace_back(area, capacity + area.size);
        }
        auto start_area = this->get_area(offset);
        auto start_offset =
            start_area->first.offset + (offset - start_area->second + start_area->first.size);
        uint64_t cur_size = 0;
        while (cur_size < size) {
            auto area = start_area->first;
            auto area_size = std::min(size - cur_size, area.size - (start_offset - area.offset));
            inner_io_->WriteImpl(data + cur_size, area_size, start_offset);
            start_area++;
            start_offset = start_area->first.offset;
            cur_size += area_size;
        }
        if (offset + size > this->size_) {
            this->size_ = offset + size;
        }
    }

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        bool ret = this->check_valid_size(size + offset);
        if (not ret) {
            return ret;
        }
        auto start_area = this->get_area(offset);
        auto start_offset =
            start_area->first.offset + (offset - start_area->second + start_area->first.size);
        uint64_t cur_size = 0;
        std::vector<uint64_t> sizes;
        std::vector<uint64_t> offsets;
        std::vector<uint8_t*> datas;
        while (cur_size < size) {
            auto area = start_area->first;
            auto area_size = std::min(size - cur_size, area.size - (start_offset - area.offset));
            sizes.emplace_back(area_size);
            offsets.emplace_back(start_offset);
            datas.emplace_back(data + cur_size);
            start_area++;
            start_offset = start_area->first.offset;
            cur_size += area_size;
        }
        ret = inner_io_->MultiReadImpl(datas.data(), sizes.data(), offsets.data(), sizes.size());
        return ret;
    }

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        bool ret = this->check_valid_size(size + offset);
        if (not ret) {
            return nullptr;
        }
        auto* data = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(size));
        ret = this->ReadImpl(size, offset, data);
        if (not ret) {
            this->allocator_->Deallocate(data);
            return nullptr;
        }
        need_release = true;
        return data;
    }

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        bool ret = true;
        for (uint64_t i = 0; i < count; i++) {
            ret &= this->ReadImpl(sizes[i], offsets[i], datas[i]);
        }
        return ret;
    }

    bool
    ReleaseImpl(const uint8_t* data) {
        return this->inner_io_->Release(data);
    }

private:
    template <typename... Args>
    NonContiguousIO(NonContiguousAllocator* non_contiguous_allocator,
                    Allocator* allocator,
                    Args&&... args)
        : BasicIO<NonContiguousIO<IOTmpl>>(allocator),
          non_contiguous_allocator_(non_contiguous_allocator),
          areas_(allocator),
          inner_io_(std::make_unique<IOTmpl>(std::forward<Args>(args)..., allocator)) {
    }

    friend IOArray<NonContiguousIO<IOTmpl>>;

    uint64_t
    mapping_offset(uint64_t offset) const {
        auto it = this->get_area(offset);
        uint64_t start_size = it->second - it->first.size;
        return it->first.offset + (offset - start_size);
    }

    auto
    get_area(uint64_t offset) const {
        return std::upper_bound(
            areas_.begin(),
            areas_.end(),
            offset,
            [](uint64_t offset, const std::pair<NonContiguousArea, PostSizeType>& area) {
                return offset < area.second;
            });
    }

    inline PostSizeType
    get_cur_max_size() const {
        return areas_.back().second;
    }

private:
    NonContiguousAllocator* const non_contiguous_allocator_{nullptr};

    std::unique_ptr<IOTmpl> inner_io_{nullptr};

    using PostSizeType = uint64_t;
    Vector<std::pair<NonContiguousArea, PostSizeType>> areas_;
};
}  // namespace vsag
