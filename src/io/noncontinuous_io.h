
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
class NonContinuousIOTest;

/**
 * @brief IO implementation for non-continuous address space storage.
 *
 * This template class wraps an inner IO implementation to manage data across
 * non-continuous address regions. It maps logical offsets to physical offsets
 * using an area table, enabling storage across fragmented memory/file regions.
 *
 * @tparam IOTmpl The underlying IO implementation type.
 */
template <typename IOTmpl>
class NonContinuousIO : public BasicIO<NonContinuousIO<IOTmpl>> {
public:
    /// Inherits InMemory property from the inner IO template type.
    static constexpr bool InMemory = IOTmpl::InMemory;

    /// Indicates deserialization is required when loading.
    static constexpr bool SkipDeserialize = false;

    /**
     * @brief Constructs a NonContinuousIO with allocator and inner IO arguments.
     *
     * @tparam Args Types of the inner IO constructor arguments.
     * @param non_continuous_allocator Allocator for managing non-continuous regions.
     * @param allocator Allocator for memory management.
     * @param args Arguments for constructing the inner IO.
     */
    template <typename... Args>
    NonContinuousIO(NonContinuousAllocator* non_continuous_allocator,
                    Allocator* allocator,
                    Args&&... args)
        : BasicIO<NonContinuousIO<IOTmpl>>(allocator),
          non_continuous_allocator_(non_continuous_allocator),
          areas_(allocator),
          inner_io_(std::make_unique<IOTmpl>(std::forward<Args>(args)...)) {
    }

    /**
     * @brief Default destructor.
     */
    ~NonContinuousIO() override = default;

    /**
     * @brief Writes data to non-continuous regions at a specified logical offset.
     *
     * Allocates new regions if needed and maps logical offset to physical offsets.
     *
     * @param data A pointer to the data to be written.
     * @param size The size of the data to be written.
     * @param offset The logical offset at which to write the data.
     */
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        auto capacity = this->get_cur_max_size();
        if (size + offset > capacity) {
            auto area = this->non_continuous_allocator_->Require(size + offset - capacity);
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
            cur_size += area_size;
            start_area++;
            if (start_area != areas_.end()) {
                start_offset = start_area->first.offset;
            }
        }
        if (offset + size > this->size_) {
            this->size_ = offset + size;
        }
    }

    /**
     * @brief Reads data from non-continuous regions at a specified logical offset.
     *
     * @param size The size of the data to be read.
     * @param offset The logical offset at which to read the data.
     * @param data A pointer to the buffer where the read data will be stored.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        bool ret = this->check_valid_offset(size + offset);
        if (not ret) {
            return ret;
        }
        auto start_area = this->get_area(offset);
        auto start_offset =
            start_area->first.offset + (offset - start_area->second + start_area->first.size);
        uint64_t cur_size = 0;
        std::vector<uint64_t> sizes;
        std::vector<uint64_t> offsets;
        while (cur_size < size) {
            auto area = start_area->first;
            auto area_size = std::min(size - cur_size, area.size - (start_offset - area.offset));
            sizes.emplace_back(area_size);
            offsets.emplace_back(start_offset);
            cur_size += area_size;
            start_area++;
            if (start_area != areas_.end()) {
                start_offset = start_area->first.offset;
            }
        }
        ret = inner_io_->MultiReadImpl(data, sizes.data(), offsets.data(), sizes.size());
        return ret;
    }

    /**
     * @brief Reads data directly from non-continuous regions, allocating a new buffer.
     *
     * @param size The size of the data to be read.
     * @param offset The logical offset at which to read the data.
     * @param need_release A reference to a boolean indicating whether the returned data needs to be released.
     * @return A pointer to the read data (always requires release).
     */
    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        bool ret = this->check_valid_offset(size + offset);
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

    /**
     * @brief Reads multiple blocks of data from non-continuous regions in a single operation.
     *
     * @param datas A pointer to a contiguous buffer where all read data will be stored sequentially.
     * @param sizes An array of sizes for each block of data to be read.
     * @param offsets An array of logical offsets for each block of data to be read.
     * @param count The number of blocks of data to be read.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        bool ret = true;
        for (uint64_t i = 0; i < count; i++) {
            ret &= this->ReadImpl(sizes[i], offsets[i], datas);
            datas += sizes[i];
        }
        return ret;
    }

    /**
     * @brief Releases data previously read via DirectReadImpl.
     *
     * @param data A pointer to the data to be released.
     */
    void
    ReleaseImpl(const uint8_t* data) const {
        this->allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

private:
    /// Type for storing cumulative size at each area boundary.
    using PostSizeType = uint64_t;

    /// Friend declaration for IOArray access.
    friend IOArray<NonContinuousIO<IOTmpl>>;

    /// Friend declaration for test access.
    friend class NonContinuousIOTest<IOTmpl>;

    /**
     * @brief Maps a logical offset to a physical offset in the inner IO.
     *
     * @param offset The logical offset to map.
     * @return The corresponding physical offset.
     */
    uint64_t
    mapping_offset(uint64_t offset) const {
        auto it = this->get_area(offset);
        uint64_t start_size = it->second - it->first.size;
        return it->first.offset + (offset - start_size);
    }

    /**
     * @brief Finds the area containing a given logical offset.
     *
     * @param offset The logical offset to find.
     * @return Iterator to the area containing the offset.
     */
    auto
    get_area(uint64_t offset) const {
        return std::upper_bound(
            areas_.begin(),
            areas_.end(),
            offset,
            [](uint64_t offset, const std::pair<NonContinuousArea, PostSizeType>& area) {
                return offset < area.second;
            });
    }

    /**
     * @brief Returns the current maximum logical size across all areas.
     *
     * @return The maximum logical size, or 0 if no areas exist.
     */
    inline PostSizeType
    get_cur_max_size() const {
        if (areas_.empty()) {
            return 0;
        }
        return areas_.back().second;
    }

private:
    /// Allocator for managing non-continuous address regions.
    NonContinuousAllocator* const non_continuous_allocator_{nullptr};

    /// The underlying IO implementation for actual storage.
    std::unique_ptr<IOTmpl> inner_io_{nullptr};

    /// Mapping table of (area, cumulative_size) pairs for offset translation.
    Vector<std::pair<NonContinuousArea, PostSizeType>> areas_;
};
}  // namespace vsag
