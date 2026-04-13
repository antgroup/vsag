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

/**
 * @file extra_info_datacell.h
 * @brief Extra info data cell implementation for storing additional metadata.
 *
 * This file provides the ExtraInfoDataCell class which implements the
 * ExtraInfoInterface for storing and retrieving additional metadata
 * associated with vectors.
 */

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

/**
 * @brief Extra info data cell for storing additional metadata associated with vectors.
 *
 * This class implements ExtraInfoInterface and provides functionality for:
 * - Storing arbitrary metadata with fixed size for each vector
 * - Batch insertion and retrieval of extra info
 * - IO-backed storage with memory and disk options
 *
 * @warning This class is thread-unsafe. External synchronization is required
 *          for concurrent access.
 *
 * @tparam IOTmpl The IO template type for storage operations.
 */
template <typename IOTmpl>
class ExtraInfoDataCell : public ExtraInfoInterface {
public:
    ExtraInfoDataCell() = default;

    /**
     * @brief Constructs an ExtraInfoDataCell with IO parameters.
     * @param io_param The IO parameters.
     * @param common_param The common index parameters.
     */
    explicit ExtraInfoDataCell(const IOParamPtr& io_param, const IndexCommonParam& common_param);

    /**
     * @brief Inserts extra info for a vector.
     * @param extra_info Pointer to the extra info data.
     * @param idx The internal ID for the extra info.
     */
    void
    InsertExtraInfo(const char* extra_info, InnerIdType idx) override;

    /**
     * @brief Inserts multiple extra info entries in batch.
     * @param extra_infos Pointer to the extra info data array.
     * @param count Number of extra info entries to insert.
     * @param idx Array of internal IDs (nullptr for auto-assignment).
     */
    void
    BatchInsertExtraInfo(const char* extra_infos, InnerIdType count, InnerIdType* idx) override;

    /**
     * @brief Prefetches extra info for cache optimization.
     * @param id The internal ID to prefetch.
     */
    void
    Prefetch(InnerIdType id) override {
        io_->Prefetch(id * extra_info_size_, extra_info_size_);
    };

    /**
     * @brief Resizes the data cell capacity.
     * @param new_capacity The new capacity size.
     */
    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        uint64_t io_size =
            static_cast<uint64_t>(new_capacity) * static_cast<uint64_t>(extra_info_size_);
        this->io_->Resize(io_size);
        this->max_capacity_ = new_capacity;
    }

    /**
     * @brief Releases the extra info data obtained from GetExtraInfoById.
     * @param extra_info Pointer to the extra info data to release.
     */
    void
    Release(const char* extra_info) override {
        if (extra_info == nullptr) {
            return;
        }
        io_->Release(reinterpret_cast<const uint8_t*>(extra_info));
    }

    /**
     * @brief Checks if the data is stored in memory.
     * @return True if stored in memory, false otherwise.
     */
    [[nodiscard]] bool
    InMemory() const override;

    /**
     * @brief Gets the extra info for a given ID and writes to output buffer.
     * @param id The internal ID.
     * @param extra_info Output buffer for the extra info.
     * @return True if successful, false otherwise.
     */
    bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const override;

    /**
     * @brief Gets the extra info for a given ID with release flag.
     * @param id The internal ID.
     * @param need_release Output flag indicating if the returned pointer needs release.
     * @return Pointer to the extra info data.
     */
    const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const override;

    /**
     * @brief Serializes the data cell to a stream.
     * @param writer The stream writer for output.
     */
    void
    Serialize(StreamWriter& writer) override;

    /**
     * @brief Deserializes the data cell from a stream.
     * @param reader The stream reader for input.
     */
    void
    Deserialize(StreamReader& reader) override;

    /**
     * @brief Gets the memory usage of this data cell.
     * @return The memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override;

    /**
     * @brief Sets the IO instance.
     * @param io The shared pointer to the IO instance.
     */
    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

public:
    /// IO instance for storage operations
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    /// Allocator for memory management
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
