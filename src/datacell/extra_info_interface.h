/**
 * @file extra_info_interface.h
 * @brief Extra info interface for storing additional metadata with vectors.
 *
 * This file defines the abstract interface for storing and retrieving
 * extra information associated with vectors in the index.
 */

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

#include <string>

#include "extra_info_datacell_parameter.h"
#include "index_common_param.h"
#include "quantization/computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(ExtraInfoInterface);

/**
 * @brief Abstract interface for storing extra information associated with vectors.
 *
 * ExtraInfoInterface provides operations for storing, retrieving, and managing
 * additional metadata that can be associated with vectors in the index.
 */
class ExtraInfoInterface {
public:
    ExtraInfoInterface() = default;

    virtual ~ExtraInfoInterface() = default;

    /**
     * @brief Creates an ExtraInfoInterface instance based on parameters.
     *
     * @param param Extra info data cell configuration parameters.
     * @param common_param Common index parameters.
     * @return Shared pointer to the created ExtraInfoInterface instance.
     */
    static ExtraInfoInterfacePtr
    MakeInstance(const ExtraInfoDataCellParamPtr& param, const IndexCommonParam& common_param);

public:
    /**
     * @brief Inserts extra information for a vector.
     *
     * @param extra_info Pointer to the extra information data.
     * @param idx Internal ID for the vector (default: max value for auto-assign).
     */
    virtual void
    InsertExtraInfo(const char* extra_info,
                    InnerIdType idx = std::numeric_limits<InnerIdType>::max()) = 0;

    /**
     * @brief Inserts extra information for multiple vectors in batch.
     *
     * @param extra_infos Pointer to the extra information array.
     * @param count Number of entries to insert.
     * @param idx Optional array of internal IDs (nullptr for auto-assign).
     */
    virtual void
    BatchInsertExtraInfo(const char* extra_infos,
                         InnerIdType count,
                         InnerIdType* idx = nullptr) = 0;

    /**
     * @brief Prefetches extra information for cache optimization.
     *
     * @param id The internal ID to prefetch.
     */
    virtual void
    Prefetch(InnerIdType id) = 0;

    /**
     * @brief Resizes the storage to a new capacity.
     *
     * @param capacity The new capacity.
     */
    virtual void
    Resize(InnerIdType capacity) = 0;

    /**
     * @brief Releases resources obtained from GetExtraInfoById.
     *
     * @param extra_info Pointer to the extra information to release.
     */
    virtual void
    Release(const char* extra_info) = 0;

public:
    /**
     * @brief Gets the maximum capacity of the storage.
     *
     * @return Maximum capacity value.
     */
    InnerIdType
    GetMaxCapacity() {
        return this->max_capacity_;
    };

    /**
     * @brief Gets extra information by internal ID.
     *
     * @param id The internal ID to query.
     * @param need_release Output flag indicating if Release() must be called.
     * @return Pointer to the extra information data.
     */
    virtual const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const = 0;

    /**
     * @brief Copies extra information by internal ID.
     *
     * @param id The internal ID to query.
     * @param extra_info Output buffer for the extra information.
     * @return True if retrieval succeeded.
     */
    virtual bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const = 0;

    /**
     * @brief Gets the memory usage of the extra info storage.
     *
     * @return Memory usage in bytes.
     */
    virtual int64_t
    GetMemoryUsage() const {
        return 0;
    }

    /**
     * @brief Gets the total count of stored extra info entries.
     *
     * @return Total count of entries.
     */
    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        return this->total_count_;
    }

    /**
     * @brief Gets the size of each extra info entry.
     *
     * @return Size in bytes of each extra info entry.
     */
    [[nodiscard]] virtual uint64_t
    ExtraInfoSize() const {
        return this->extra_info_size_;
    }

    /**
     * @brief Serializes the extra info storage to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->extra_info_size_);
    }

    /**
     * @brief Deserializes the extra info storage from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->extra_info_size_);
    }

    /**
     * @brief Calculates the serialized size of the extra info storage.
     *
     * @return Size in bytes needed for serialization.
     */
    uint64_t
    CalcSerializeSize() {
        auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
        WriteFuncStreamWriter writer(calSizeFunc, 0);
        this->Serialize(writer);
        return writer.cursor_;
    }

    /**
     * @brief Checks if the data is stored in memory.
     *
     * @return True if in-memory, false otherwise.
     */
    [[nodiscard]] virtual bool
    InMemory() const = 0;

    /**
     * @brief Enables force in-memory mode for the storage.
     */
    virtual void
    EnableForceInMemory(){};

    /**
     * @brief Disables force in-memory mode for the storage.
     */
    virtual void
    DisableForceInMemory(){};

    virtual void
    Move(InnerIdType from, InnerIdType to) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Move not implemented in ExtraInfoInterface");
    }

public:
    /// Total count of stored extra info entries
    InnerIdType total_count_{0};
    /// Maximum capacity of the storage
    InnerIdType max_capacity_{0};
    /// Size of each extra info entry in bytes
    uint64_t extra_info_size_{0};
};

}  // namespace vsag