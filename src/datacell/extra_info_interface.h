
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

class ExtraInfoLease {
public:
    ExtraInfoLease() = default;

    ~ExtraInfoLease();

    ExtraInfoLease(const ExtraInfoLease&) = delete;
    ExtraInfoLease&
    operator=(const ExtraInfoLease&) = delete;

    ExtraInfoLease(ExtraInfoLease&& other) noexcept;
    ExtraInfoLease&
    operator=(ExtraInfoLease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const {
        return data_ != nullptr;
    }

    [[nodiscard]] const char*
    Data() const {
        return data_;
    }

private:
    friend class ExtraInfoInterface;

    ExtraInfoLease(const ExtraInfoInterface* source, const char* data, bool needs_release) noexcept
        : source_(source), data_(data), needs_release_(needs_release) {
    }

    void
    Reset() noexcept;

    const ExtraInfoInterface* source_{nullptr};
    const char* data_{nullptr};
    bool needs_release_{false};
};

class ExtraInfoInterface {
public:
    ExtraInfoInterface() = default;

    virtual ~ExtraInfoInterface() = default;

    static ExtraInfoInterfacePtr
    MakeInstance(const ExtraInfoDataCellParamPtr& param, const IndexCommonParam& common_param);

public:
    virtual void
    InsertExtraInfo(const char* extra_info,
                    InnerIdType idx = std::numeric_limits<InnerIdType>::max()) = 0;

    virtual void
    BatchInsertExtraInfo(const char* extra_infos,
                         InnerIdType count,
                         InnerIdType* idx = nullptr) = 0;

    virtual void
    Prefetch(InnerIdType id) = 0;

    virtual void
    Resize(InnerIdType capacity) = 0;

    virtual void
    Release(const char* extra_info) const = 0;

public:
    InnerIdType
    GetMaxCapacity() {
        return this->max_capacity_;
    };

    virtual const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const = 0;

    [[nodiscard]] ExtraInfoLease
    AcquireExtraInfoById(InnerIdType id) const {
        bool needs_release = false;
        const char* data = this->GetExtraInfoById(id, needs_release);
        return ExtraInfoLease(this, data, needs_release);
    }

    virtual bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const = 0;

    virtual uint64_t
    GetMemoryUsage() const {
        return 0;
    }

    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        return this->total_count_;
    }

    [[nodiscard]] virtual uint64_t
    ExtraInfoSize() const {
        return this->extra_info_size_;
    }

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->extra_info_size_);
    }

    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->extra_info_size_);
    }

    uint64_t
    CalcSerializeSize() {
        auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
        WriteFuncStreamWriter writer(calSizeFunc, 0);
        this->Serialize(writer);
        return writer.cursor_;
    }

    [[nodiscard]] virtual bool
    InMemory() const = 0;

    virtual void
    EnableForceInMemory(){};

    virtual void
    DisableForceInMemory(){};

    virtual void
    Move(InnerIdType from, InnerIdType to) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Move not implemented in ExtraInfoInterface");
    }

    virtual void
    ShrinkToFit(InnerIdType /*capacity*/) {
    }

public:
    InnerIdType total_count_{0};
    InnerIdType max_capacity_{0};
    uint64_t extra_info_size_{0};
};

inline ExtraInfoLease::~ExtraInfoLease() {
    Reset();
}

inline ExtraInfoLease::ExtraInfoLease(ExtraInfoLease&& other) noexcept
    : source_(other.source_), data_(other.data_), needs_release_(other.needs_release_) {
    other.source_ = nullptr;
    other.data_ = nullptr;
    other.needs_release_ = false;
}

inline ExtraInfoLease&
ExtraInfoLease::operator=(ExtraInfoLease&& other) noexcept {
    if (this != &other) {
        Reset();
        source_ = other.source_;
        data_ = other.data_;
        needs_release_ = other.needs_release_;
        other.source_ = nullptr;
        other.data_ = nullptr;
        other.needs_release_ = false;
    }
    return *this;
}

inline void
ExtraInfoLease::Reset() noexcept {
    if (needs_release_ and source_ != nullptr and data_ != nullptr) {
        source_->Release(data_);
    }
    source_ = nullptr;
    data_ = nullptr;
    needs_release_ = false;
}

}  // namespace vsag
