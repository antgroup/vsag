
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
#include "index/index_common_param.h"
#include "quantization/computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"

namespace vsag {
class ExtraInfoInterface;
using ExtraInfoInterfacePtr = std::shared_ptr<ExtraInfoInterface>;

class ExtraInfoInterface {
public:
    ExtraInfoInterface() = default;

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
    Release(const char* extra_info) = 0;

public:
    InnerIdType
    GetMaxCapacity() {
        return this->max_capacity_;
    };

    virtual const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const = 0;

    virtual bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const = 0;

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

public:
    InnerIdType total_count_{0};
    InnerIdType max_capacity_{0};
    uint64_t extra_info_size_{0};
};

}  // namespace vsag
